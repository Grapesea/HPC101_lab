# RISC-V S2场景热点分析与优化记录

## 1. 问题特征

S2的形状是：

```text
N=1, D=1024, H=512, E=16, K=4
```

每次前向需要计算1个shared expert和4个routed expert。一个expert包含Gate、Up和Down
三个int8投影，因此主要MAC数为：

$$
5\times(1024\times512+1024\times512+512\times1024)
=7,864,320
$$

S2只有一个token，而IME的`vmadot`固定计算$4\times4\times8$ tile，所以M方向只有
一行有效。另一方面，5个expert的三个投影需要读取大约7.5 MiB权重。它既有单token
矩阵利用率问题，也容易受到权重带宽和指令load-to-use延迟限制，不能直接套用S3的
多token双输出内核。

## 2. 单独采集S2热点

使用当前IME代码重新构建`-pg`版本，只运行S2优化路径1000轮：

```bash
./build-rv-gprof/lab2 1 1024 512 16 4 1000 --benchmark
gprof -b ./build-rv-gprof/lab2 gmon.out
```

得到的主要结果如下。`init_tokens`和`preprocess`发生在计时区外，不作为优化目标；
OpenMP下gprof不能完整合并每个worker的调用图，所以主要使用热点排序而不是绝对时间。

| 热点 | flat profile占比 | 含义 |
| ---- | ---------------- | ---- |
| `ime_vmadot_tile` | 63.25% | 单token Gate/Up/Down的IME K循环 |
| `ime_gemm_packed_single`调用子树 | 约16.4% | output tile遍历和IME leaf调用 |
| `rvv_expert_ffn` self | 1.53% | SwiGLU、量化和最终写回 |
| `rvv_prepare_block` | 0.22% | Router、Top-K和输入量化 |

这个结果说明Router已经不是S2问题。继续近似sigmoid或优化Top-K最多只能影响零点几个
百分点，正确方向应当是单token投影和`vmadot` leaf本身。

## 3. RVV/IME混合内核实验

首先验证“IME浪费3个M行，因此RVV单token内核可能更快”这个判断。代码中原有的RVV
packed weight仍然可用，所以分别测试Gate/Up和Down使用RVV，其他部分保持不变。

300轮、同一作业中的三次结果为：

| Gate/Up | Down | 三次优化时间 | 中位数 |
| ------- | ---- | ------------ | ------ |
| IME | IME | 0.493 / 0.471 / 0.471 s | 0.471 s |
| RVV | IME | 0.511 / 0.523 / 0.531 s | 0.523 s |
| IME | RVV | 0.531 / 0.503 / 0.500 s | 0.503 s |
| RVV | RVV | 0.537 / 0.520 / 0.519 s | 0.520 s |

虽然IME在M方向只有25%结果有效，但一条`vmadot`仍然比RVV的`vle8 + vsext + vwmacc`
指令流更有优势。Gate/Up改RVV约回退11%，Down改RVV约回退7%，因此最终继续使用全IME。
调优用的环境变量和动态分派在实验后全部删除，没有留在提交路径中。

## 4. shared expert并行化为什么失败

S2当前先串行计算shared expert，再用4个OpenMP worker计算4个routed expert。看起来可以
把shared的output tile分给4核。我分别实现了两版专用路径：

1. 4核并行执行shared的IME Gate/Up和Down，SwiGLU作为中间串行段；
2. shared投影改用16-output RVV微内核并按output block并行。

两种版本都正确，但300轮时间分别约为`0.924--0.985 s`和`0.915--0.977 s`，几乎是
原实现的两倍。原因不是数学工作量，而是额外创建了一个很短的OpenMP region，并在
Gate/Up、SwiGLU、Down之间引入多次barrier；同时四核并发读取同一个shared expert的
大权重流，使IME执行资源和内存带宽竞争加重。原来的routed并行已经提供了最合适的
粗粒度，shared内部再嵌一层细粒度并行不划算，因此完整撤销。

## 5. K循环展开因子实验

原IME leaf每8个K-block执行一次循环控制。S2的Gate/Up有128个K-block，Down有64个，
所以又测试了展开4、8、16：

| 对比 | 原8展开中位数 | 候选中位数 | 结论 |
| ---- | ------------- | ---------- | ---- |
| 8 vs 16 | 0.4743 s | 0.4761 s | 16展开代码更大，没有收益 |
| 8 vs 4 | 约0.477 s | 约0.479 s | 4展开增加分支，也没有收益 |

因此热点并不是循环分支本身，8展开是K60前端代码体积和循环控制之间较好的平衡。

## 6. 最终保留：双输入寄存器软件流水

检查内联汇编后发现，原来的每个K-block严格执行：

```text
vle8 A -> vle8 B -> vmadot -> 指针递增
```

`vmadot`紧跟在两次load后面，可能直接暴露向量load-to-use延迟。S2专用leaf仍保留一个
`v28:v29` int32m2累加器，但使用两组输入寄存器交替：

```text
load block 0 -> v0/v1
load block 1 -> v2/v3
vmadot v28, v0, v1
load block 2 -> v0/v1
vmadot v28, v2, v3
...
```

节点的assembler不认识厂商助记符，所以第二组输入的指令同样使用编码：

```asm
.word 0xe2103e2b    # vmadot v28, v0, v1
.word 0xe2313e2b    # vmadot v28, v2, v3
```

它没有增加矩阵指令、权重读取或累加器数量，只是把下一K块的load提前到当前
`vmadot`之前，让load延迟和矩阵计算重叠。该leaf仅在运行时形状精确匹配S2且
`t_count==1`时使用，其他形状继续走原来的8展开内核。

同一二进制、同一节点交替运行五组A/B：

| 内核 | 五次时间（300轮） | 中位数 |
| ---- | ----------------- | ------ |
| 原8展开 | 0.478 / 0.471 / 0.475 / 0.479 / 0.472 s | 0.47536 s |
| 软件流水 | 0.458 / 0.466 / 0.453 / 0.458 / 0.478 s | 0.45847 s |

中位时间降低约3.55%。这也与profile结论一致：收益来自占比63%的IME leaf，而不是
只占0.22%的Router。

## 7. 最终验证

在`riscv-rv03`节点上完整运行baseline和optimized，S2使用300轮，连续三次结果为：

| 轮次 | Baseline | Optimized | Speedup | rel RMSE |
| ---- | -------- | --------- | ------- | -------- |
| 1 | 9.14785 s | 0.460424 s | 19.8683x | 3.12946e-4 |
| 2 | 9.21083 s | 0.463990 s | 19.8514x | 3.12946e-4 |
| 3 | 9.17875 s | 0.463097 s | 19.8203x | 3.12946e-4 |

三轮最低值仍高于18.5x目标。其他样例回归也全部通过：S1、S3、S4的optimized时间
分别为`0.0560 s / 500`、`0.4448 s / 200`、`0.2903 s / 5`，对应RMSE约为
`2.46e-5`、`7.67e-4`和`1.06e-3`。

最终处理逻辑可以概括为：先用S2自己的profile排除Router，再用投影级A/B验证RVV/IME，
通过负优化确认并行粒度和展开因子，最后在真正的leaf热点中用软件流水隐藏load延迟。
每项实验都在同一节点重复并检查正确性，只保留能够稳定降低中位时间的改动。
