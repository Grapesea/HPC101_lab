# 将 Token 批量矩阵化并用矩阵运算库加速 MoE

## 1. 核心思路

将一批 $N$ 个、每个维度为 $D$ 的 token 按行存放：

$$
X\in\mathbb{R}^{N\times D}.
$$

标量参考实现以 token 为中心，反复执行矩阵向量乘（GEMV）。矩阵运算库更擅长矩阵矩阵乘（GEMM），所以优化的关键不是简单地把外层 token 循环交给库，而是：

1. 用一次 GEMM 计算整个 batch 的 Router。
2. 完成 Top-K 后，把 token 按其选中的专家分桶（grouped/segmented dispatch）。
3. 让每个专家一次处理分给它的多个 token，将若干 GEMV 变成一个 GEMM。
4. 将专家结果按原 token 编号 scatter 回输出矩阵并加权求和。

共享专家天然接收全部 $N$ 个 token，不需要分桶，可以直接批量计算。路由专家接收的 token 数不相同，因此整个 MoE 不是一个规则的稠密 GEMM，而是一组行数不同的 GEMM。

## 2. Router：直接使用 FP32 GEMM

本实验的 Router 权重为：

$$
W_r\in\mathbb{R}^{E\times D}.
$$

所有 token 的 Router logit 可以写成：

$$
Z=XW_r^T,\qquad Z\in\mathbb{R}^{N\times E}.
$$

这可以直接调用 FP32 GEMM，例如 BLAS 的 `sgemm`。按当前行优先布局，可在 API 中把右矩阵设为转置，而不必真的在前向阶段转置权重。若库的打包接口支持固定权重预打包，也可以在 `preprocess` 阶段将 $W_r^T$ 打包成库的内部格式。

GEMM 之后仍需逐元素执行 sigmoid，并逐行完成 Top-K：

$$
S=\operatorname{sigmoid}(Z).
$$

选择专家时比较 $S_{t,e}+b_e$，但归一化 gate 时只能使用原始的 $S_{t,e}$。Top-K 是小规模、不规则操作，本实验中 $K\leq4$，通常没有必要强行转换成矩阵乘法。

## 3. 输入量化：矩阵存储、逐行 scale

虽然输入现在是 $N\times D$ 矩阵，量化规则仍是 per-token，也就是每一行有自己的 scale：

$$
s_{x,t}=\frac{\max_d|X_{t,d}|}{127},
\qquad
X_{q,t,d}=\operatorname{round}\left(\frac{X_{t,d}}{s_{x,t}}\right).
$$

因此得到：

$$
X_q\in\mathbb{Z}_8^{N\times D},
\qquad
s_x\in\mathbb{R}^{N}.
$$

这里不能为整个矩阵只计算一个 scale，否则会改变实验规定的数值语义。求每行最大值和量化通常用库的逐元素/归约原语，或者直接用 SIMD 并行循环；它们不属于 GEMM 热点。

## 4. 按专家分桶：把稀疏路由变成小批量矩阵

设 token $t$ 的第 $k$ 个被选专家为 $I_{t,k}$，对应 gate 为 $G_{t,k}$。可以先统计每个专家接收多少个 token：

$$
M_e=\left|\{(t,k)\mid I_{t,k}=e\}\right|.
$$

所有路由专家的任务总数满足：

$$
\sum_{e=0}^{E-1}M_e=NK.
$$

用计数加前缀和可建立连续分桶缓冲区。每条分桶记录至少保留：

- 量化后的 token 行或其行号；
- 原 token 编号 $t$，用于最后 scatter；
- 对应 gate $G_{t,k}$；
- 该 token 的输入 scale $s_{x,t}$。

对专家 $e$，将其 token 紧凑地组成：

$$
X_q^{(e)}\in\mathbb{Z}_8^{M_e\times D}.
$$

同一个 token 会进入 $K$ 个路由专家的桶，因此逻辑上会出现 $K$ 次。为了让 GEMM 顺序读取，实际实现通常复制 token 行到连续缓冲区；也可以只保存索引后再 gather，但随机访问可能降低缓存与矩阵内核效率。

## 5. 专家 FFN：三个投影改写成 INT8 GEMM

对专家 $e$，原始权重布局为：

$$
W_g^{(e)},W_u^{(e)}\in\mathbb{Z}_8^{H\times D},
\qquad
W_d^{(e)}\in\mathbb{Z}_8^{D\times H}.
$$

### 5.1 Gate 和 Up 投影

分桶后可以进行两个 INT8 GEMM，并用 INT32 累加：

$$
A_g^{(e)}=X_q^{(e)}(W_g^{(e)})^T
\in\mathbb{Z}_{32}^{M_e\times H},
$$

$$
A_u^{(e)}=X_q^{(e)}(W_u^{(e)})^T
\in\mathbb{Z}_{32}^{M_e\times H}.
$$

还可以在预处理时沿输出维拼接 Gate 和 Up 权重：

$$
W_{gu}^{(e)}=
\begin{bmatrix}W_g^{(e)}\\W_u^{(e)}\end{bmatrix}
\in\mathbb{Z}_8^{2H\times D},
$$

然后只调用一次：

$$
A_{gu}^{(e)}=X_q^{(e)}(W_{gu}^{(e)})^T
\in\mathbb{Z}_{32}^{M_e\times2H}.
$$

这样可以让输入块在缓存或 Tile 中被两种投影复用，并减少矩阵内核调用与调度开销。

INT32 结果需要按行乘输入 scale、按投影乘权重 scale：

$$
V_{g,t,f}=A_{g,t,f}\,s_{x,t}s_g^{(e)},
\qquad
V_{u,t,f}=A_{u,t,f}\,s_{x,t}s_u^{(e)}.
$$

接着逐元素计算：

$$
H_{t,f}=\operatorname{SiLU}(V_{g,t,f})V_{u,t,f}.
$$

SwiGLU 是非线性的，不能和前后的 GEMM 代数合并；不过矩阵库若提供 fused post-op，可以融合反量化缩放或部分逐元素操作，减少中间矩阵的读写。

### 5.2 隐藏层重量化与 Down 投影

隐藏层仍需逐行计算 scale：

$$
s_{h,t}=\frac{\max_f|H_{t,f}|}{127},
\qquad
H_{q,t,f}=\operatorname{round}\left(\frac{H_{t,f}}{s_{h,t}}\right).
$$

然后执行：

$$
A_d^{(e)}=H_q^{(e)}(W_d^{(e)})^T
\in\mathbb{Z}_{32}^{M_e\times D},
$$

$$
O_{t,d}^{(e)}=A_{d,t,d}^{(e)}s_{h,t}s_d^{(e)}.
$$

至此，一个专家原来的逐 token GEMV 已变为尺寸分别为 $M_e\times D$ 乘 $D\times2H$，以及 $M_e\times H$ 乘 $H\times D$ 的两次主要 GEMM。

## 6. 共享专家：最规整、最适合先矩阵化

共享专家处理所有 token，所以直接令 $M_{shared}=N$：

$$
A_{gu}^{(shared)}=X_q(W_{gu}^{(shared)})^T,
$$

$$
A_d^{(shared)}=H_q^{(shared)}(W_d^{(shared)})^T.
$$

它不需要统计、前缀和、复制或 scatter，是最容易接入矩阵库的部分。即使路由专家分布很不均匀，共享专家的 GEMM 形状也始终规则。

## 7. 输出合并：按原 token 编号 scatter-add

输出先放入残差和共享专家结果：

$$
Y=X+O^{(shared)}.
$$

随后遍历每个专家桶中的结果，根据保存的原 token 编号和 gate 执行：

$$
Y_{t,:}\mathrel{+}=G_{t,k}O_{e,j,:}.
$$

这一步是带索引的加权 scatter-add，不是普通 GEMM。可以在 $D$ 维上使用 SIMD。多线程按专家并行时，两个专家可能同时更新同一 token，必须避免数据竞争。常见方案是：

- 专家输出先写入独立缓冲区，最后按 token 统一归约；
- 每个线程使用私有输出缓冲区，最后合并；
- 或者在专家计算完成后改为按 token 并行，每个 token 读取自己的 $K$ 份结果。

最后一种方式通常最简单，因为 $K$ 很小且每个输出行只由一个线程写入。

## 8. 可选矩阵库与数据类型

在 CPU 上可考虑以下接口：

| 计算 | 合适的库接口 | 输入与累加 |
| --- | --- | --- |
| Router | BLAS/oneMKL `sgemm`、oneDNN MatMul | FP32 × FP32 → FP32 |
| 专家投影 | oneDNN MatMul、oneMKL INT8 GEMM | INT8 × INT8 → INT32 |
| 多个不同形状的专家 | grouped/batched GEMM，或逐专家调用 MatMul primitive | 每个专家有自己的 $M_e$ 和权重 |
| SiLU、scale、量化 | oneDNN post-op/逐元素原语，或手写 SIMD | INT32/FP32/INT8 混合 |

普通 BLAS 主要面向浮点 GEMM；是否支持“有符号 INT8 × 有符号 INT8 → INT32”以及所需的零点语义，要检查具体接口。不能为了套用某个只支持 `u8 × s8` 的接口而直接把 `int8_t` 解释成 `uint8_t`，除非同时正确处理加 128 后引入的补偿项。对本实验的 W8A8 语义，能明确支持 `s8 × s8 → s32` 的库或自定义 AMX/VNNI 内核更稳妥。

矩阵库还必须实际出现在构建和评测环境中。当前工程只显式链接 OpenMP，并未链接 oneMKL 或 oneDNN；如果评测只接收 `student/` 下的代码，就不能假设修改 `CMakeLists.txt` 后新增的依赖会被保留。因此，在本实验中，矩阵化的数据组织方法仍然适用，但底层 GEMM 可能需要由 AVX-512/VNNI/AMX intrinsic 实现，而不是依赖额外动态库。

## 9. 预处理、布局与线程策略

### 权重预打包

权重跨多次前向保持不变，应在计时外的 `preprocess` 中完成：

- 将 Gate 和 Up 权重拼接；
- 将逻辑上的转置或分块布局提前生成；
- 按矩阵库要求对齐行首和块边界；
- 使用库提供的 packed-weight/primitive cache，或生成 AMX/VNNI 友好的布局。

不要在每次前向中重新转置或打包全部专家权重，否则数据重排成本可能抵消 GEMM 的收益。

### 行优先布局

本实验中 token 和权重都按行优先保存。数学表达式使用 $XW^T$，实现时通常有两种选择：

1. 调用 GEMM 时把权重操作数标记为转置；
2. 在预处理阶段保存适合内核读取的预打包版本。

第二种方式常有更好的稳定性能，但预打包后的权重与原始逻辑布局不同，索引和尾块处理必须严格验证。

### 避免线程过量

矩阵库通常自己创建线程。如果外层又用 OpenMP 并行专家，容易出现“OpenMP 线程数 × 库线程数”的过量线程。一般选择一种主要策略：

- 外层串行，单个大 GEMM 由矩阵库多线程；
- 外层按专家多线程，每个小 GEMM 强制库单线程；
- 使用库的 grouped GEMM，由库统一调度多个专家任务。

专家负载 $M_e$ 不均匀时，简单的静态专家划分可能失衡。可以按 $M_eDH$ 估算任务大小后动态调度，但任务过小时应合并调度，避免每个空桶或小桶都产生一次昂贵的库调用。

## 10. 什么时候矩阵化会加速，什么时候不会

矩阵化的收益来自同一份专家权重被多个 token 复用，以及高性能 GEMM 内核对缓存、SIMD、VNNI 或 AMX 的充分利用。当 $M_e$ 较大时，收益最明显。

本实验四种形状下可作如下判断：

| 场景 | 矩阵库策略 |
| --- | --- |
| S1、S2：$N=1$ | 每个专家只有一个 token，本质仍是 GEMV；分桶和库调用开销可能更大，优先使用专门的 SIMD/VNNI GEMV 内核 |
| S3：$N=128,E=16,K=4$ | 平均每个路由专家约有 $NK/E=32$ 个任务，非常适合分桶后做 INT8 GEMM；共享专家直接做 $128$ 行 GEMM |
| S4：$N=1024,E=512,K=2$ | 平均每个专家仅约有 $4$ 个任务，Router GEMM 很大，但路由专家 GEMM 很“瘦”；需要 grouped GEMM、专家任务并行、小矩阵内核和负载均衡 |

因此应保留按形状分派的能力。例如 $N=1$ 或 $M_e$ 小于经验阈值时走 GEMV/小矩阵内核，$M_e$ 足够大时再走通用 GEMM。阈值需要在目标机器上实测，而不能只根据理论 FLOPS 决定。

## 11. 推荐的整体流水线

```text
X [N,D] (FP32)
  ├─ Router GEMM: X · Wr^T -> Z [N,E]
  │    └─ sigmoid -> biased Top-K -> expert_id [N,K], gate [N,K]
  │
  └─ per-row quantize -> Xq [N,D], sx [N]
       ├─ shared expert batched INT8 GEMM
       │    Xq · Wgu_shared^T -> SwiGLU -> per-row requantize
       │    Hq · Wd_shared^T -> O_shared [N,D]
       │
       └─ dispatch/group by expert
            for expert e with Me > 0:
              Xq_e · Wgu_e^T -> SwiGLU -> per-row requantize
              Hq_e · Wd_e^T -> O_e [Me,D]

Y = X + O_shared
按 token 对 K 份 routed output 做 gate 加权并 scatter/reduce 到 Y
```

从端到端角度看，真正需要同时优化的是 Router、量化、dispatch、专家 GEMM 和 combine。只把专家点积替换成库调用，却在每个 token 上单独调用 GEMM，通常无法充分利用矩阵库，甚至可能因函数调用、权重重复打包和线程调度而变慢。

## 12. 正确性检查要点

矩阵化会改变循环和 FP32 求和顺序，允许出现容差内的浮点差异，但必须保持以下语义：

- Top-K 使用 `sigmoid(logit) + bias`，相同分数时仍优先较小专家编号；
- gate 使用未加 bias 的亲和度归一化；
- 输入和 SwiGLU 隐藏层均为逐 token、逐行 scale；
- INT8 乘法使用 INT32 累加；
- 每个权重矩阵使用各自的 scale；
- 共享专家不乘路由 gate；
- 同一个 token 被复制到多个专家桶后，必须携带正确的 scale、gate 和原 token 编号；
- $M_e=0$ 的专家应跳过，非整块的 $M_e$、$D$、$H$ 尾部需要正确处理；
- scatter/reduce 不能发生并发写冲突。

概括来说，矩阵库能加速 MoE 的前提是先把“token 依次调用专家”改造成“专家批量处理 token”。Router 和共享专家可以直接矩阵化；路由专家则必须先分桶，再把每个专家的若干矩阵向量乘合并为矩阵矩阵乘。
