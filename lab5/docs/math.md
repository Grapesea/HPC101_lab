# Lab 5 数学原理总结

## 1. 自回归推理、Prefill 与 Decode

Decoder-only 语言模型把序列概率分解为

$$
p(x_1,\ldots,x_L)=\prod_{t=1}^{L}p(x_t\mid x_{<t}).
$$

推理包括两个阶段：Prefill 一次处理 prompt 中的全部 token，并生成各层的
KV Cache；Decode 每轮只输入一个新 token，复用历史 KV Cache 计算下一个
token。Prefill 通常包含较大的矩阵乘，偏计算密集；Decode 的矩阵乘退化为
矩阵向量乘，反复读取全部权重，通常更受显存带宽和 kernel 发射开销限制。

## 2. Gemma4 的注意力

缩放点积注意力为

$$
A=\frac{QK^T}{\sqrt{d_h}}+M,\qquad
P=\operatorname{softmax}(A),\qquad O=PV.
$$

因果掩码保证位置 $i$ 只能看到不晚于自身的位置。Gemma4 混合使用全局注意力
与窗口大小为 $W=1024$ 的滑动窗口注意力。后者只保留

$$
j\le i,\qquad j>i-W
$$

范围内的键值。真正高效的实现应跳过窗口外的分数计算，而不是先计算完整
$L\times L$ 分数矩阵再加掩码。

Grouped Query Attention 令多个 Query heads 共享一个 KV head。若 Query head
数为 $H_q$、KV head 数为 $H_{kv}$，则 Query head $h$ 使用

$$
h_{kv}=\left\lfloor\frac{h}{H_q/H_{kv}}\right\rfloor.
$$

Gemma4-12B 的滑窗层为 $H_q=16,H_{kv}=8$，全局层为
$H_q=16,H_{kv}=1$。GQA 不减少 Query 数量，但减少 K/V 投影、注意力计算和
KV Cache；实现时不应物化重复的 K/V。

## 3. SwiGLU 前馈网络

Gemma4 的前馈层为

$$
\operatorname{FFN}(x)=W_{down}
\left[\operatorname{SiLU}(W_{gate}x)\odot(W_{up}x)\right],
$$

其中

$$
\operatorname{SiLU}(x)=x\sigma(x).
$$

因此每层包含 gate、up、down 三个大矩阵。Decode 每一步都要读取这些权重，
所以 W4A16 量化可同时降低常驻显存和权重读取量。

## 4. 权重与 KV Cache 显存

设模型有 $N$ 层，隐藏维度 $D$，Query heads 为 $H_q$，KV heads 为
$H_{kv}$，head dimension 为 $D_h$，FFN 中间维度为 $F$，词表为 $V$。
忽略 Norm 等小参数，参数量近似为

$$
P\approx 2VD+ND(2H_qD_h+2H_{kv}D_h+3F).
$$

$x$-bit 权重占用约为

$$
M_w=P\frac{x}{8}\ \text{bytes}.
$$

12B 参数的 BF16 理论值约 24 GB，而 INT4 理论值约 6 GB；实际 INT4 checkpoint
还包含 scale、zero point、未量化参数、对齐和运行时内存。

静态 BF16 KV Cache 的一般表达式为

$$
M_{kv}=2BSNH_{kv}D_h\frac{16}{8}\ \text{bytes},
$$

其中 2 表示 K 和 V。对于混合注意力，更准确地写为

$$
M_{kv}=2B\frac{y}{8}
\left(N_lS H_{l,kv}D_{l,h}+N_gS H_{g,kv}D_{g,h}\right).
$$

滑窗层采用 ring cache 后变为

$$
M_{kv}^{ring}=2B\frac{y}{8}
\left(N_l\min(S,W)H_{l,kv}D_{l,h}
+N_gS H_{g,kv}D_{g,h}\right).
$$

## 5. INT4 线性量化

实数权重 $r$ 和整数 code $q$ 的映射为

$$
q=\operatorname{clip}\left(\operatorname{round}\left(\frac r s+z\right),
q_{min},q_{max}\right),\qquad
\hat r=s(q-z).
$$

本框架每个输出通道、每 `group_size` 个输入列共享一组量化参数。对称 INT4
使用有符号值 $[-8,7]$，checkpoint 中编码为 $[0,15]$，反量化为

$$
\hat r=s(q_{code}-8),\qquad
s=\frac{\max |r|}{7}.
$$

非对称量化使用

$$
s=\frac{r_{max}-r_{min}}{15},\qquad
z=\operatorname{round}\left(-\frac{r_{min}}s\right).
$$

一个 `uint8` 的低、高清半字节依次保存两个 INT4 code。输入维度补齐到 group
size 的整数倍；补齐位置必须编码成反量化后的零。

## 6. 从 RTN 到 GPTQ

RTN 只最小化权重数值误差 $\lVert W-Q\rVert$，没有考虑输入分布。设校准
激活为 $X\in\mathbb R^{N\times d_{in}}$，线性层权重为
$W\in\mathbb R^{d_{out}\times d_{in}}$，更有意义的层重构目标是

$$
\min_Q\lVert XW^T-XQ^T\rVert_F^2
=\min_Q\lVert X(W-Q)^T\rVert_F^2.
$$

忽略不影响优化的常数，其 Hessian 为

$$
H=\frac{2}{N}X^TX.
$$

$H$ 的对角项表示单个输入通道的重要程度，非对角项表示通道相关性。GPTQ
按输入列量化：当前列舍入后，利用逆 Hessian 调整尚未量化的列，让后续列
补偿当前误差。

令

$$
H^{-1}=U^TU
$$

且 $U$ 为上三角矩阵。第 $i$ 列由 $w_i$ 量化为 $q_i$ 后，归一化误差为

$$
e_i=\frac{w_i-q_i}{U_{ii}},
$$

剩余权重更新为

$$
W_{:,i:}\leftarrow W_{:,i:}-e_iU_{i,i:}.
$$

该更新不会改变已经量化的列，只把误差沿校准数据所揭示的相关方向传播。
预测的二次损失可累计为

$$
L_{pred}=\frac12\sum_i
\frac{\lVert w_i-q_i\rVert_2^2}{U_{ii}^2}.
$$

### 数值稳定性

若某输入通道在校准集中恒为零，则对应 Hessian 对角项为零，称为 dead
column。该列不影响校准输出，可以把对应权重清零并把 Hessian 对角项设为
1。随后加入阻尼

$$
H\leftarrow H+\lambda I,\qquad
\lambda=\alpha\operatorname{mean}(\operatorname{diag}H).
$$

阻尼保证 Cholesky 分解稳定；过大会削弱通道相关信息。实现中使用 FP32
构造 Hessian、求 Cholesky 因子并传播误差。

### Group 与 Block

`group_size` 是量化格式：多少输入列共享 scale/zero，影响精度与 checkpoint
大小。`block_size` 是 GPTQ 的计算分块：先在 block 内逐列传播，再用一次矩阵
乘把累计误差更新到右侧矩阵，影响量化速度和临时内存。两者概念独立。

逐层量化在完成一层后，用反量化权重重新计算该层输出，再交给下一层校准。
因此后续层能够观察前面层已经产生的量化误差。

## 7. 质量指标

Teacher-forcing NLL 为

$$
\operatorname{NLL}=-\frac1{L-1}\sum_{t=1}^{L-1}
\log p(x_{t+1}\mid x_{\le t}).
$$

实验使用所有有效 token 加权的全局平均值，并比较

$$
\Delta\operatorname{NLL}=
\operatorname{NLL}_{INT4}-\operatorname{NLL}_{BF16}.
$$

任务一要求 $\Delta\operatorname{NLL}<0.16$；低于 0.1 可得到该项满分。

## 8. Baseline 推理路径

本实现采用实验框架提供的正确性 baseline：packed INT4 权重常驻设备；每次
Linear forward 临时反量化为输入 dtype，再调用 `F.linear`；注意力采用 eager
实现；KV Cache 按最大 batch 和最大序列长度静态连续分配；请求按静态 batch
执行。它不追求高性能，但保持模型语义完整，可作为后续融合 kernel、paged
KV cache 和 continuous batching 的正确性与性能参照。

## 9. 本轮优化与修正记录

本轮保留 GPTQ checkpoint 的语义，同时把运行时从“每次 Linear 反量化”改
为按负载分派：

- **Decode / 小 `M`**：走 `int4_triton` 融合 kernel，直接读取 packed
  INT4 权重和 group scale，避免物化完整 BF16 权重。
- **Prefill / 大 `M`**：反量化一次权重后交给 cuBLAS GEMM，让反量化成本
  被更多 token 摊薄。
- **分块注意力**：按 query 维 512 分块物化分数，降低长 prefill 的峰值
  显存；该实现仍会计算窗口外分数，是显存优化而非 Flash Attention。
- **滑窗 ring KV cache**：滑窗层只分配
  $\min(S,W)$ 个 KV slot，全局层仍分配完整 $S$；decode 时按逻辑位置
  gather 最近窗口。
- **静态 batch 重排**：无限并发评测下按生成长度、prompt 长度排序组批，
  减少同一 batch 中短请求等待长请求的无效 decode 和 padding。

### INT4 嵌入修正

为共享 embedding 与 LM head 的存储，实验中加入了在线量化后的
`QuantizedTiedEmbedding`。该实现曾将 packed `uint8` 拆成：

```text
codes[..., 0::2] = packed & 0x0F   # 错误：uint8 没有 uint4
codes[..., 1::2] = packed >> 4
```

在 PyTorch 中这不是宽度感知赋值：偶数列会先接收低位结果，随后右移值也被
写入同一片内存。修正为：

```python
low = packed & 0x0F
high = packed >> 4
codes = torch.stack((low, high), dim=-1).flatten(-2)
```

再与每 group scale 相乘。该修正解释并消除此前输出中大量 `token_id=0` 的
现象；修正后需要重新运行公开质量与性能评测来刷新提交结果。

随后发现模型入口的 embedding 缩放链式赋值在赋值前引用了
`hidden_states`，导致 `UnboundLocalError`；已拆成两步赋值修复。

### 实验现象

公开质量评测中，GPTQ 的全局
$\Delta\mathrm{NLL}=0.0861$，小于 0.1 的满分线，且各长度桶均通过 0.16
约束。公开性能测试在 A100 MIG 10GB 环境下，batch size 1 的参考路径约
248.1s；启用 INT4 kernel、分块注意力和滑窗 KV 后的完整路径约 74.5s。
中间实验显示：仅 batching 约 86.1s，增加 kernel 后在 74.5s–104.5s 区间
波动，说明剩余瓶颈仍在注意力实现、同步和调优空间中。

### 已知限制

- `QuantizedTiedEmbedding` 采用在线 RTN 量化，尚未纳入 GPTQ 校准流程；
  精度需以当前公开质量结果为准。
- Triton kernel 的 tile 参数仍为固定启发式，未针对不同 `N,K,M` 自动调优。
- eager attention 的 mask/softmax 仍是主要显存和带宽来源，长序列风险高。
- 当前结果是单次运行，重复运行可能受 MIG 邻居负载影响。
