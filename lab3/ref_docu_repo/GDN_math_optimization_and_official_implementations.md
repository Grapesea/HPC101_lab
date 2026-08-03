# GDN Prefill 的数学优化：从显式 W/U 到融合残差与并行 chunk

本文面向本项目的 GDN prefill forward，重点回答两个问题：

1. 哪些优化来自严格的代数等价变换，而不只是 CUDA 调度技巧？
2. FLA、FlashQLA、FlashInfer 的实现分别采用了什么数学表达，哪些思路适合迁移到本项目？

项目当前固定参数为 chunk size `C=64`、`d_k=d_v=128`，Q/K/V 与 `A` 为 BF16，gate、beta 和跨 chunk state 为 FP32。课程计时范围只包含 [`student/tilelang_fwd.py`](../student/tilelang_fwd.py) 的 W/U、state 和 output；[`preprocessing/tilelang_cumsum.py`](../preprocessing/tilelang_cumsum.py) 与 [`preprocessing/tilelang_kkt_solve.py`](../preprocessing/tilelang_kkt_solve.py) 在核心计时之外。

## 1. 结论先行

对当前 student core，最有价值的等价变换是：

\[
\boxed{
V_\Delta
=A_g B\left(V-DKS_{\mathrm{in}}\right)
}
\]

其中 `A_g` 是项目已经传入的 gated KKT inverse，`B=diag(beta)`，`D=diag(exp(g_cumsum))`。它等价于当前代码的：

\[
W=A_gBDK,\qquad U=A_gBV,\qquad V_\Delta=U-WS_{\mathrm{in}}.
\]

因此无需分别生成 W 和 U。可以先算预测残差 `R=V-DKS_in`，再算 `V_delta=A_g(BR)`。这个重排：

- 严格减少一次 `A @ K` 型 chunk GEMM；
- 可以取消 W、U 两个全局 BF16 中间张量；
- 与 FlashQLA 和 FlashInfer 的融合 forward 主路径一致；
- 不要求修改目前已经 gated 的 `A`，最适合作为第一步实现。

建议的实施顺序是：

1. **P0：残差重排并合并 W/U 与 state kernel。** 暂时仍把 `V_delta`、chunk-entry state 写给 output kernel，以较低风险验证公式和数值误差。
2. **P1：把 state、V_delta、output 合并成逐 head 的 persistent kernel。** state 跨 chunk 常驻 FP32 registers，消除 `V_delta` 和每个 chunk 的 entry-state 全局往返。
3. **P2：令一个 CTA 覆盖完整 `d_v=128`，复用同一份 QK score。** 当前两个 64-wide V tile 会重复计算 QK。
4. **P3：针对 GVA 提取共享的 raw `QK^T`。** `parallel_gva` 中 4 个 V head 共享一个 Q/K head；连同两个 V tile，raw score 当前最多被重复 8 次。
5. **P4：预计算/复用 gate 系数，研究因果 block sparsity。** 这类优化收益较小，但实现独立。
6. **P5：若优化完整 forward，再采用 gate-free KKT、直接三角求解或 context parallel。** 这些改变主要作用于预处理或低并行度长序列，不应先于 P0/P1。

此前对 `parallel_gva` 的 profiling 归类显示，student core 约为 2.094 ms，其中 chunk state、chunk output、W/U 分别约占 56.38%、31.31%、12.31%。所以不能只微调 W/U kernel；真正的大方向是用上述等价变换把三个阶段合成同一条数据流。

## 2. 从逐 token recurrence 到 chunk 公式

### 2.1 记号

对一个 chunk，记：

- \(Q,K\in\mathbb R^{C\times d_k}\)，\(V\in\mathbb R^{C\times d_v}\)；
- \(\gamma_i=\sum_{t\le i}g_t\) 为 chunk-local log-gate 前缀和；
- \(D=\operatorname{diag}(e^{\gamma_1},\ldots,e^{\gamma_C})\)；
- \(B=\operatorname{diag}(\beta_1,\ldots,\beta_C)\)；
- \(S_{\mathrm{in}}\in\mathbb R^{d_k\times d_v}\) 为进入 chunk 的 state；
- \(s=d_k^{-1/2}\) 为 output scale；
- \(\mathcal L(X)=\operatorname{tril}(X,-1)\) 只保留严格下三角，\(\overline{\mathcal L}(X)=\operatorname{tril}(X,0)\) 保留下三角及对角线。

项目的 gated KKT 矩阵及其逆为：

\[
M_g=I+\mathcal L\left(BDKK^TD^{-1}\right),
\qquad
A_g=M_g^{-1}.
\]

逐元素写出正是 [`references/torch_gdr.py`](../references/torch_gdr.py) 和项目预处理中的定义：

\[
(M_g)_{ij}=
\begin{cases}
1,&i=j,\\
\beta_i\langle k_i,k_j\rangle e^{\gamma_i-\gamma_j},&i>j,\\
0,&i<j.
\end{cases}
\]

### 2.2 逐 token state 是一个仿射映射

令线性 forget gate \(\alpha_t=e^{g_t}\)。按项目 reference 的 delta-rule 更新，可整理为：

\[
S_t
=\alpha_t\left(I-\beta_t k_t^Tk_t\right)S_{t-1}
+\beta_t k_t^Tv_t.
\]

定义：

\[
F_t=\alpha_t\left(I-\beta_t k_t^Tk_t\right),
\qquad
N_t=\beta_t k_t^Tv_t,
\]

则 \(S_t=F_tS_{t-1}+N_t\)。这个“线性变换 + 常量项”的结构，是后文 chunk composition 和 context parallel 的数学基础。

### 2.3 当前 chunk 表达

当前实现先构造：

\[
W=A_gBDK,\qquad U=A_gBV,
\]

然后：

\[
V_\Delta=U-WS_{\mathrm{in}}.
\]

output 和 chunk 末 state 分别为：

\[
O=s\left[
DQ S_{\mathrm{in}}
+\overline{\mathcal L}\left(DQK^TD^{-1}\right)V_\Delta
\right],
\]

\[
S_{\mathrm{out}}
=e^{\gamma_C}S_{\mathrm{in}}
+K^TEV_\Delta,
\qquad
E=\operatorname{diag}\left(e^{\gamma_C-\gamma_i}\right).
\]

这里 output 必须使用更新前的 \(S_{\mathrm{in}}\)，而 state update 使用同一 chunk 算出的 \(V_\Delta\)。融合 kernel 中必须保持这个依赖顺序。

## 3. 第一优先级：消去显式 W/U

### 3.1 严格等价推导

利用矩阵乘法的结合律和分配律：

\[
\begin{aligned}
V_\Delta
&=A_gBV-A_gBDKS_{\mathrm{in}}\\
&=A_gB\left(V-DKS_{\mathrm{in}}\right).
\end{aligned}
\]

所以可改写为：

\[
R=V-DKS_{\mathrm{in}},
\qquad
V_\Delta=A_g(BR).
\]

注意乘法次序不能随意改变。正确的是 `A_g @ (beta * residual)`；一般情况下不能把 beta 移到 `A_g` 左边，因为 \(A_gB\ne BA_g\)。

### 3.2 对应的 kernel 数据流

保守的两阶段版本可采用：

1. 每个 `(batch, value_head, value_tile)` CTA 把进入 chunk 的 state 保存在 FP32 fragment。
2. 计算 `prediction = K @ state`。
3. 在 FP32 中形成 `residual = V - exp(gamma) * prediction`，再逐行乘 beta。
4. 将 residual 按 BF16 Tensor Core operand 所需格式放入 shared memory，计算 `V_delta = A_g @ residual`。
5. 写出 `V_delta`，并用 `K^T @ (end_decay * V_delta)` 更新 FP32 state。
6. output kernel 暂时保持不变。

这样可以直接删除 `_tilelang_recompute_w_u` 的 W/U 输出及两个张量分配。第二阶段再把 output 搬进同一 persistent kernel，使 `V_delta` 也不落 HBM。

### 3.3 计算量变化

只统计主要 GEMM，并按一次乘加为 2 FLOPs。令当前 output 的 V 维 tile 数为 \(r_v=\lceil d_v/B_V\rceil\)。现有三个 kernel 的主要 FLOPs 为：

\[
F_{\mathrm{current}}
=(2+2r_v)C^2d_k+4C^2d_v+6Cd_kd_v.
\]

其中额外的 \(2r_vC^2d_k\) 来自每个 V tile 都重新计算完整 \(QK^T\)。残差重排、但仍保留原 output kernel 后：

\[
F_{\mathrm{residual}}
=2r_vC^2d_k+4C^2d_v+6Cd_kd_v.
\]

严格减少：

\[
\boxed{2C^2d_k}
\]

即一次 `A_g @ K` 型 GEMM。若进一步用完整 `d_v=128` 的 fused CTA，使 \(r_v=1\)：

\[
F_{\mathrm{fused}}
=2C^2d_k+4C^2d_v+6Cd_kd_v.
\]

对 `C=64, d_k=d_v=128, B_V=64`：

| 路径 | 每个 chunk/value-head 的主要 FLOPs | 相对当前 |
|---|---:|---:|
| 当前，\(r_v=2\) | 11,534,336 | 100% |
| 只做残差重排，output 不变 | 10,485,760 | 90.9% |
| 残差重排 + 完整 V tile 融合 | 9,437,184 | 81.8% |

这还没有计入 kernel launch、重复 gate 指数以及中间张量的 HBM 流量，因此实际收益不能只由 FLOPs 推断。

### 3.4 中间张量流量

按每个 chunk/value-head 统计 BF16 张量一次写出再读回的逻辑流量：

| 中间量 | 形状 | 往返流量 |
|---|---:|---:|
| W | \(C\times d_k\) | \(4Cd_k=32\) KiB |
| U | \(C\times d_v\) | \(4Cd_v=32\) KiB |
| `V_delta` | \(C\times d_v\) | 32 KiB |
| chunk-entry state | \(d_k\times d_v\) | 64 KiB |
| 合计 |  | 160 KiB |

第一阶段消去 W/U 后，逻辑流量从 160 KiB 降为 96 KiB；完全融合后，上述四类内部量都不再往返 HBM，只保留输入、output 和必要的 final state。

对 `parallel_gva`（`B=1,T=2048,Hv=16`），当前 W、U、`V_delta` 各占 8 MiB，chunk-entry state 占 16 MiB，共 40 MiB 中间存储；其逻辑写回加重读约为 80 MiB/次 forward。残差两阶段版本降到 24 MiB，完全融合则可取消这 40 MiB 工作集。

## 4. 第二优先级：把整个 chunk recurrence 融成 persistent kernel

残差重排给出了适合融合的顺序：

\[
\begin{array}{rcl}
O_{\mathrm{state}}&=&sDQ S_{\mathrm{in}},\\
R&=&V-DKS_{\mathrm{in}},\\
V_\Delta&=&A_gBR,\\
O_{\mathrm{local}}&=&s\overline{\mathcal L}(DQK^TD^{-1})V_\Delta,\\
S_{\mathrm{out}}&=&e^{\gamma_C}S_{\mathrm{in}}+K^TEV_\Delta.
\end{array}
\]

一个 CTA 可以按时间顺序遍历该 `(batch, value_head)` 的所有 chunk，并让 state 始终驻留 FP32 registers。每个 chunk 内，Q/K/V/A 通过多 stage shared-memory pipeline 载入；不同 warp group 分别负责 state、残差/三角变换、QK/output 和 load/store。

这不是新的数学近似，但只有在上一节的表达下才容易做到：显式 W/U 会把跨 chunk recurrence 与 chunk-local 工作人为切成多个全局张量。

风险在于资源占用。当前 state kernel 已经是高 register/high shared-memory kernel，简单地把更多代码塞进 128 threads 很可能产生 register spill。FlashQLA 与 FlashInfer 的 Hopper 路径都使用 512 threads 和 warp specialization，把 load/store、state MMA、auxiliary QK/KKT 工作分给不同 warp group；迁移时应复用这种角色划分，而不是只做机械 kernel fusion。

## 5. Gate-free KKT：FlashQLA 的对角相似变换

### 5.1 推导

定义不含 gate ratio 的矩阵：

\[
M_0=I+\mathcal L\left(BKK^T\right),
\qquad
A_0=M_0^{-1}.
\]

因为 B、D 都是对角矩阵且彼此可交换，并且对角相似变换不改变上下三角位置：

\[
\begin{aligned}
DM_0D^{-1}
&=I+\mathcal L\left(DBKK^TD^{-1}\right)\\
&=I+\mathcal L\left(BDKK^TD^{-1}\right)\\
&=M_g.
\end{aligned}
\]

因此：

\[
\boxed{A_g=DA_0D^{-1}}.
\]

逐元素为：

\[
(A_g)_{ij}=e^{\gamma_i-\gamma_j}(A_0)_{ij}.
\]

这解释了 FlashQLA 为什么在 KKT solve 中只计算 `beta * K K^T`，而在 fused forward 中才把 `exp(gamma_i-gamma_j)` 和列方向 beta 乘到 inverse 上。它不是近似，而是精确的 diagonal similarity transform。

### 5.2 与残差表达合并

令 \(T=A_gB\)，则：

\[
T=DA_0D^{-1}B.
\]

FlashQLA 在 kernel 中构造的矩阵逐元素正是：

\[
T_{ij}=e^{\gamma_i-\gamma_j}(A_0)_{ij}\beta_j,
\]

随后计算：

\[
V_\Delta=T\left(V-DKS_{\mathrm{in}}\right).
\]

对本课程核心计时，`A_g` 已经在外部算好，所以不应为了这一变换先改接口；直接使用第 3 节的 `A_g @ (beta * residual)` 即可。若目标变成完整 forward，则 gate-free KKT 有三个优点：

- KKT 构造时不再计算 \(C^2\) 个 pairwise gate exponent；
- gate-free inverse 与 gate 处理解耦，便于融合和复用；
- 避免在 KKT 输入中直接形成跨度较大的 `exp(gamma_i-gamma_j)`，通常更利于数值稳定。

### 5.3 只做 forward 时可考虑直接三角求解

由于：

\[
M_gV_\Delta=B\left(V-DKS_{\mathrm{in}}\right),
\]

若不需要显式保存 inverse，可直接对多个 RHS 做 unit-lower-triangular solve，而不是先形成 \(A_g=M_g^{-1}\) 再 GEMM。理论上，三角 solve 只访问下三角，算术量约为 dense `A_g @ R` 的一半，并取消 A 的全局存储。

但这不适用于当前 student 接口，因为 A 已经作为输入传入且预处理不计时；同时逐行 forward substitution 的 GPU 并行性弱于 dense Tensor Core GEMM。更现实的完整-forward方案是 16×16 block solve：对角块做 substitution，非对角更新使用 GEMM。FLA 的 fused KKT/inverse 也采用了类似的 16→32→64 分块合并思想。

## 6. QK/KK 公共子表达式与 GVA

### 6.1 当前重复比例

`parallel_gva` 的 `Hq=4,Hv=16`，所以每个 Q/K head 被 4 个 value head 共享。与此同时，当前 output kernel 把 `d_v=128` 分成两个 64-wide tile；每个 tile 都重新计算完整的 raw：

\[
P_{\mathrm{raw}}=QK^T.
\]

因此，同一份 raw score 在一个 chunk 中最多计算：

\[
\frac{H_v}{H_q}\times\frac{d_v}{B_V}=4\times2=8
\]

次。要注意，应用 gate 后的 score 不一定能在 4 个 value head 间共享，因为每个 value head 有自己的 \(\gamma\)；能共享的是 gate 之前的 raw `QK^T`。

### 6.2 三种精确方案

1. **一个 CTA 覆盖完整 V 维。** 把 `block_DV` 从 64 提到 128，同一 CTA 保留 score 并服务两个 V half；raw QK 重复从 8 次降到 4 次。FlashQLA 会按 CTA 数自适应选择 `block_DV=128/64/32`，本项目 `parallel_gva` 在 14-SM MIG 上有 16 个 value heads，优先尝试 128 是合理的。
2. **按 `(chunk,value_head)` 预计算 gated score。** score 写一次、由两个 V tile 读取，改动较小，但不能跨 4 个 GVA heads 共享。
3. **按 `(chunk,qk_head)` 预计算 raw score。** 每个 Q/K head 只算一次，value-head kernel 读取后应用各自 gate。它把理论重复从 8 次降到 1 次，但会引入一个 \(C\times C\) score workspace 及读写流量。

应以 benchmark 决定方案 1 还是 3。对 `C=64`，一份 BF16 score tile 仅 8 KiB；但写 BF16 raw score会改变“FP32 score → gate → BF16”的舍入顺序。若需要最接近当前结果，应保存 FP32 raw score，或者保存每个 value head 已 gated 的 BF16 score。

### 6.3 KKT 的 GVA 复用

同样地，raw Gram：

\[
G_K=KK^T
\]

只依赖 Q/K head。不同 value head 只是在其上应用各自 beta、gate 和三角 inverse。完整 forward 可以先按 Q/K head 计算 Gram，再给同组的 4 个 value heads 使用，从而省去 75% 的 raw KK GEMM。

这一优化作用于预处理，不影响 student-core 成绩；而且需要在“保存 Gram 到 HBM”与“同一 CTA 内串行处理同组 value heads”之间权衡。

## 7. Gate 系数复用与因果块稀疏

### 7.1 不要在 V 维循环内重复 exponent

当前 state update 所需：

\[
r_i=e^{\gamma_C-\gamma_i}
\]

只依赖 token row，却在 `(row,dim_v)` 工作中重复计算。应先生成 C 个 FP32 `r_i`，再广播到整个 V row。类似地：

- `exp(gamma_i)` 每个 token 只需计算一次；
- `exp(gamma_i-gamma_j)` 每个因果 `(i,j)` 只需计算一次，不应在不同 V tile 重算；
- raw QK 和 gate matrix 应先逐元素结合，再供所有 V columns 使用。

不建议简单保存 `exp(gamma_i)` 与 `exp(-gamma_j)` 后相乘。虽然代数上：

\[
e^{\gamma_i-\gamma_j}=e^{\gamma_i}e^{-\gamma_j},
\]

但单独的 `exp(-gamma_j)` 可能溢出，而在 GDN 的因果区域 `i>=j`、log-gate 通常非正时，直接计算 `exp(gamma_i-gamma_j)` 始终不大于 1，更稳定。

### 7.2 16×16 因果 block sparsity

当前先计算完整 64×64 QK，再把上三角清零。若以 16×16 分块，一个 4×4 block grid 只需计算下三角的：

\[
\frac{4(4+1)}2=10
\]

个 block，而不是 16 个，可跳过 37.5% 的 block-level QK 与对应 score-value 累加。FLA 的 fused KKT kernel 正是只构造 4 个对角块和 6 个严格下三角块。

但小 tile/多次 GEMM 可能降低 Tensor Core 利用率，所以这属于需要实测的二级优化。建议在 P0/P1 完成后，再比较 dense 64×64 与 10 个 16×16 block 的端到端时间。

## 8. Chunk 是可组合的仿射映射

### 8.1 精确 chunk transfer

令 \(T=A_gB\)。把第 2 节的 state 公式代入：

\[
\begin{aligned}
S_{\mathrm{out}}
&=e^{\gamma_C}S_{\mathrm{in}}
+K^TET\left(V-DKS_{\mathrm{in}}\right)\\
&=\underbrace{\left(e^{\gamma_C}I-K^TETDK\right)}_{M_c}S_{\mathrm{in}}
+\underbrace{K^TETV}_{N_c}.
\end{aligned}
\]

所以每个 chunk 都定义一个仿射映射：

\[
\boxed{S_{c+1}=M_cS_c+N_c}.
\]

两个相邻 chunk 的组合满足：

\[
(M_2,N_2)\circ(M_1,N_1)
=\left(M_2M_1,\;M_2N_1+N_2\right),
\]

而且该 composition 具有结合律，因此原则上可以对 chunk 做 parallel prefix scan，把串行深度从 \(O(n_{chunk})\) 降到 \(O(\log n_{chunk})\)。

### 8.2 为什么不能对所有 case 无脑做 scan

这里 \(M_c\) 是 128×128，显式构造和组合需要：

- 每个 chunk 保存一个 FP32 \(M_c\) 和 \(N_c\)，约 128 KiB/value-head；
- composition 包含 128×128×128 的矩阵乘法；
- prefix 还需要额外 workspace 和多轮同步。

原本逐 chunk recurrence 的代价已经是 `C×128×128` 量级；在 `C=64` 时，一次 128³ composition 并不便宜。因此 context parallel 只应在 batch×heads 太小、序列很长、GPU 因串行 chunk 链而空闲时启用。

### 8.3 FlashInfer：精确 transfer + fixup

FlashInfer 的 SM90 context-parallel 路径采用精确的仿射结构：

1. 每 64 tokens 预计算三角变换 T；
2. 把长序列切成较大的 CP chunks，每个 CP chunk 独立计算 local transfer \(M_c\) 与 local state \(N_c\)；
3. fixup kernel 按 \(S_{c+1}=M_cS_c+N_c\) 修正各 CP chunk 的真实边界 state；
4. 各 CP chunk 使用修正后的入口 state 并行执行 fused prefill。

这在实数数学中是精确算法，只有浮点结合次序不同。FlashInfer 的常规 CP 选择以 512 tokens 为 chunk-length 粒度，短工作负载的平衡分支则可按 64 tokens 对齐；它用 SM 数、head 数和序列长度共同选择 CP chunk length。公开接口只在非 CP 的 `batch×state_heads` 并行度低于设备阈值时自动启用。

### 8.4 FlashQLA：gate 驱动的 warmup，必要时精确 fallback

FlashQLA 也会切分长序列，但利用 forget gate 判断旧 state 的影响是否已经足够小。其默认阈值是累计 log-gate `< -10`：从某个 CP segment 的末尾向前取若干 warmup chunks，从零 state 重算局部 state；若在整个 segment 内仍达不到阈值，则额外计算 transfer matrix 并执行精确 correction。

为什么 `-10` 有数学依据？课程输入中 K 被 L2-normalize，且 \(0\le\beta_t\le1\)。于是：

\[
\left\|I-\beta_tk_t^Tk_t\right\|_2\le1.
\]

对一段 token 的入口 state 扰动 \(\Delta S\)：

\[
\|\Delta S_{\mathrm{out}}\|_2
\le e^{\sum_t g_t}\|\Delta S_{\mathrm{in}}\|_2.
\]

当 \(\sum_tg_t<-10\) 时，残留系数小于：

\[
e^{-10}\approx4.54\times10^{-5}.
\]

所以 FlashQLA 在达到阈值时省略旧 segment 的 transfer contribution 是**有误差界的近似**；达不到阈值时，`fallback_mask` 触发 matrix correction，恢复精确依赖。它与 FlashInfer 的始终精确 fixup 不同，不能笼统地都称为 exact CP。

该界依赖 \(\|k_t\|_2=1\) 和 \(0\le\beta_t\le1\)。课程数据满足这两个条件；对未做 Q/K normalization 或允许其他 beta 范围的通用接口，不能直接沿用此截断保证。

对当前 `parallel_gva`，`B×Hv=16`，而设备只有 14 SM，head 级并行本身已接近一整 wave；context parallel 通常不是首要矛盾。它更适合 `chain_equal` 一类 head 数少且序列长的 case。

## 9. 三种官方实现的数学路径对比

本地 adapter 分别位于：

- [`references/official/fla.py`](../references/official/fla.py)
- [`references/official/flash_qla.py`](../references/official/flash_qla.py)
- [`references/official/flashinfer.py`](../references/official/flashinfer.py)

| 实现 | KKT / triangular transform | `V_delta` 路径 | state/output | Context parallel | 对本项目的启示 |
|---|---|---|---|---|---|
| 当前 student | 外部构造 gated \(A_g\) | 显式 `W=A_gBDK`、`U=A_gBV`，再 `U-WS` | state、output 分离；保存 `V_delta` 和每 chunk entry state | 无 | 公式正确，但中间量和重复 GEMM 最多 |
| FLA | C=64 时融合 gated KKT 构造与 16→32→64 inverse；A 写回 | 经典 WY：仍显式生成 W/U | state recurrence 与 output kernel 分离 | API 可接 CP context；本地 adapter 未传 | 最接近 student，可参考 fused KKT 和 lower-block 构造，但 W/U 不是最终目标 |
| FlashQLA | gate-free \(A_0=(I+\mathcal L(BKK^T))^{-1}\) | `V-exp(gamma)KS`，再乘 \(DA_0D^{-1}B\) | 512-thread warp-specialized persistent kernel；自适应 `block_DV=128/64/32` | gate-threshold warmup；衰减足够时近似截断，否则 transfer fallback | 残差重排、对角相似变换、完整融合最值得迁移 |
| FlashInfer SM90 | fused kernel 内计算 gated KKT inverse，并把 beta 折入 T | `V-exp(gamma)KS`，再乘 T | 512-thread CuTe DSL kernel；state 以 `[V,K]` 保存 | local affine transfer + exact fixup + parallel prefill | 证明无需改成 gate-free A 也能直接采用残差路径；CP 数学结构最清晰 |

### 9.1 FLA

当前上游 FLA 的 `chunk_gated_delta_rule_fwd_intra` 在 chunk size 64 时，把 `K K^T`、gate/beta 缩放和 lower-triangular inverse 合到一个 Triton kernel，避免未求逆 KKT 矩阵的 HBM round trip；随后 `recompute_w_u_fwd` 仍显式生成 W、U。它代表“保留传统 WY 分解、先优化预处理”的路线。

这一设计对训练/反向传播很合理：A 会为 backward 保存，而 W/U 表达在 backward 中可以按需重算并提供结构化梯度路径。但本课程只计 forward core，且 W/U 正是可消去的中间量，所以不应照搬最后一步。

### 9.2 FlashQLA

FlashQLA 的 Hopper forward 直接体现第 3、5 节公式：

- KKT kernel 求 gate-free \(A_0\)；
- 一个 warp group 计算 `K @ state` 与 `V-exp(gamma)*(K@state)`；
- 另一个 warp group 把 gate ratio 和 beta 列缩放折入 \(A_0\)；
- 通过一次 `A_transform @ residual` 得到 `V_delta`；
- 同一 CTA 完成旧 state output、chunk-local output 和 state update。

其 512-thread 角色划分和 adaptive `block_DV` 是调度优化；残差重排与 diagonal similarity 才是使这种调度可行的数学基础。

### 9.3 FlashInfer

本地 adapter 先把课程的 raw log-gate 转成线性 \(\alpha=e^g\)，把 batch/sequence 打包为 flat tokens，并把 state 从 `[K,V]` 转置为 `[V,K]`。转置只是布局选择，不改变 recurrence；K-last 对按 value row 读取/更新 state 更友好。

FlashInfer SM90 kernel 每个 64-token block 同时生成 QK 与 KK，应用 gate/beta、做 collective inverse，然后执行：

\[
K S,\quad V-DKS,\quad T(V-DKS),\quad O,\quad S_{\mathrm{next}}.
\]

它说明本项目无需先改 KKT preprocessing：即使保留 gated \(A_g\)，`A_g @ (beta * residual)` 也完全适合 fused forward。

## 10. 数值语义与正确性风险

上述代数变换在实数中严格等价，但 BF16/FP32 下会改变舍入点。建议明确区分：

| 类别 | 示例 | 数学性质 | 主要风险 |
|---|---|---|---|
| 严格代数变换 | `U-WS` → `A_g B(V-DKS)`；diagonal similarity；affine composition | 实数中完全等价 | 浮点结合次序改变 |
| 数值重排 | `exp(a-b)` → `exp(a)/exp(b)`；raw score 落 BF16 workspace | 代数等价 | overflow/underflow 与 BF16 舍入不同 |
| 调度变化 | warp specialization、double buffering、TMA、adaptive block V | 不改变公式 | register spill、occupancy、同步错误 |
| 近似 | FlashQLA gate-threshold warmup 截断 | 有显式误差界 | 可能超过课程 `rtol=atol=5e-3`，需全 case 验证 |

推荐的数值策略：

- state accumulator 始终保持 FP32；
- `K @ state`、`Q @ state` 和 state update 使用 FP32 accumulator；
- gate exponent 与 residual subtraction 在 FP32 中完成；
- 在进入下一次 Tensor Core GEMM 前再按设计转 BF16，参考 FlashQLA 的“FP32 residual → BF16 shared operand”路径；
- 直接计算因果区域内的 `exp(gamma_i-gamma_j)`，不要拆成可能溢出的 reciprocal；
- 每次改变舍入边界后，至少运行所有 benchmark cases，而不只验证 `parallel_gva`。

建议验证命令：

```bash
python run.py --case parallel_gva --warmup 10 --repetitions 100
python run.py --warmup 10 --repetitions 100
```

优化过程应分别记录 output 最大误差、final-state 最大误差和 median latency。项目容差定义在 [`evaluation/support.py`](../evaluation/support.py)，当前为 `rtol=5e-3, atol=5e-3`。

## 11. 可落地的分阶段改造方案

### 阶段 A：验证残差公式

- 新写一个 fused residual-state kernel，输入 `K,V,g,beta,A,initial_state`。
- 在 kernel 内计算 `V_delta=A @ (beta * (V-exp(g)*K@S))`。
- 保留现有 output kernel和 `hidden`/`V_delta` 接口。
- 删除 W/U 分配和 `_tilelang_recompute_w_u` 调用。
- 目标：所有 cases 正确；`parallel_gva` 至少消除 W/U 的 16 MiB allocation 和约 32 MiB 逻辑往返。

### 阶段 B：完整 V tile 与 gate hoist

- 在资源允许时使用 `block_DV=128`，只算一次 QK。
- 将 `exp(gamma_i)`、`exp(gamma_last-gamma_i)` 预计算成 row-wise FP32 shared arrays。
- 若 128-wide 发生 spill，则仿照 FlashQLA 按可用 CTA 数选择 128/64/32，而不是固定一个配置。

### 阶段 C：persistent fused forward

- 一个 CTA 遍历一个 `(batch,value_head)` 的所有 chunks。
- state 常驻 FP32 registers；同一 CTA 按“旧 state output → residual → V_delta → local output → state update”的顺序执行。
- 取消 `hidden` 与 `V_delta` 全局中间量，并合并 output launch。
- 用 512 threads/warp specialization 控制 register ownership；用 Nsight Compute 检查 spill、active warps 和 Tensor Core utilization。

### 阶段 D：GVA score workspace 或 grouped CTA

- 先测量完整 V tile 后 QK 仍占多少时间。
- 若明显，再按 `(chunk,qk_head)` 生成 raw QK workspace，并让 4 个 value heads应用各自 gate。
- 同时可在完整 forward preprocessing 中复用 raw KK Gram。
- 对比 BF16 与 FP32 score workspace 的性能和误差。

### 阶段 E：低并行度长序列

- 仅在 `batch×heads` 不能覆盖 SM 且 chunks 很多时构造 chunk affine transfer。
- 若要求严格正确，优先 FlashInfer 式 local transfer + exact fixup。
- 若允许受控近似，利用课程中 normalized K 和 beta 范围，按累计 gate 给出截断误差界，再尝试 FlashQLA 式 warmup。

## 12. 资料与版本

本文于 2026-08-02 对照以下上游快照。固定 commit 是必要的，因为三套实现都在快速演进：

- FlashQLA，commit [`821fd9d37ede18fdc2a4e707fefe3770bfc32e58`](https://github.com/QwenLM/FlashQLA/tree/821fd9d37ede18fdc2a4e707fefe3770bfc32e58)
  - [Hopper fused forward](https://github.com/QwenLM/FlashQLA/blob/821fd9d37ede18fdc2a4e707fefe3770bfc32e58/flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py)
  - [gate-free KKT solve](https://github.com/QwenLM/FlashQLA/blob/821fd9d37ede18fdc2a4e707fefe3770bfc32e58/flash_qla/ops/gated_delta_rule/chunk/hopper/kkt_solve.py)
  - [intra-card CP context](https://github.com/QwenLM/FlashQLA/blob/821fd9d37ede18fdc2a4e707fefe3770bfc32e58/flash_qla/ops/gated_delta_rule/chunk/cp_context.py)
- FLA，commit [`b9afc14acc1545ba4149f16895e945962ddaf9a9`](https://github.com/fla-org/flash-linear-attention/tree/b9afc14acc1545ba4149f16895e945962ddaf9a9)
  - [gated-delta chunk forward](https://github.com/fla-org/flash-linear-attention/blob/b9afc14acc1545ba4149f16895e945962ddaf9a9/fla/ops/gated_delta_rule/chunk_fwd.py)
  - [W/U representation](https://github.com/fla-org/flash-linear-attention/blob/b9afc14acc1545ba4149f16895e945962ddaf9a9/fla/ops/gated_delta_rule/wy_fast.py)
- FlashInfer，commit [`d020372b068f335e2fe427372e134977a2235c49`](https://github.com/flashinfer-ai/flashinfer/tree/d020372b068f335e2fe427372e134977a2235c49)
  - [public GDN prefill API](https://github.com/flashinfer-ai/flashinfer/blob/d020372b068f335e2fe427372e134977a2235c49/flashinfer/gdn_prefill.py)
  - [SM90 fully fused delta-rule kernel](https://github.com/flashinfer-ai/flashinfer/blob/d020372b068f335e2fe427372e134977a2235c49/flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm90.py)
  - [SM90 context-parallel kernel](https://github.com/flashinfer-ai/flashinfer/blob/d020372b068f335e2fe427372e134977a2235c49/flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm90.py)

项目内最直接的公式基准是 [`references/torch_gdr.py`](../references/torch_gdr.py)。实现任何重排前，应先用它确认矩阵方向、beta 是按 row 还是 column 缩放，以及 state 使用 `[K,V]` 还是 `[V,K]` 布局。
