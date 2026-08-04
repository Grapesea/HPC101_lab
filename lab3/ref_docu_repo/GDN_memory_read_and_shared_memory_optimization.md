# GDN Prefill 内存读取与 Shared Memory 优化

## 1. 优化目标

本文针对 [`student/tilelang_fwd.py`](../student/tilelang_fwd.py) 的 persistent fused forward，重点降低以下开销：

- 同一 `(chunk,value_head)` 因两个 64-wide value tile 而产生的重复 Q/K/A/gate global load；
- Q、K 在 $QS_{[c]}$、$KS_{[c]}$、$QK^T$ 和 state update 之间的重复读取；
- `g_cumsum` 在求 $\gamma_r$ 和 gate ratio 时的重复 global load；
- residual、$\hat W$、decayed $\hat W$ 和 output 各自占用 shared buffer 所造成的 shared-memory 膨胀；
- `hidden`、`v_new` 等中间张量的 HBM 往返。

这里的主要目标是减少 global-memory read request 和逻辑读取字节数。实际 DRAM 读取量还会受到 L2 cache 命中影响，不能直接用逻辑字节数代替硬件计数器。

## 2. 优化前的内存访问方式

完成 kernel fusion、但仍使用 `BLOCK_V=64` 时，一个 value head 被拆成两个 CTA：

```text
CTA 0：处理 value columns [0, 64)
CTA 1：处理 value columns [64, 128)
```

虽然两个 CTA 处理的 V/state columns 不同，但下面这些量与 value column 无关：

$$
Q,\qquad K,\qquad A,\qquad B,\qquad \Gamma,\qquad QK^T.
$$

因此两个 CTA 会分别：

1. 从 global memory 加载完整 $Q[64,128]$；
2. 加载完整 $K[64,128]$；
3. 加载完整 $A[64,64]$；
4. 加载同一组 $g^{\text{cumsum}}$ 和 $\beta$；
5. 分别计算一次完整的 $QK^T[64,64]$。

L2 可能使命中后的第二次读取不访问 DRAM，但 load 指令、L2 transaction、global-to-shared copy 和 CTA 内同步仍然存在。

## 3. 完整 value tile：把两个 CTA 合成一个

当前实现把：

```python
BLOCK_DIM = 64
threads = 128
```

改为：

```python
BLOCK_DIM = 128
threads = 256
```

grid 中：

```python
HEAD_DIM_V // BLOCK_DIM
```

从 2 变为 1。因此每个 `(batch,value_head)` 只有一个 persistent CTA，负责完整的：

$$
S_{[c]}\in\mathbb R^{128\times128},\qquad
\hat W\in\mathbb R^{64\times128},\qquad
O_{[c]}\in\mathbb R^{64\times128}.
$$

同一 CTA 在一个 chunk 内只加载一次 Q/K/A/gate，并依次复用：

```text
Q_shared ──┬─→ Q S_[c]
           └─→ Q K^T

K_shared ──┬─→ K S_[c]
           ├─→ Q K^T
           └─→ K^T (γ_l Γ^-1 W_hat)
```

这比依赖 L2 cache 更彻底：数据进入 shared memory 后，后续 GEMM 不再发出 global load。

## 4. 每个 chunk/value-head 的逻辑读取量

固定 $C=64,d_k=d_v=128$，Q/K/V/A 为 BF16，gate/beta 为 FP32。

### 4.1 fused-64

两个 CTA 合计：

| 输入 | 单份大小 | 读取次数 | 逻辑读取量 |
|---|---:|---:|---:|
| Q | 16 KiB | 2 | 32 KiB |
| K | 16 KiB | 2 | 32 KiB |
| A | 8 KiB | 2 | 16 KiB |
| V | 16 KiB/完整 head | 两个半 tile | 16 KiB |
| $\beta$ | 0.25 KiB | 2 | 0.50 KiB |
| $g^{\text{cumsum}}$ | 0.25 KiB | 2 | 0.50 KiB |
| 合计 |  |  | **约 97 KiB** |

如果编译器没有合并“保存 gate”和“计算 exponent”中的相同 global load，旧代码的 $g^{\text{cumsum}}$ 读取还可能额外增加约 0.5 KiB。

### 4.2 fused-128

一个 CTA：

| 输入 | 读取次数 | 逻辑读取量 |
|---|---:|---:|
| Q | 1 | 16 KiB |
| K | 1 | 16 KiB |
| A | 1 | 8 KiB |
| V | 1 个完整 value tile | 16 KiB |
| $\beta$ | 1 | 0.25 KiB |
| $g^{\text{cumsum}}$ | 1 | 0.25 KiB |
| 合计 |  | **约 56.5 KiB** |

因此不计 initial/final state 和 output，主要输入的逻辑 global read 从约 97 KiB 降至 56.5 KiB：

$$
\frac{97-56.5}{97}\approx41.8\%.
$$

其中 V 的总读取量没有变化，因为两个 fused-64 CTA 原本各读取一半；减少的是 Q/K/A/gate 的重复读取。

initial state 只在整个 chunk chain 开始时读取一次，final state 只在结束时写一次，不应重复计入每个 chunk。output 是必要写回，也不属于 read 优化目标。

## 5. `g_cumsum`：global load 一次，shared 中多次使用

旧写法在同一个 row 中直接引用两次 global tensor：

```python
gate_shared[row] = g[...]
gate_exp_shared[row] = exp2(g[...] * LOG2E)
```

虽然编译器可能做 common-subexpression elimination，但不能依赖它一定跨 TileLang lowering 保留同一个值。

当前写法先完成一次 global-to-shared load：

```python
gate_shared[row] = g[...]
beta_shared[row] = beta[...]
```

再从 shared memory 生成：

```python
gate_exp_shared[row] = exp2(gate_shared[row] * LOG2E)
```

后续三个消费者都读取 shared-memory gate：

1. residual 中的 $\Gamma KS_{[c]}$；
2. output 中的 $\Gamma\operatorname{Lower}(QK^T)\Gamma^{-1}$；
3. state update 中的 $\gamma_\ell\Gamma^{-1}\hat W$。

这样每个 token/head/chunk 的 `g_cumsum` 主加载只有一次。

## 6. Shared-memory 生命周期复用

完整 value tile 若为每个数学阶段单独分配 shared buffer，会快速超过 100 KiB，并降低 occupancy。当前实现按生命周期复用两块关键 buffer。

### 6.1 `residual_shared[64,128]`

同一块 16 KiB BF16 shared memory 依次保存：

```text
V
↓
B(V - Γ K S_[c])
↓
W_hat = A B(V - Γ K S_[c])
↓
γ_l Γ^-1 W_hat
↓
O_[c] 的 BF16 store staging
```

严格的生命周期为：

1. 加载 V；
2. FP32 中形成 residual，再写回该 buffer；
3. `A @ residual` 完成后，residual 已死亡，覆盖为 $\hat W$；
4. chunk-local output 完成后，未衰减的 $\hat W$ 已死亡，覆盖为 $\gamma_\ell\Gamma^{-1}\hat W$；
5. state update 完成后，$\hat W$ 完全死亡，覆盖为 output staging。

因此不再需要单独的 `value_shared` 或 `output_shared`。

### 6.2 `a_shared[64,64]`

同一块 8 KiB BF16 shared memory 依次保存：

```text
A
↓  A @ residual 完成
Γ Lower(QK^T) Γ^-1
```

$A$ 只用于生成 $\hat W$。该 GEMM 结束后，A 不再被当前 chunk 使用，因此可立即被 gated causal score 覆盖。

### 6.3 当前 shared-memory 预算

| Buffer | Shape/dtype | 大小 |
|---|---|---:|
| `state_shared` | `[128,128]` BF16 | 32 KiB |
| `q_shared` | `[64,128]` BF16 | 16 KiB |
| `k_shared` | `[64,128]` BF16 | 16 KiB |
| `residual_shared` | `[64,128]` BF16 | 16 KiB |
| `a_shared` | `[64,64]` BF16 | 8 KiB |
| gate/beta/decay arrays | 4×`[64]` FP32 | 1 KiB |
| 合计 |  | **约 89 KiB/CTA** |

若为 $\hat W$ 和 output 再各分配一个 `[64,128]` BF16 buffer，shared memory 会增加到约 121 KiB。生命周期复用节约约 32 KiB/CTA。

FP32 `state_fragment`、`value_fragment`、`output_fragment` 和 `scores_fragment` 位于 registers，不计入 shared-memory 大小，但必须通过 profiling 检查 register spill。

## 7. 为什么暂不做全量双缓冲

可以用 ping-pong buffer 预取下一 chunk 的 Q/K/V/A，从而重叠 global load 和当前 chunk 的计算。然而一套下一 chunk 输入约为：

$$
16\text{ KiB}(Q)+16\text{ KiB}(K)+16\text{ KiB}(V)+8\text{ KiB}(A)
=56\text{ KiB}.
$$

在当前约 89 KiB 基础上全量双缓冲会达到约 145 KiB/CTA。它可能：

- 只能在每个 SM 驻留一个 CTA；
- 降低 active warps；
- 给尾块和编译器临时 shared allocation 留下很小余量；
- 在 global load 已被计算掩盖时得不偿失。

因此更合理的后续方案是先用 NCU 确认 memory dependency stall，再选择性双缓冲 K/V/A，而不是默认复制全部输入。

## 8. FLOPs 与 memory read 的共同下降

fused-64 中，同一 value head 的两个 CTA 各计算一次 $QK^T$。fused-128 只计算一次，因此每 chunk/head 还减少：

$$
2C^2d_k
=2\times64^2\times128
=1,048,576\text{ FLOPs}.
$$

主要 GEMM FLOPs 从：

$$
10,485,760
$$

降为：

$$
9,437,184,
$$

下降 10%。所以本次延迟改善同时来自：

- 约 41.8% 的主要输入逻辑读取下降；
- QK GEMM 减少一次；
- CTA 数和同步/地址计算减半；
- shared-memory 中的数据跨多个数学阶段复用。

## 9. 实测结果

所有公开 case 均通过正确性检查。以下是 `warmup=1,repetitions=2` 的快速对比，仅用于观察优化方向：

| Case | fused-64 (ms) | fused-128 (ms) | 延迟下降 |
|---|---:|---:|---:|
| `short_tail_state` | 0.408 | 0.301 | 26.2% |
| `chain_equal` | 2.924 | 1.992 | 31.9% |
| `parallel_equal` | 1.519 | 1.043 | 31.3% |
| `parallel_gva` | 1.520 | 1.039 | 31.6% |
| `long_low_gva` | 11.486 | 8.049 | 29.9% |
| `batch_split_gva` | 9.119 | 6.199 | 32.0% |
| `wide_gva_state` | 15.266 | 10.615 | 30.5% |
| `deep_gva_state` | 18.182 | 12.560 | 30.9% |

`parallel_gva` 使用 `warmup=10,repetitions=100` 的正式测量：

| 版本 | Median |
|---|---:|
| fused-64 | 1.467 ms |
| fused-128 | 1.036 ms |
| 改善 | **29.4%** |

延迟改善小于逻辑 read 字节下降是合理的，因为 kernel 还包含 Tensor Core GEMM、指数、因果 mask、同步和 state recurrence。

## 10. 如何验证实际 memory read

逻辑流量只能说明程序必需发出的 load。要区分 L2 hit 和真实 DRAM traffic，应在目标 GPU 上使用 NCU，并关注：

- `dram__bytes_read.sum`：实际 DRAM read bytes；
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`：DRAM 带宽压力；
- `lts__t_bytes.sum`：L2 slice 总流量；
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum`：global load 进入 L1/TEX 的字节；
- global-load instruction/request 数；
- shared load/store throughput 和 bank conflict；
- registers/thread、local-memory load/store；
- active warps/SM 和 active CTA/SM。

不同 NCU 版本的具体 metric 名称可能变化，先运行：

```bash
ncu --query-metrics | rg 'dram__bytes_read|global_op_ld|lts__t_bytes'
```

再对同一 case、同一 warmup 条件比较 fused-64 和 fused-128。

## 11. 后续可继续降低 read 的方向

### 11.1 GVA 的 Q/K 跨 value-head 复用

`parallel_gva` 中 4 个 value head 共享一个 Q/K head。当前 full-value CTA 仍按 value head 加载 Q/K，因此 L2 之外还有跨 CTA 的重复逻辑读取。

可选方案：

- 按 `(chunk,qk_head)` 预计算 raw $QK^T$ workspace；
- 一个 grouped CTA 连续处理共享 Q/K 的多个 value head；
- 使用 CTA cluster/shared-memory multicast（需要硬件和 TileLang 支持）。

预计算 workspace 会引入额外 HBM 写读，只有 profile 证明 Q/K/QK 重复明显时才值得采用。

### 11.2 选择性 async prefetch

若 NCU 显示 global-memory dependency stall 较高，可只双缓冲下一 chunk 的 K/V/A，Q 延后加载或与当前 output 交叠。选择性方案比全量 145 KiB shared-memory 方案更容易保持 occupancy。

### 11.3 gate_last 的 shared 复用

当前 `gate_last` 仍可进一步直接从 `gate_shared` 的最后一个有效 row 读取，避免每 chunk 一个标量 global load。该收益很小，应在确保动态尾块 shared index 能稳定 lowering 后再实现。

## 12. 结论

本次内存优化的核心不是简单增加 shared memory，而是扩大 CTA 的数据所有权并按生命周期复用：

1. 一个 CTA 覆盖完整 $d_v=128$，让 Q/K/A/gate 只加载一次；
2. Q/K 在 shared memory 中供多个 GEMM 使用；
3. `g_cumsum` 只从 global memory 加载一次，$\Gamma$ 和 gate ratio 从 shared copy 派生；
4. `residual_shared` 依次承担 V、residual、$\hat W$、decayed $\hat W$ 和 output staging；
5. `a_shared` 从 A buffer 复用为 causal score buffer。

结果是主要输入逻辑 global read 约下降 41.8%，主要 GEMM FLOPs 再下降 10%，`parallel_gva` 正式中位延迟从 1.467 ms 降到 1.036 ms，并且全部 8 个公开 case 保持正确。
