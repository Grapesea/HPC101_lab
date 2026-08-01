# GDN Prefill 流程与 TileLang 实现说明

## 1. 当前 `student/` 的状态

当前 [`student/tilelang_fwd.py`](../student/tilelang_fwd.py) 还没有真正的 TileLang kernel，只是在 `gdn_prefill_forward` 中调用 PyTorch reference。最终需要替换的是这次调用，而不是前面的 gate 前缀和与 KKT 求解。

完整 GDN forward 的数据流为：

```text
raw_g
  │ chunk 内前缀和
  ▼
g_cumsum ─┐
K, beta ──┴─ KKT solve ──► A
                             │
Q, K, V, g_cumsum, beta, A, initial_state
                             │
                             ▼
                 student GDN Prefill core
                             │
                       output, final_state
```

其中：

- `chunk_local_cumsum`：每 64 个 token 独立计算 gate 的前缀和，参见 [`preprocessing/tilelang_cumsum.py`](../preprocessing/tilelang_cumsum.py)。
- `kkt_solve`：计算每个 chunk 的下三角逆矩阵 `A`，参见 [`preprocessing/tilelang_kkt_solve.py`](../preprocessing/tilelang_kkt_solve.py)。
- `student` 只负责后面的 `W/U/state/output`。核心 benchmark 中，前两步在计时区外完成，参见 [`run.py`](../run.py) 中的 `_run_core_case`。

GDN 的关键思想是：跨 chunk 的 state 必须按时间顺序推进，但一个 64-token chunk 内的递归可以借助 `A` 转换成矩阵乘法并行计算。

## 2. 输入、输出与张量形状

接口张量为：

| 张量 | Shape | Dtype | 含义 |
|---|---:|---|---|
| `q`, `k` | `[B, T, Hq, 128]` | BF16 | query 和 key |
| `v` | `[B, T, Hv, 128]` | BF16 | value |
| `g_cumsum` | `[B, T, Hv]` | FP32 | chunk-local log gate 前缀和 |
| `beta` | `[B, T, Hv]` | FP32 | delta update 强度 |
| `A` | `[B, T, Hv, 64]` | BF16 | 每个 chunk 的下三角逆矩阵 |
| `initial_state` | `[B, Hv, 128, 128]` | FP32 | 可选的初始 recurrent state |
| `output` | `[B, T, Hv, 128]` | BF16 | token 输出 |
| `final_state` | `[B, Hv, 128, 128]` | FP32 | 最后一个 chunk 后的 state |

`A` 虽然存储为 `[B,T,Hv,64]`，但对于从 `start` 开始、有效长度为 `L` 的 chunk，实际使用的是：

```python
A[:, start:start + L, :, :L]
```

也就是每个 head 对应一个 `L×L` 小矩阵。

### GVA head 映射

当 `Hv > Hq` 时，多个 V head 共享一个 Q/K head。对于 value head `h`：

```python
hq = h // (Hv // Hq)
```

不需要真的执行 `repeat_interleave`。预处理 kernel 中也是通过这个下标映射读取 K。

## 3. 一个 chunk 内的 GDN 计算

设：

- chunk 有效长度为 `L≤64`；
- `Dk=Dv=128`；
- 当前 value head 为 `h`；
- `S` 是进入 chunk 时的 state，shape 为 `[128,128]`；
- `G` 是该 chunk 内的 `g_cumsum`。

### 3.1 矩阵 `A` 的来源

KKT 预处理首先计算 key Gram 矩阵：

\[
\operatorname{Gram}_{ij}=\langle k_i,k_j\rangle
\]

然后构造单位下三角矩阵：

\[
M_{ij}=
\begin{cases}
1,&i=j\\
\beta_i\langle k_i,k_j\rangle e^{G_i-G_j},&i>j\\
0,&i<j
\end{cases}
\]

最后求：

\[
A=M^{-1}
\]

`A` 把 chunk 内原本逐 token 依赖的 delta-rule 更新转换成并行矩阵运算。

### 3.2 计算 W

先构造带 beta 和 gate 的 key：

\[
\bar K_j=\beta_j e^{G_j}K_j
\]

然后：

\[
W=A\bar K
\]

`W` 的 shape 为 `[L,128]`。

### 3.3 计算 U

构造带 beta 的 value：

\[
\bar V_j=\beta_jV_j
\]

然后：

\[
U=A\bar V
\]

`U` 的 shape 同样为 `[L,128]`。

### 3.4 扣除旧 state 的预测

\[
V_{\text{new}}=U-WS
\]

`V_new` 是当前 chunk 中真正需要写入 state 的有效 value update，shape 为 `[L,128]`。

### 3.5 计算 chunk 内因果贡献

先计算 Q/K 分数并加入 gate 衰减：

\[
P_{ij}=1_{j\le i}\langle q_i,k_j\rangle e^{G_i-G_j}
\]

即：

```text
P = tril(Q @ Kᵀ) × decay
```

上三角必须清零，因为 token 只能看到自己和之前的 token。

### 3.6 计算输出

输出由旧 state 贡献和当前 chunk 贡献两部分组成：

\[
O=\frac{1}{\sqrt{128}}
\left[
e^G\odot(QS)+PV_{\text{new}}
\right]
\]

其中：

- `exp(G) × Q @ S`：进入当前 chunk 前的 state 对输出的贡献；
- `P @ V_new`：当前 chunk 内已经发生的更新对输出的贡献。

### 3.7 更新跨 chunk state

设 `G_last=G[L-1]`，则：

\[
S_{\text{next}}
=
e^{G_{\text{last}}}S
+
K^\mathsf{T}
\left(
e^{G_{\text{last}}-G}\odot V_{\text{new}}
\right)
\]

计算完成后，下一个 chunk 使用 `S_next`。

依赖关系为：

```text
S_c ──► Vnew_c ──► O_c
 │          │
 └──────────┴────► S_(c+1)
```

因此 chunk 内可以并行，chunk 之间必须保持时间顺序。

## 4. 将计算映射到 TileLang

GDN Prefill 的主体可以看成一串固定规模的小 GEMM：

| 数学计算 | TileLang 实现 |
|---|---|
| `A @ K_scaled` | `T.gemm` 得到 `W` |
| `A @ V_scaled` | `T.gemm` 得到 `U` |
| `W @ S` | `T.gemm` 得到旧 state 的预测 |
| `Q @ S` | `T.gemm` 得到 state 输出 |
| `Q @ Kᵀ` | `T.gemm(..., transpose_B=True)` |
| `P @ V_new` | `T.gemm` 得到 chunk 内输出 |
| `K_decayᵀ @ V_new` | 转置布局后使用 `T.gemm` 更新 state |

这与 [`ref_docu_repo/tilelang/example_gemm.py`](tilelang/example_gemm.py) 的结构一致：

- `@tilelang.jit`：定义并缓存编译后的 kernel；
- `T.Kernel`：定义 GPU grid 和每个 block 的线程数；
- `T.alloc_shared`：缓存当前 tile；
- `T.alloc_fragment(..., float32)`：保存 GEMM 的 FP32 累加结果；
- `T.copy` 或 `T.async_copy`：在 global memory 和 shared memory 之间搬运数据；
- `T.gemm`：执行矩阵乘加；
- `T.Parallel`：执行 beta、指数、mask 等逐元素操作；
- `T.Pipelined`：重叠下一批数据搬运和当前计算。

`preprocessing/tilelang_kkt_solve.py` 还给出了几个与本题直接相关的模式：

- 使用 `T.dynamic` 表示动态的 batch size 和 sequence length；
- 把 `(batch, chunk, head)` 展平到一维 block id；
- 用 `T.async_copy` 搬运完整 chunk；
- 用 `T.gemm(..., transpose_B=True)` 计算 `K @ Kᵀ`；
- 用 `T.Parallel` 加入 gate、beta、上下三角 mask；
- 对不足 64 个 token 的尾块进行补零和带条件写回。

## 5. 推荐的 TileLang kernel 分层

### 5.1 第一版：两个 kernel

第一版可以优先保证正确性，将计算拆成两个阶段。

#### Kernel 1：并行计算 W 和 U

grid 可以展平为：

```text
B × num_chunks × Hv
```

每个 block 负责一个 `(batch, chunk, value_head)`：

```text
K_scaled = beta × exp(G) × K
V_scaled = beta × V
W = A @ K_scaled
U = A @ V_scaled
```

所有 chunk 的 `W/U` 都不依赖输入 state，因此可以完全并行。

#### Kernel 2：按顺序推进 state

推荐 grid：

```text
B × Hv × ceildiv(128, BLOCK_V)
```

每个 block 持有 state 的一个 value-column tile：

```text
S_tile = S[:, v_start:v_end]

for chunk in T.serial(num_chunks):
    Vnew = U - W @ S_tile
    P = causal_decay(Q @ Kᵀ)
    O = scale × (exp(G) × Q @ S_tile + P @ Vnew)
    S_tile = exp(G_last) × S_tile + K_decayᵀ @ Vnew
```

同一个 block 必须按顺序处理属于自己的所有 chunk。不能简单地让每个 chunk 使用独立 block，因为普通 CUDA grid 中不存在可以安全推进 state 的跨 block 全局同步。

### 5.2 为什么要沿 value 维切 state

一个完整 FP32 state 占用：

```text
128 × 128 × 4 bytes = 64 KiB
```

如果完整 state 和 `Q/K/A/W/V_new` 同时进入 shared memory，通常会超过合理的 shared-memory 使用量，并降低 occupancy。

因此可以选择：

```text
BLOCK_V = 32 或 64
```

- `BLOCK_V=32` 时，一个 state tile 占 16 KiB；
- `BLOCK_V=64` 时，一个 state tile 占 32 KiB。

较小的 `BLOCK_V` 会产生更多 blocks、提高低 head 数情况下的并行度，但也可能重复读取 Q/K/W。

### 5.3 优化版：融合 W/U

进一步优化时，可以让 persistent state kernel 在每个 chunk 内直接计算 `W/U`，避免：

- 分配大尺寸的临时 `W/U` 张量；
- 将 `W/U` 写入 global memory 后再读回来。

代价是：沿 value 维切成多个 block 后，和 value tile 无关的 `W=A@K_scaled` 可能被重复计算。

因此建议先完成两-kernel 版本，再根据测试 case 比较：

- 额外 global-memory 流量；
- 重复 GEMM 开销；
- shared-memory 占用；
- register pressure；
- 低 head 数时的 GPU occupancy。

## 6. 一个结构化的 TileLang 伪代码

下面只展示调度结构，不是可直接运行的完整实现：

```python
@tilelang.jit
def gdn_state_kernel(Hq, Hv, block_v, qk_dtype, gate_dtype):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def kernel(q, k, g, W, U, initial_state, output, final_state):
        num_chunks = T.ceildiv(num_tokens, 64)

        with T.Kernel(
            batch_size * Hv * T.ceildiv(128, block_v),
            threads=128,
        ) as (pid,):
            # pid -> batch、value head、value-column tile
            bb = ...
            hv = ...
            value_tile = ...
            hq = hv // (Hv // Hq)

            state_fragment = T.alloc_fragment((128, block_v), "float32")
            # 从 initial_state 复制，或在 initial_state=None 时清零。

            for chunk in T.serial(num_chunks):
                left = chunk * 64
                length = T.min(64, num_tokens - left)

                # global -> shared
                # 加载 Q/K/G/W/U，并对尾块补零。

                # Vnew = U - W @ state
                # scores = Q @ K^T
                # scores *= exp(G_i-G_j)，并清除上三角
                # output = scale * (exp(G) * Q @ state + scores @ Vnew)
                # state = exp(G_last) * state + K_decay^T @ Vnew

                # 仅写回有效 token。

            # 将 state_fragment 写入 final_state 对应的列。

    return kernel
```

实际实现中需要根据 `T.gemm` 的输入布局，把参与 GEMM 的矩阵放进合适的 shared buffer，并使用 FP32 fragment 累加。

## 7. 精度策略

推荐采用：

- Q/K/V/A：保持输入 BF16；
- GEMM accumulator：FP32；
- `g_cumsum`、`beta`、指数运算：FP32；
- recurrent state：FP32；
- `V_new` 和 output 中间结果：尽量 FP32；
- 最终 output 写回时转为 BF16；
- final state 保持 FP32。

指数运算可以参考 KKT kernel：

```python
T.exp2(x * LOG2E)
```

并通过：

```python
tilelang.PassConfigKey.TL_ENABLE_FAST_MATH
```

启用 fast math。不过在确认正确性之前，不宜过早将 gate、state 或关键中间结果降为 BF16。

## 8. 最容易出错的地方

### 尾块

测试包含 `T=1025`，最后一个 chunk 的有效长度只有 1。需要保证：

- 无效 Q/K/V/W/U 位置补零；
- 无效 token 不写回 output；
- state update 不包含无效 token；
- `G_last` 使用 `G[L-1]`，不能固定读取 `G[63]`。

### GVA

使用：

```python
hq = hv // (Hv // Hq)
```

避免在显存中实际展开 Q/K。

### 因果 mask

`Q @ Kᵀ` 的上三角必须清零。只有 `j≤i` 的位置能进入 `P @ V_new`。

### gate 的空间

`g_cumsum` 位于 log-space，正确的 token 间衰减为：

\[
e^{G_i-G_j}
\]

不能直接把 `G_i-G_j` 当成线性 gate 使用。

### initial state

- `initial_state is None` 时从零开始；
- 存在 initial state 时，应复制到独立 state/output buffer；
- 不应原地修改调用者传入的 initial state。

### 返回类型

评测要求：

```text
output.dtype      == torch.bfloat16
final_state.dtype == torch.float32
```

并且 output 与 final state 都会参与正确性检查。

## 9. 总结

GDN Prefill 的 TileLang 实现可以概括为：

```text
用 A 把 chunk 内递归变成 GEMM，
用 persistent block 保持 chunk 间 state，
用 value-dimension tiling 控制 shared memory，
用 T.Parallel 完成 gate、mask 和尾块逻辑。
```

建议实现顺序为：

1. 先实现并验证 W/U kernel；
2. 实现按 chunk 串行推进的 state/output kernel；
3. 验证尾块、GVA 和 initial state；
4. 调整 `BLOCK_V`、线程数和 shared-memory 布局；
5. 最后尝试融合 W/U、异步拷贝和流水线。
