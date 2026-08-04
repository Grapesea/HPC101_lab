# GDN Prefill Kernel Fusion 修改方案与效率分析

## 1. 目标与边界

本文沿用 [`lab3.md`](../lab3.md) 的数学符号，重点比较 kernel fusion 前后的数学过程，以及公式中的中间量如何映射到 global memory、shared memory 和 FP32 fragment。

这里区分三个实现阶段：

1. 原始三 kernel：显式计算并保存 $U$、$W$，再分别计算 state 和 output；
2. 当前两 kernel：利用等价变换消除显式 $U/W$，但仍保存 $\hat W$ 和每个 chunk 的入口 state；
3. 目标单 kernel：公式保持不变，在一个 persistent CTA 内依次算出 $\hat W$、$O_{[c]}$ 和 $S_{[c+1]}$。

预处理不参与 fusion：

- [`preprocessing/tilelang_cumsum.py`](../preprocessing/tilelang_cumsum.py) 在计时区外生成 $g^{\text{cumsum}}$；
- [`preprocessing/tilelang_kkt_solve.py`](../preprocessing/tilelang_kkt_solve.py) 在计时区外生成 $A$；
- fused kernel 直接消费 $g^{\text{cumsum}}$ 和 $A$，不重复构造 $\Gamma$、$M$ 或求逆。

## 2. 与 `lab3.md` 一致的符号

考虑第 $c$ 个 chunk，有效长度为 $\ell\le C=64$。把 chunk 内的 Q/K/V 按 token 行堆叠：

$$
Q,K\in\mathbb R^{\ell\times d_k},\qquad
V\in\mathbb R^{\ell\times d_v},\qquad
S_{[c]}\in\mathbb R^{d_k\times d_v}.
$$

沿用 `lab3.md` 中的定义：

$$
\gamma_r=\exp(g_r^{\text{cumsum}}),\qquad
B=\operatorname{diag}(\beta_1,\ldots,\beta_\ell),\qquad
\Gamma=\operatorname{diag}(\gamma_1,\ldots,\gamma_\ell).
$$

chunk 内的 KKT 矩阵和预处理传入的逆矩阵为：

$$
M=I+\operatorname{StrictLower}(B\Gamma KK^T\Gamma^{-1}),
\qquad A=M^{-1}.
$$

$\hat W\in\mathbb R^{\ell\times d_v}$ 是逐 token 的 delta 写入量，代码变量为 `v_new`。这里要特别区分：

- 大写 $W=AB\Gamma K\in\mathbb R^{\ell\times d_k}$ 是原始 WY 分解的中间矩阵；
- 带帽的 $\hat W\in\mathbb R^{\ell\times d_v}$ 才是最终参与 output 和 state update 的写入量。

## 3. Fusion 前：显式 $U/W$ 的三 kernel 数学过程

### 3.1 Kernel 1：计算 $U$ 和 $W$

原始实现先对所有 chunk 并行计算：

$$
U=ABV,
\qquad
W=AB\Gamma K.
$$

然后把 $U\in\mathbb R^{\ell\times d_v}$ 和 $W\in\mathbb R^{\ell\times d_k}$ 写入 global memory。二者只依赖 chunk-local 输入，不依赖 $S_{[c]}$。

### 3.2 Kernel 2：计算 $\hat W$ 并推进 state

state kernel 读取 $U$、$W$ 和进入 chunk 的 $S_{[c]}$：

$$
\hat W=U-WS_{[c]}.
$$

随后按照 `lab3.md` 中的跨 chunk 递推式更新：

$$
S_{[c+1]}
=\gamma_\ell S_{[c]}
+\gamma_\ell K^T\Gamma^{-1}\hat W.
$$

此 kernel 为了让 output kernel 按 chunk 并行，还要把两个量写入 global memory：

$$
\hat W\longrightarrow \texttt{v\_new},
\qquad
S_{[c]}\longrightarrow \texttt{hidden}[c].
$$

### 3.3 Kernel 3：计算 $O_{[c]}$

output kernel 再读回 $\hat W$ 和更新前的 $S_{[c]}$，计算：

$$
O_{[c]}
=\frac1{\sqrt{d_k}}
\left[
\Gamma Q S_{[c]}
+\Gamma\operatorname{Lower}(QK^T)\Gamma^{-1}\hat W
\right].
$$

这里 $\operatorname{Lower}$ 保留下三角和对角线，对应 chunk 内的因果关系。fusion 前的数据依赖为：

```text
A,B,Γ,K,V ─→ U,W ─┐
                    ├─→ W_hat ─────────────┬─→ S_[c+1]
S_[c] ─────────────┘                       │
  │                                        │
  └──────────────→ hidden[c] ──────────────┴─→ O_[c]
```

数学上只有 chunk 间的 $S_{[c]}\to S_{[c+1]}$ 必须串行；$U/W$、`hidden` 和 `v_new` 的 global-memory 边界是实现造成的，不是公式要求的。

## 4. 第一次 Fusion：把 $U/W$ 合入 state kernel

### 4.1 等价数学变换

把 $U$、$W$ 的定义代回 $\hat W$：

> 记号校正：`lab3.md` 前文定义 $A=M^{-1}$，因此这里应为 $U=ABV$。优化段中出现的 $U=A^{-1}BV$ 是笔误，否则会变成重新左乘 $M$，与式 (3) 的求解结果不一致。

$$
\begin{aligned}
\hat W
&=U-WS_{[c]}\\
&=ABV-(AB\Gamma K)S_{[c]}\\
&=AB\left(V-\Gamma KS_{[c]}\right).
\end{aligned}
$$

因此不再需要先生成 $U$ 和 $W$。当前 `_tilelang_residual_state` 直接执行：

$$
\boxed{
\hat W=AB\left(V-\Gamma KS_{[c]}\right)
}
$$

其计算顺序是：

$$
KS_{[c]}
\ \longrightarrow\ 
V-\Gamma KS_{[c]}
\ \longrightarrow\ 
B\left(V-\Gamma KS_{[c]}\right)
\ \longrightarrow\ 
A\left[B\left(V-\Gamma KS_{[c]}\right)\right].
$$

乘法次序不能改成 $BA(\cdot)$，因为一般情况下 $AB\ne BA$。代码中的正确映射是先逐 token row 乘 $\beta_r$，再执行 `A @ residual`。

### 4.2 Fusion 后仍保留的 kernel 边界

第一次 fusion 后，$U/W$ 不再存在于 global memory，但数学上的 $\hat W$ 仍然存在：

```text
K,S_[c] ─→ K S_[c] ─┐
V,Γ ─────────────────┼─→ V-ΓKS_[c] ─→ B(·) ─→ A(·) ─→ W_hat
                     │                                  ├─→ S_[c+1]
                     │                                  └─→ v_new(HBM)
S_[c] ───────────────┴────────────────────────────────────→ hidden[c](HBM)
```

然后独立 output kernel 仍按原式计算：

$$
O_{[c]}
=\frac1{\sqrt{d_k}}
\left[
\Gamma Q S_{[c]}
+\Gamma\operatorname{Lower}(QK^T)\Gamma^{-1}\hat W
\right].
$$

所以当前实现是“两 kernel”：

1. residual-state kernel：算 $\hat W$ 和 $S_{[c+1]}$，写出 `v_new`、`hidden`；
2. output kernel：读回 `v_new`、`hidden`，算 $O_{[c]}$。

## 5. 第二次 Fusion：在一个 persistent kernel 内完成 $\hat W$、$O$ 和 $S$

### 5.1 完全融合前后的公式没有变化

第二次 fusion 不再引入新的代数变换，只改变同一组公式的求值位置。对每个 chunk，persistent CTA 持有 $S_{[c]}$，依次计算：

$$
\boxed{
\begin{aligned}
\hat W
&=AB\left(V-\Gamma KS_{[c]}\right),\\
O_{[c]}
&=\frac1{\sqrt{d_k}}
\left[
\Gamma Q S_{[c]}
+\Gamma\operatorname{Lower}(QK^T)\Gamma^{-1}\hat W
\right],\\
S_{[c+1]}
&=\gamma_\ell S_{[c]}
+\gamma_\ell K^T\Gamma^{-1}\hat W.
\end{aligned}
}
$$

完整 fusion 消除的是 $\hat W$ 和 $S_{[c]}$ 的全局物化：

- $S_{[c]}$ 保持在 FP32 `state_fragment`，只在 chunk 之间由同一 CTA 传递；
- $\hat W$ 只在 FP32 fragment/BF16 shared buffer 中短暂存在；
- $O_{[c]}$ 和 $S_{[c+1]}$ 都消费 CTA 内的同一份 $\hat W$；
- global memory 只接收最终 output 和整条 chunk chain 结束后的 final state。

### 5.2 为什么必须先算 output，再覆盖 state

$O_{[c]}$ 的第一项是 $\Gamma QS_{[c]}$，不是 $\Gamma QS_{[c+1]}$。因此不能先更新 state 再计算 output。合法的 chunk 内顺序是：

$$
\begin{array}{rcl}
X_Q&=&QS_{[c]},\\
X_K&=&KS_{[c]},\\
\hat W&=&AB(V-\Gamma X_K),\\
O_{[c]}&=&\dfrac1{\sqrt{d_k}}
\left[\Gamma X_Q+\Gamma\operatorname{Lower}(QK^T)\Gamma^{-1}\hat W\right],\\
S_{[c+1]}&=&\gamma_\ell S_{[c]}+\gamma_\ell K^T\Gamma^{-1}\hat W.
\end{array}
$$

$X_Q$ 和 $X_K$ 只是为了说明执行顺序而引入的临时记号，不需要写入 global memory。依赖图变为：

```text
                         ┌─→ Γ Q S_[c] ─────────────────────────┐
S_[c] ───────────────────┤                                      ├─→ O_[c]
                         └─→ K S_[c] ─→ W_hat ─→ intra-chunk ───┤
                                                │               │
                                                └───────────────┴─→ S_[c+1]
```

在 $O_{[c]}$ 完成、$\hat W$ 已准备好之后，才允许把 `state_fragment` 从 $S_{[c]}$ 更新为 $S_{[c+1]}$。

### 5.3 跨 chunk 仍然串行，chunk 内全部就地消费

fusion 没有消除以下真实依赖：

$$
S_{[0]}\longrightarrow S_{[1]}\longrightarrow\cdots
\longrightarrow S_{[N_{\text{chunk}}]}.
$$

它消除的是每个箭头内部不必要的 kernel/显存边界：

```text
Fusion 前：S_[c] → state kernel → HBM → output kernel → O_[c]
Fusion 后：S_[c] → persistent CTA 内完成 W_hat/O_[c]/S_[c+1]
```

只要一个 CTA 负责同一 `(batch,value_head,value_tile)` 的完整 chunk chain，就不需要跨 CTA 同步。若一个 CTA 覆盖完整 $d_v=128$，同一份 $QK^T$ 可以与 $\hat W$ 的全部 value columns 相乘，不再为两个 64-wide V tile 重算。

## 6. 当前版本的主要额外开销

### 6.1 全局中间张量

按每个 `(chunk,value_head)` 计算：

| 中间量 | Shape | 大小 | 当前逻辑流量 |
|---|---:|---:|---:|
| `V_new` | `64×128` BF16 | 16 KiB | 写 16 KiB + 读 16 KiB |
| `hidden` | `128×128` BF16 | 32 KiB | 写 32 KiB + 读 32 KiB |
| 合计 |  | 48 KiB | **96 KiB/chunk/head** |

完全融合后，两者都只在 CTA 内短暂存在，以上 96 KiB 逻辑往返可以全部取消。

以公开 case `parallel_gva` 为例，`B=1,T=2048,Hv=16`，共有 32 个 chunk：

- `V_new` 工作集为 8 MiB；
- `hidden` 工作集为 16 MiB；
- 合计取消 24 MiB 中间分配和约 48 MiB 逻辑读写。

这里的“逻辑流量”不等于必然发生的 DRAM 流量，因为 L2 cache 可能命中，但它仍会占用 L2、load/store pipeline 和地址计算资源。

### 6.2 重复读取和重复 QK

当前 `BLOCK_V=64`，一个 value head 由两个 output CTA 处理。每个 CTA 都会完整计算一次：

$$
QK^T\in\mathbb R^{64\times64}.
$$

因此同一 `(chunk,value_head)` 的 raw QK 被计算两次。GVA 中不同 value head 的 gate 不同，所以 gated score 不能直接跨 value head 共享；但同一 value head 的两个 V tile 一定可以共享 raw QK。

当前 residual-state 和 output kernel 还分别读取 K。按逻辑字节数，一个 chunk/value-head 的 K 会在两个 V tile、两个 kernel 中被读取最多 4 次。融合且使用完整 V tile 后只需加载一次。实际 DRAM 节省取决于 Q/K 的 L2 复用，但 shared-memory 搬运和指令不会由 cache 自动消失。

## 7. 修改方案 A：先融合，保持 `BLOCK_V=64`

这是最稳妥的第一步。grid 和 state 分片保持不变：

```python
with T.Kernel(
    HEAD_DIM_V // 64,
    batch_size * H,
    threads=128,
) as (bv, bbh):
    ...
```

每个 CTA 仍负责 state 的 64 个 value columns，并串行遍历所有 chunk。

### 7.1 接口修改

把当前两个 prim func 合并为一个 `_tilelang_fused_forward_64`：

```text
输入：q, k, v, g, beta, A, initial_state, num_chunks
输出：output, final_state
```

删除以下全局张量和调用：

- `v_new = torch.empty_like(v)`；
- `hidden = torch.empty(...)`；
- `_tilelang_chunk_output(...)` 的编译和 launch；
- state kernel 中对 `hidden`、`v_new` 的写回。

### 7.2 chunk 内调度顺序

在现有 `_tilelang_residual_state` 的 chunk loop 中插入 output 路径：

1. 从 FP32 `state_fragment` 得到 BF16 `state_shared`；
2. 加载 Q/K，并先计算 `Q @ state_shared` 到 FP32 `output_fragment`；
3. 计算 `K @ state_shared` 到 FP32 `value_fragment`；
4. 形成 $B(V-\Gamma KS_{[c]})$，转为 BF16 shared operand；
5. 左乘 $A$ 得到 FP32 $\hat W$；
6. 计算 raw $QK^T$，再形成 $\Gamma\operatorname{Lower}(QK^T)\Gamma^{-1}$；
7. 计算 $\Gamma\operatorname{Lower}(QK^T)\Gamma^{-1}\hat W$，与 $\Gamma QS_{[c]}$ 相加后写出 $O_{[c]}$；
8. 对同一份 $\hat W$ 施加 $\gamma_\ell\Gamma^{-1}$ 的 row decay；
9. 执行 $K^T(\gamma_\ell\Gamma^{-1}\hat W)$，并与 $\gamma_\ell S_{[c]}$ 相加；
10. 进入下一个 chunk。

对应伪代码为：

```python
for chunk in T.serial(num_chunks):
    load_q_k_v_a_gate_beta()

    # 必须在更新 state 之前完成。
    T.gemm(q_shared, state_shared, output_fragment, clear_accum=True)
    apply_row_scale(output_fragment, gamma)  # Γ Q S_[c]

    T.gemm(k_shared, state_shared, value_fragment, clear_accum=True)
    residual = beta * (v - gamma * value_fragment)  # B(V-ΓKS_[c])
    T.gemm(a_shared, residual_shared, value_fragment, clear_accum=True)
    T.copy(value_fragment, hat_w_shared)  # W_hat

    T.gemm(
        q_shared,
        k_shared,
        auxiliary_fragment,
        transpose_B=True,
        clear_accum=True,
    )
    apply_gamma_lower_gamma_inverse(auxiliary_fragment)
    T.gemm(scores_shared, hat_w_shared, output_fragment)
    write_scaled_output(output_fragment)

    apply_gamma_last_gamma_inverse(value_fragment)
    T.copy(value_fragment, hat_w_shared)
    scale_old_state(state_fragment, gamma_last)
    T.gemm(k_shared, hat_w_shared, state_fragment, transpose_A=True)
    T.copy(state_fragment, state_shared)
```

### 7.3 buffer 复用

不能简单地把两个现有 kernel 的所有 buffer 相加，否则 register 和 shared-memory 压力过大。建议按生命周期复用：

- `residual_shared` 在 `A @ residual` 后复用为保存 $\hat W$ 的 `hat_w_shared`；
- `a_shared` 在 `A @ residual` 后复用为 `scores_shared`，两者都是 `64×64` BF16；
- 当前 `input_fragment` 在 residual 形成后已经失效，可复用为 `scores_fragment`；
- 删除单独的 `value_shared`，让 residual、$\hat W$、$\gamma_\ell\Gamma^{-1}\hat W$ 共用一个 `64×64` BF16 shared buffer；
- `output_fragment` 从 `Q @ state` 开始一直保留到 output 写回，之后才释放其生命周期。

复用后的主要 shared-memory 估算为：

| Buffer | 大小 |
|---|---:|
| `state_shared[128,64]` BF16 | 16 KiB |
| `k_shared[64,128]` BF16 | 16 KiB |
| `q_shared[64,128]` BF16 | 16 KiB |
| `residual/hat_w_shared[64,64]` BF16 | 8 KiB |
| `A/scores_shared[64,64]` BF16 | 8 KiB |
| gate、beta、decay | 约 1 KiB |
| 合计 | **约 65 KiB** |

若不复用 buffer，shared memory 容易接近或超过 90 KiB。65 KiB 通常更容易部署，但实际 occupancy 仍取决于目标 GPU 的每 SM shared-memory 上限和寄存器使用量。

### 7.4 方案 A 的收益与限制

收益：

- kernel launch 从 2 次降为 1 次；
- 取消 `hidden/V_new` 的 96 KiB/chunk/head 逻辑往返；
- K、gate 可以在 state、output 和 state-update 路径间复用；
- 不改变 CTA 数量，低 head case 仍有 `2×B×Hv` 个 persistent CTA。

限制：

- 两个 V tile 仍分别计算 QK，主要 FLOPs 不变；
- output 工作从 chunk-parallel kernel 搬入串行 chunk loop，增加了 persistent CTA 的 critical path；
- 增加 `scores_fragment` 和 `output_fragment` 后可能发生 register spill。

因此方案 A 的第一验证指标不是“理论 FLOPs 降低”，而是中间流量、launch 和 cache 压力减少后是否抵消更长的 persistent critical path。

## 8. 修改方案 B：一个 CTA 覆盖完整 `D_v=128`

在方案 A 正确后，再增加 `_tilelang_fused_forward_128`。grid 改为一个 CTA 对应一个 `(batch,value_head)`：

```python
with T.Kernel(batch_size * H, threads=256) as (bbh,):
    bb = bbh // H
    bh = bbh % H
```

主要 tile 变为：

```text
state_fragment       [128,128] FP32
state_shared         [128,128] BF16
residual/hat_W       [64,128]
output_fragment      [64,128] FP32
scores               [64,64]
```

### 8.1 为什么完整 V tile 更适合 fusion

Q、K、gate 和 raw score 都与 value column 无关。一个 CTA 覆盖 128 个 value columns 后：

- `QK^T` 从每 head 两次降为一次；
- Q/K/A/gate 每个 chunk 只需由一个 CTA 加载；
- 不需要两个 CTA 对同一个 head 做重复控制和地址计算；
- $\hat W[64,128]$ 可直接服务 $O_{[c]}$ 和 $S_{[c+1]}$。

### 8.2 线程数选择

建议分两步：

1. 先用 256 threads 和 TileLang 高层 `T.gemm` 验证完整 V tile；
2. 若 register pressure 或 MMA 调度成为瓶颈，再尝试 Hopper 上的 512-thread warp-specialized 版本。

512-thread 版本可以概念上划分为：

| Warp group | 主要职责 |
|---|---|
| WG0 | global/shared load、gate/beta、output store |
| WG1 | `Q@S`、`K@S` 等 state 相关 MMA |
| WG2 | residual、$A(B\cdot residual)$、chunk-local output |
| WG3 | `QK^T`、causal gate、state-update 辅助 |

这需要显式 shared-memory handoff 和 barrier。不能只把 `threads=128` 改成 512 就假设会自动获得 warp specialization；若继续使用高层 `T.gemm`，应先检查 TileLang 生成代码中的 warp 分工和寄存器分配。

### 8.3 shared-memory 估算

通过与方案 A 相同的生命周期复用，完整 V tile 的主要 shared memory 为：

| Buffer | 大小 |
|---|---:|
| `state_shared[128,128]` BF16 | 32 KiB |
| `q_shared[64,128]` BF16 | 16 KiB |
| `k_shared[64,128]` BF16 | 16 KiB |
| `residual/hat_W[64,128]` BF16 | 16 KiB |
| `A/scores[64,64]` BF16 | 8 KiB |
| gate、beta、decay | 约 1 KiB |
| 合计 | **约 89 KiB** |

这通常适合 shared-memory 容量较大的 Hopper，但可能限制每 SM 同时驻留的 CTA 数。不要在第一版中把所有 buffer 双缓冲；对 Q/K/V/A 全量 double buffering 会再增加约 56 KiB，可能把 shared-memory 使用推到约 145 KiB。

推荐先做单 stage，profile 证明 global-to-shared latency 明显后，再只对下一 chunk 的 K/V/A 做选择性 async prefetch。

## 9. FLOPs 分析

令：

- chunk size $C=64$；
- $d_k=d_v=128$；
- value tile 数 $r_v=d_v/B_V$；
- 一次乘加按 2 FLOPs 计算。

采用 residual 公式后的主要 GEMM FLOPs 为：

$$
F(r_v)
=6Cd_kd_v+4C^2d_v+2r_vC^2d_k.
$$

各项分别来自：

- $KS_{[c]}$、$QS_{[c]}$、$K^T(\gamma_\ell\Gamma^{-1}\hat W)$：$6Cd_kd_v$；
- $AB(V-\Gamma KS_{[c]})$ 中的 dense $A$ GEMM，以及 chunk-local output GEMM：$4C^2d_v$；
- 每个 V tile 重算的 `QK^T`：$2r_vC^2d_k$。

代入固定维度：

| 实现 | 主要 FLOPs/chunk/head | 相对最初显式 W/U |
|---|---:|---:|
| 最初显式 W/U，`B_V=64` | 11,534,336 | 100% |
| 当前 residual 两 kernel，`B_V=64` | 10,485,760 | 90.9% |
| 方案 A：fused，`B_V=64` | 10,485,760 | 90.9% |
| 方案 B：fused，`B_V=128` | 9,437,184 | **81.8%** |

方案 A 不降低 GEMM FLOPs，收益来自中间流量和 launch；方案 B 再消除一次 QK，较当前 residual 版本减少：

$$
2C^2d_k=1,048,576\ \text{FLOPs/chunk/head},
$$

即主要 GEMM FLOPs 再下降 10%。相对最初显式 W/U 版本，总主要 FLOPs 下降 18.2%。

## 10. 预计端到端效率

不能把 18.2% FLOPs 降低直接等同于 18.2% latency 降低，原因包括：

- state recurrence 使 chunk loop 串行；
- 小矩阵 GEMM 的效率受到 MMA tile、同步和 occupancy 影响；
- exponent、mask、BF16/FP32 转换不是上述 FLOPs 模型的一部分；
- 完整 V tile 减少 CTA 数，可能降低低 head case 的 SM 覆盖率；
- register spill 可以完全抵消 HBM 流量收益。

更合理的上界/下界判断是：

| 收益来源 | 方案 A | 方案 B |
|---|---:|---:|
| 取消 `hidden/V_new` HBM 往返 | 是 | 是 |
| 少一次 kernel launch | 是 | 是 |
| QK FLOPs 减半 | 否 | 是 |
| Q/K/A/gate 跨 V tile 复用 | 部分 | 最充分 |
| persistent CTA 数 | `2×B×Hv` | `B×Hv` |
| register/shared-memory 风险 | 中 | 高 |

若原两-kernel版本确实由中间流量和重复 QK 主导，方案 B 最有可能获得明显加速；若 case 由长 chunk chain 和低并行度主导，则完全融合可能只是把原本并行的 output 工作串行化到同一条 chain 上。

## 11. 必须做自适应选择

公开 case 的 `B×Hv` 差异很大：

| Case | `B×Hv` | `BLOCK_V=64` CTA 数 | `BLOCK_V=128` CTA 数 | 建议起点 |
|---|---:|---:|---:|---|
| `chain_equal` | 4 | 8 | 4 | 保留两-kernel或 fused-64 |
| `long_low_gva` | 8 | 16 | 8 | 优先 fused-64 |
| `parallel_equal` | 16 | 32 | 16 | 尝试 fused-128 |
| `parallel_gva` | 16 | 32 | 16 | 优先 fused-128 |
| `batch_split_gva` | 32 | 64 | 32 | fused-128 |
| `wide_gva_state` | 64 | 128 | 64 | fused-128 |
| `deep_gva_state` | 32 | 64 | 32 | fused-128 |

上表只是以约 14 SM 的课程 profiling 环境为参考的初始选择，不应硬编码为最终结论。建议在 Python wrapper 中保留三个已编译变体：

```text
residual_state + parallel_output  # 低 head、超长 chain
fused_forward_64                  # 中等并行度
fused_forward_128                 # head 并行度足够
```

选择条件至少考虑：

- `batch_size * num_heads_v`；
- `num_chunks`；
- 是否有 initial state；
- 实测的目标 GPU SM 数和每 CTA 资源占用。

不要仅凭 `T` 很长就选择 fused-128。长序列会加重 persistent critical path；只有 head 级 CTA 足以覆盖 SM 时，长 loop 才不会造成严重设备空闲。

## 12. 数值和正确性要求

fusion 不改变实数数学，但改变 BF16 舍入位置和 GEMM 累加次序。建议保持：

- recurrent `state_fragment` 为 FP32；
- `K@S`、`Q@S`、state update 的 accumulator 为 FP32；
- residual subtraction、beta、gate exponent 为 FP32；
- residual 在进入 `A@residual` Tensor Core GEMM 前转 BF16；
- $\hat W$ 在进入 chunk-local output GEMM 和 $K^T(\gamma_\ell\Gamma^{-1}\hat W)$ 前转 BF16；
- output 最终写回 BF16，final state 写回 FP32；
- 直接计算因果区域的 $\gamma_i/\gamma_j=\exp(g_i^{\text{cumsum}}-g_j^{\text{cumsum}})$，不要拆成可能溢出的两个指数。

尾块必须继续满足：

- 无效 Q/K/V row 补零；
- 无效 `A` row 不参与计算；
- $\gamma_\ell$ 使用最后一个有效 token 的 $g_\ell^{\text{cumsum}}$；
- output 只写有效 row；
- state update 不包含 padding row。

## 13. 推荐实施顺序

### P1：shared-buffer 清理

先在现有 residual-state kernel 中合并 `residual_shared` 和 `value_shared` 的生命周期。该修改不改变 kernel 边界，可单独验证数值和 shared-memory 降低是否生效。

### P2：实现 fused-64

- 把 `q`、`output` 加入 state kernel 接口；
- 在 state update 前加入 $QS_{[c]}$、$QK^T$ 和 $\Gamma\operatorname{Lower}(QK^T)\Gamma^{-1}\hat W$；
- 复用 `A` buffer 保存 scores；
- 删除 `hidden/V_new` 全局写回和 output launch；
- 先用 128 threads 保持当前 MMA mapping。

### P3：实现 fused-128

- CTA grid 去掉 `bv` 维；
- value/state/output tile 扩到 128；
- 先尝试 256 threads；
- 确认同一 chunk 只计算一次 QK；
- 与 fused-64 比较 register spill、shared-memory 和 active CTA。

### P4：选择性 pipeline 和 warp specialization

- 用异步搬运预取下一 chunk 的部分输入；
- 不要在没有 profile 证据时双缓冲全部 Q/K/V/A；
- Hopper 上再评估 512-thread warp specialization；
- 如果 `local_load/store` 或 register spill 上升，回退到较小 tile。

### P5：自适应 dispatch

对所有公开 case 分别记录三种变体的正确性和 median latency，再按 `B×Hv`、`num_chunks` 建立简单 dispatch。低 head 超长序列若仍然受串行 chain 限制，需要进一步做 chunk affine transfer/context parallel，而不是继续把更多工作塞进同一个 CTA。

## 14. 验证与 profiling 清单

每个阶段都运行：

```bash
python run.py --case short_tail_state --warmup 10 --repetitions 100
python run.py --case parallel_gva --warmup 10 --repetitions 100
python run.py --warmup 10 --repetitions 100
```

至少记录：

- output 最大绝对/相对误差；
- final state 最大绝对/相对误差；
- median latency；
- registers/thread；
- local-memory load/store，即 register spill；
- shared memory/CTA；
- active CTA 和 active warps/SM；
- Tensor Core utilization；
- DRAM/L2 bytes；
- kernel launch 数量。

判断顺序应是：先保证所有 case 正确，再确认没有 spill，最后比较 latency。若 fused-128 的理论 FLOPs 更少但更慢，首先检查 CTA 覆盖率、shared-memory occupancy 和 spill，而不是继续增加 pipeline stage。

## 15. 结论

当前 residual 公式已经给 kernel fusion 准备好了正确的数据流。下一步最合理的路线是：

1. 用 fused-64 取消 `hidden` 和 `V_new` 的 HBM 往返，把两个 kernel 合为一个；
2. 用 fused-128 让同一 CTA 覆盖完整 value 维，将 QK 从两次降为一次；
3. 对低 head 长序列保留较高 CTA 数或两-kernel fallback，不无条件采用完整 V tile；
4. 在确认 shared-memory 与 register 资源后，再引入 async pipeline 和 512-thread warp specialization。

这一方案在数学上保持严格等价。理论上，完全融合相对最初显式 W/U 路径减少 18.2% 的主要 GEMM FLOPs，并取消当前 residual 两-kernel 路径的 96 KiB/chunk/head 中间逻辑流量；实际加速需要通过各 case 的 occupancy、spill 和端到端 benchmark 决定。
