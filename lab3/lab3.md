# Lab 3: GDN Prefill前向优化

<center>Grapesea</center>

[TOC]

> [LLM推理优化基础知识](https://blogs.erix025.me/EfficientAI/sct-llm-talk/sct-llm-talk/)  [从零开始的 GDN 推导 - Eric025's Blog](https://blogs.erix025.me/EfficientAI/GDN_from_scratch/)
>
> 涉及论文：
>
> * [[2412.06464] Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
> * [[2406.06484] Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)
> * [[2312.06635] Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)
>
> [TileLang Documentation](https://tilelang.com/)
>
> [Nsight Compute Documentation — NsightCompute 13.3 documentation](https://archive.docs.nvidia.com/nsight-compute/2026.2/)
>
> [Python Profiling: NVIDIA Nsight Tools Feature Spotlight](https://www.youtube.com/watch?v=aQ1NYoRvp7o)
>
> [User Guide — Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#profiling-from-the-gui)性能评测包含两种计时区间：
>
> - **forward 端到端时间**：从 `raw_g` 开始，包含 `g_cumsum`、$A$、$U$、$W$、$S$ 和 $O$ 的全部计算，以及中间结果读写和 kernel launch 的开销。**这部分时间仅供参考与其他实现对比，最终评分不以此为准**。
> - **核心计算时间**：只统计学生函数内 $U$、$W$、$S$ 和 $O$ 的计算，不包括 `g_cumsum` 和 的预处理。学生函数内的张量分配、数据转换和 kernel launch 均包含在内。 **这部分时间将作为最终评分的主要依据**。
>
> 正确性评测和性能评测都会在多组不同的输入 shape 下进行。最终的性能分数是各组 shape 的加权平均。
>
> FlashQLA 将作为本实验的主要性能基线，同时也会给出 FLA 和 FlashInfer 在相同 case 下的结果作为 参考。性能分根据多个 case 上的综合表现计算，而不是由某一个 case 的最快结果决定。未通过正确性 检查、运行出错或超时的 case 不计性能分

## 理论学习与拆解

### 理论推导

Prefill是指在Auto Regression Model Reasoning中，一次性处理prompt中的所有token，并生成各层后续decode所需状态的过程.

Gated DeltaNet这里的数学推导着实复杂，我研究了几个小时，把实验文档没写的细节也补充进来：



> 对于逐token的更新，GDN的状态转移过程如下：
>
> 记GDN的每个head在 $t$ 时刻维护的状态矩阵是$S_t \in \mathbb{R}^{d_k\times d_v}$，参数$\alpha_t$表示旧状态保留度，$\beta_t$表示当前key的关联修改度，当前的value矩阵是$v_t$，预期状态矩阵是$\overline{S_t} = \alpha_tS_{t-1}$.
>
> 而实际的更新公式：
>
> $$
> S_t = \overline{S_t} + \beta_tk_t^T(v_t-k_t\overline{S_t})，\text{其中}\beta_t \in (0,1)
> $$
>
> 带入$\overline{S_t}$后展开，得到在逐token的情形下更新法则如下：
>
> $$
> S_t = \alpha_t (I-\beta_tk_tk_t^T)S_{t-1} + \beta_tk_tv_t^T \quad \alpha_t = \exp(\text{raw}\_g_t)
> $$
> 记此时输出为$o_t$，得到：
> $$
> o_t = \dfrac1{\sqrt{d_k}}q_t^TS_t
> $$

在后面的推导中需要经常利用Tensor维度来做一些小trick，所以先列一下在Python代码里的参数对应接口形状：

$B$表示`batch_size`，$T$表示`num_tokens`，

| 参数                                    | 形状                 | 类型 |
| --------------------------------------- | -------------------- | ---- |
| $\alpha_t, \beta_t , g^{\text{cumsum}}$ | [B, T, $H_v$]        | FP32 |
| $q_t, k_t$                              | [B, T, $H_q$, 128]   | BF16 |
| $v_t$                                   | [B, T, $H_v$, 128]   | BF16 |
| $A$                                     | [B, T, $H_v$, 64]    | BF16 |
| $S_j$                                   | [B, $H_v$, 128, 128] | FP32 |

> 把括号展开：$S_t=\alpha_tS_{t-1}+k_t\underbrace{\big[\beta_t v_t^\top-\alpha_t\beta_t k_t^\top S_{t-1}\big]}_{=:w_t^\top}$，令 $w_t=\beta_t\big(v_t-\alpha_t k_t^\top S_{t-1}\big)$，即
>
> $$
> S_t=\alpha_tS_{t-1}+k_t w_t^T
> $$
>
> $w_t$ 就是代码里的 `v_new`.
>
> 上面的过程中有几步乘法交换，因为$\alpha_t, \beta_t$都是相对低维度的，所以能够成立.
>
> 至此不难看出，整个方程计算上的效率问题在于， $w_t$ 依赖 $S_{t-1}$，而 $S_{t-1}$ 又依赖 $w_1,\dots,w_{t-1}$，形成了一条长度为 $T$ 的依赖链，必须串行.  引入chunk-wise 来优化的目标是把这条依赖链拆成 $\lceil T/C\rceil$ 个**跨 chunk 串行**（只传一个 $d_k\times d_v$ 的 state）、**chunk 内完全并行**（矩阵乘法）的步骤. 
>
> 同时，chunk 内长度 $\ell\le C=64$ 的串行依赖部分被转化成一个 $\ell\times\ell$ 的三角形线性方程组求解，于是产生表格里的矩阵 $A$ .
>
> 
>
> 设 chunk 内局部时间 $r=1,\dots,\ell$，$S_0=S_{[c]}$（上一 chunk 传入的 state）.
>
> 定义**局部累计衰减参数** $\gamma_r=\exp(g^{\text{cumsum}}_r)=\prod\limits_{j\le r}\alpha_j$（$g^{\text{cumsum}}=\log\gamma_r$，每个 chunk 下标从 0 重新开始），于是原方程可以转换成：
>
> $$
> S_r=\gamma_r S_0+\sum_{1 \le j\le r}\frac{\gamma_r}{\gamma_j}k_j w_j^T \tag{1}
> $$
>
> > 证明：此处我们用第一数学归纳法就能证明. 
> >
> > 首先可以看出，$\gamma_i$的下标从1开始.
> >
> > 1. 当$r = 1$时，$\gamma_1 = r_1$，$\sum\limits_{1 \leq j \leq 1} \dfrac{\gamma_1}{\gamma_j}k_jw_j^T = k_1w_1^T =  k_1\beta_1(v_1-\alpha_1k_1^TS_0)$，这是显然成立的；
> >
> > 2. 设$r\leq k$时都成立，则：
> >
> >     $$
> >     S_k = \gamma_kS_0 + \sum\limits_{1 \leq j \leq k}\dfrac{\gamma_k}{\gamma_j}k_jw_j^T
> >     $$
> >
> >     从而考虑$S_{t+1}$：
> >
> >     $$
> >     \begin{align*}
> >     S_{t+1} 
> >     &= \alpha_{t+1}S_t+k_{t+1} w_{t+1}^T \\
> >     &= \alpha_{t+1}(\gamma_kS_0 + \sum\limits_{1 \leq j \leq k}\dfrac{\gamma_k}{\gamma_j}k_jw_j^T) + k_{t+1} w_{t+1}^T \\
> >     &= (\alpha_{t+1}\gamma_k)S_0 +  \sum\limits_{1 \leq j \leq k}\dfrac{\alpha_{t+1}\gamma_k}{\gamma_j}k_jw_j^T + k_{t+1} w_{t+1}^T \\
> >     &= \gamma_{k+1}S_0 + \sum\limits_{1 \leq j \leq k+1}\dfrac{\gamma_{k+1}}{\gamma_j}k_jw_j^T
> >     \end{align*}
> >     $$
> >
> >     从而对$r = k+1$也成立，证毕.
>
> 代入 $w_r=\beta_r(v_r-\alpha_r k_r^T S_{r-1})$，把 $S_{r-1}$ 用$(1)$展开，得到
>
> $$
> w_r=\textcolor{red}{\beta_r v_r}-\textcolor{blue}{\beta_r\gamma_r k_r^T S_0}-\textcolor{green}{\beta_r\gamma_r\sum_{j \leq r-1}\dfrac{1}{\gamma_j}(k_r^T\cdot k_j)w_j^T} \tag{2}
> $$
>
> **利用行堆叠把$(2)$写成矩阵形式**：
>$$
> \underbrace{\Big[I+\mathrm{StrictLower}\big(B\Gamma KK^T\Gamma^{-1}\big)\Big]}_{M}\hat W = BV-B\Gamma K S_0 \tag{3}
> $$
> 
>> 行堆叠：顾名思义，把每个张量看成“行”来排列，叠成一个更高维度的张量，这样就能并行计算了. 我们先假设$\beta_r$堆叠出的4维张量是$B = \operatorname{diag}(\beta_1, \cdots, \beta_m)$，$\gamma_r$堆叠出的张量是$\Gamma = \operatorname{diag}(\gamma_1, \cdots, \gamma_m)$，$w_j$按行堆叠形成$\hat{W}$.
> >
> > 在$(2)$这个式子中，红色部分$\textcolor{red}{\beta_r v_r}$和蓝色部分$\textcolor{blue}{\beta_r\gamma_r k_r^\top S_0}$显然是可以直接堆叠的，行与行之间独立；然而绿色部分$\textcolor{green}{\beta_r\gamma_r\sum\limits_{j \leq r-1}\dfrac{1}{\gamma_j}(k_r^T\cdot k_j)w_j^T}$比较麻烦，因为涉及到各个行的累加.
> >
> > 考虑矩阵$G = K\cdot K^T$，则$k_r^T\cdot k_j$是$G$的$(r,j)$项；$\dfrac1{\gamma_j}$是$\Gamma^{-1}$的第 $j$ 列，起到缩放作用. 于是我们可以看出，求和后堆叠得到的结果是：
> >
> > $$
> > B\Gamma \cdot \text{StrictLower}(KK^T\Gamma^{-1}) \cdot \hat{W}
> > $$
> >
> > $\text{StrictLower}$表示下三角矩阵. 由于下三角矩阵右乘其他矩阵仍然得到下三角矩阵，所以：
> >
> > $$
> > B\Gamma \cdot \text{StrictLower}(KK^T\Gamma^{-1})\cdot \hat{W} = \text{StrictLower}(B\Gamma KK^T\Gamma^{-1})\cdot \hat{W}
> > $$
> >
> > 所以$(2)$可以在堆叠后化成：
> >
> > $$
> > \hat{W} = BV - B\Gamma KS_0 - B\Gamma\text{StrictLower}(KK^T\Gamma^{-1})\hat{W}
> > $$
> >
> > 移项即得$(3)$.
> 
>$M$ 是对角线恒为 1 的下三角矩阵，所以一定可逆，于是令 $A=M^{-1}$，代入化简得：
> 
>$$
> \hat W = A(BV) - A(B\Gamma K)S_0 = \underbrace{ABV}_{U} - \underbrace{AB\Gamma K}_{W}\cdot S_{[c]} = U-WS_{[c]}
> $$
> 
>现在考虑跨chunk的递推，在 ansatz 里取 $r=\ell$（chunk 末尾），所以有：
> 
>$$
> S_{[c+1]}=S_\ell=\gamma_\ell S_{[c]}+\gamma_\ell K^\top\Gamma^{-1}(U-WS_{[c]})
> $$
> 
>这个就是最终结果.

再看看代码里面的变量对应关系：（显然，这是codex总结的）

> | 数学符号                                                     | 含义                 | 代码变量                                                     | 出现位置                                                     |
> | ------------------------------------------------------------ | -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
> | $\gamma_r=\exp(g^{\text{cumsum}}_r)$                         | chunk 内累计衰减     | `gc` / `exp(gc)` / `exp(gc_i-gc_j)`                          | `g_cumsum` 输入，逐 chunk 清零                               |
> | $\beta_r$                                                    | delta rule 写入强度  | `bc`（即 `beta`）                                            | 输入                                                         |
> | $M=I+\mathrm{StrictLower}(B\Gamma KK^\top\Gamma^{-1})$       | 分块 KKT 矩阵        | `matrix`（`torch_kkt_solve` 内部）                           | —                                                            |
> | $A=M^{-1}$                                                   | KKT 逆矩阵（下三角） | `Ac` / 接口里的 `A`                                          | `torch_kkt_solve` 产出，**在核心计时区间之外算好、作为输入传入** |
> | $S_{[c]}$                                                    | chunk 起始 state     | `state`（循环变量）                                          | 初值来自 `initial_state` 或全零                              |
> | $U=ABV$                                                      |                      | `u`                                                          | `einsum("bhij,bhjv->bhiv", Ac, vc*bc)`                       |
> | $W=AB\Gamma K$                                               |                      | `w`                                                          | `einsum("bhij,bhjd->bhid", Ac, kc*bc*exp(gc))`               |
> | $\hat W=U-WS_{[c]}$（每 token 的 delta 写入量）              |                      | `v_new`                                                      | `u - w@state`                                                |
> | $S_{[c+1]}=\gamma_\ell S_{[c]}+\gamma_\ell K^\top\Gamma^{-1}\hat W$ | 跨 chunk 递推        | `state = exp(g_last)*state + einsum(kc*exp(g_last-gc), v_new)` | 用差值 exp 隐式实现 $\Gamma^{-1}$，避免显式构造              |
> | $O_{[c]}=\tfrac1{\sqrt{d_k}}[\Gamma QS_{[c]}+\Gamma\mathrm{Lower}(QK^\top)\Gamma^{-1}\hat W]$ | chunk 输出           | `output_from_state + output_in_chunk`                        | 同上，`decay=tril(exp(gc_i-gc_j))` 一步实现 $\Gamma,(\cdot),\Gamma^{-1}$ 且带因果 mask |
>
> 需要注意的是$\Gamma$ 或 $\Gamma^{-1}$这两个对角阵实际实现不需要显式构造，而是始终用 `exp(gc_i - gc_j)` 来代替 $\gamma_i/\gamma_j$.
>
> 这样做的好处是指数的自变量恒 $\le 0$（因为衰减是单调递减的），避免了单独算 $1/\gamma_j$ 时可能出现的数值爆炸.



拉取文件夹：

```bash
npx degit ZJUSCT/HPC101/src/lab3 lab3
```

可以看到已经封装好了`chunk_local_cumsum`和`kkt_solve`函数，因此`run.py`中只需要导入，不计入时间：

```python
from preprocessing.tilelang_cumsum import chunk_local_cumsum
from preprocessing.tilelang_kkt_solve import kkt_solve
```

参考代码实现的过程：

- `torch_chunk_local_cumsum`：实现 $g^{\text{cumsum}}_r=\sum\limits_{j=1}^r\text{raw}\_g_j$，每 64 个 token 重新从 0 累加——对应"chunk 内的门控前缀和".
- `torch_kkt_solve`：先做 `_expand_qk_heads`（GVA 展开：把 $H_q$ 个 key head 沿 head 维 `repeat_interleave` 成 $H_v$ 份，等价于"每个 query/key head 连续对应 $G=H_v/H_q$ 个 value head"），再构造 $M$、用 `torch.linalg.solve_triangular`解出 $A$ (接口表里预先算好、直接作为 `A` 传入的部分).
- **`torch_gdn_prefill_forward`**：按 chunk 顺序循环，chunk 内用矩阵乘法并行算出 $W,U,\hat W$（`v_new`）、输出 $O$，再把 state 递推到下一个 chunk. 这部分可以算出$U,W,S,O$.

## 优化过程

首先跑一下没有优化的版本，是完全基于上面的数学推导得出的：

```bash
hpc submit -p lab3 "python run.py"
```

结果：

<center><img src="./figures/lab3/0.png" alt="overview" style="zoom:100%;" /></center>

可以看到跟开源的实现差距还是挺大的，比如跑了一下benckmark甚至超时了：

````bash
hpc submit -p lab3 "python run.py --reference-benchmarks"
````

结果：

<center><img src="./figures/lab3/1.png" alt="overview" style="zoom:100%;" /></center>

可以看出即使与其中最慢的FLA相比，也慢了不少. 此时跟评分表格相比差距极大，预测只能拿

现在先定位一下hotspot找找方向，命名为`baseline`：

```bash
hpc submit -p lab3 -t 5m "nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --stats=true --force-overwrite=true -o baseline python run.py --case parallel_gva"
```

得到结果后在本地查看：

```bash
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --report cuda_gpu_mem_time_sum --force-export=true --format table baseline.nsys-rep
```

<center><img src="./figures/lab3/2.png" alt="overview" style="zoom:100%;" /></center>

<center><img src="./figures/lab3/3.png" alt="overview" style="zoom:100%;" /></center>

在GUI中也查看一下：

<center><img src="./figures/lab3/4.png" alt="overview" style="zoom:100%;" /></center>

可以得到几个信息：

* CUDA这一行几乎是空的，只有几个很短的蓝色小竖条(kernel 执行)，这说明在 GPU 上跑的计算时间极短；
* 灰色方块是 CPU busy 区间,呈现出"忙一段、停一段"的间歇模式；
* OS runtime libraries 行里密集的 `read`大量反复出现，贯穿整个 timeline，表明每次调用都在做 cache 查找或重复 IO，而不是命中缓存直接跑，所以效率很低下.



三个 TileLang kernel 在上面都会打包显示为 kernel_kernel，合计占 GPU kernel 时间 73.6%，具体来看的话，有`chunk_state`, `chunk_ouput`和`W/U`三种kernel：

``````bash
nsys stats --report cuda_gpu_kern_gb_sum --force-export=true --format table baseline.nsys-rep
``````

结果提取出来是这样的：

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | GridXYZ | BlockXYZ | Name |
| -------- | --------------- | --------- | -------- | -------- | -------- | -------- | ----------- | ------- | -------- | ---- |
| 41.5 | 131,076,074 | 111  | 1,180,865.5 | 1,186,086.0 | 1,156,773 | 1,204,869 | 13,566.5 | 2   16    1 | 128    1    1 | kernel_kernel |
| 23.0 | 72,800,456 | 111  | 655,860.0 | 656,163.0 | 641,539 | 670,723 | 7,519.9 | 2  512    1 | 128    1    1 | kernel_kernel |
| 9.1  | 28,699,232 | 112  | 256,243.1 | 257,073.0 | 89,473 | 265,825 | 16,135.7 | 512    1    1 | 128    1    1 | kernel_kernel |

从代码给出的形状上看，依次就是`chunk_state`, `chunk_ouput`和`W/U`三种kernel：

```python
with T.Kernel(total_chunks * H, threads=128) as (block,): # W/U
...
with T.Kernel(
            HEAD_DIM_V // BLOCK_DIM,
            batch_size * H,
            threads=128,
        ) as (bv, bbh):  # chunk_state
...
with T.Kernel(
            HEAD_DIM_V // BLOCK_DIM,
            total_chunks * H,
            threads=128,
        ) as (bv, block): # chunk_output
```

所以接下来着重优化`chunk_state`这个kernel，然后再考虑`chunk_output`和`W/U`这两个kernel.

这里有个问题是，一开始直接使用ncu会导致超时，不得不先尝试nsys profile. 唉，这个分区的最大运行时长不能调大一点吗？

（使用`--clock-control none`是为了防止ncu在MIG情况下锁掉时钟分区的报错）

```bash
hpc submit -p lab3 "ncu --clock-control none -o baseline python run.py"
```

然而还是跑不完，刚要开始分析kernel信息的时候就停止了，哎.

### 等价数学变换

肉眼可见的优化是把W/U的计算合并起来，即：

$$
U = A^{-1}BV, W = AB\Gamma K, \hat{W} = U - WS_0 \Longrightarrow \hat W = AB(V - \Gamma KS_0) 
$$

这样矩阵乘法次数从6次降低到4次，目测可以提升很多效率.

#### Profiling变化

观察一下修改之后的用时，可以发现比baseline确实快了一些，并且超过评分标准的60分线了：

<center><img src="./figures/lab3/5.png" alt="overview" style="zoom:100%;" /></center>

hotspot:

<center><img src="./figures/lab3/6.png" alt="overview" style="zoom:100%;" /></center>

看起来跟前面的没什么区别.

### Kernel Fusion

上面的过程其实也相当于做了第一步kernel fusion，把W/U放到了final_state的残差计算过程中，先从三个核变成`_tilelang_residual_state`和`_tilelang_chunk_output`两个核.

继续观察公式可以发现，$O_{[c]}$和$S_{[c+1]}$使用的是同一个$\hat W$，而且output必须读取更新前的$S_{[c]}$. 所以只要在覆盖state之前先算完$QS_{[c]}$，就可以在同一个persistent CTA里按照：

$$
QS_{[c]}\longrightarrow\hat W\longrightarrow O_{[c]}\longrightarrow S_{[c+1]}
$$

的顺序完成全部计算. 最后代码只剩下`_tilelang_fused_forward`一个核，`hidden`和`v_new`也不需要再写回HBM.





### Shared Memory

完成这个fusion之后，$S_{[c]}$和$\hat W$已经不用在HBM里来回写了，但是还有一个比较明显的问题：`BLOCK_DIM=64`时，同一个value head要用两个CTA处理. 这两个CTA虽然负责不同的value列，却会各自完整读取一次$Q,K,A,\Gamma,B$，还会把同一个$QK^T$算两遍.

所以这里的思路很直接：把`BLOCK_DIM`从64扩大到128，让一个CTA直接负责完整的$d_v=128$，线程数相应从128增加到256. 这样$Q,K,A$和gate每个chunk只需要从global memory加载一次，然后放在shared memory中给后面的多个GEMM反复使用：

```text
Q_shared ──┬─> Q S_[c]
           └─> Q K^T

K_shared ──┬─> K S_[c]
           ├─> Q K^T
           └─> K^T(γ_l Γ^-1 W_hat)
```

这里shared memory的作用不是单纯地“做缓存”，而是明确让同一个CTA拥有整个chunk需要的数据. 数据从HBM读进来一次后，后面的$QS_{[c]}$、$KS_{[c]}$、$QK^T$和state update都直接读取shared memory，不再重新发global load.

按照$C=64,d_k=d_v=128$计算，原来两个64-wide CTA读取$Q,K,A,V,g,\beta$大约需要：

$$
32+32+16+16+0.5+0.5=97\text{ KiB/chunk/head}.
$$

合成一个128-wide CTA后变为：

$$
16+16+8+16+0.25+0.25=56.5\text{ KiB/chunk/head},
$$

所以主要输入的逻辑global read降低了大约：

$$
\frac{97-56.5}{97}\approx41.8\%.
$$

当然这个数是代码层面的逻辑读取量，不一定等于真正的DRAM流量，因为原来的第二次读取有可能命中L2；不过即使命中L2，重复的load指令、L2 transaction和global-to-shared copy仍然存在.

shared memory本身也不能无脑分配，否则会压低occupancy. 这里利用不同中间量的生命周期不会重叠，复用两块buffer：

* `residual_shared`依次保存$V\rightarrow B(V-\Gamma KS_{[c]})\rightarrow\hat W\rightarrow\gamma_\ell\Gamma^{-1}\hat W\rightarrow O_{[c]}$的写回暂存；
* `a_shared`先保存$A$，在$A\cdot residual$完成后再覆盖成$\Gamma\mathrm{Lower}(QK^T)\Gamma^{-1}$.

这样不需要额外的`value_shared`和`output_shared`，整个CTA的shared memory大约是：

| Buffer | 大小 |
| ------ | ----: |
| `state_shared[128,128]` | 32 KiB |
| `q_shared[64,128]` | 16 KiB |
| `k_shared[64,128]` | 16 KiB |
| `residual_shared[64,128]` | 16 KiB |
| `a_shared[64,64]` | 8 KiB |
| gate、beta和decay | 1 KiB |
| 合计 | 约89 KiB |

另外`g_cumsum`也只从global memory读取一次，先写进`gate_shared`，$\gamma_r=\exp(g_r^{\text{cumsum}})$以及后续的gate ratio都从这份shared copy计算.

修改后所有case都能通过，`parallel_gva`在`warmup=10,repetitions=100`下从fused-64的1.467 ms降低到fused-128的1.036 ms，大约提升了29.4%. 这个提升不全来自内存，因为$QK^T$也从两次变成了一次；但是从数据流上看，减少重复global read确实是这一步优化的核心.


### ping-pong buffer / multi-buffering

这里没有直接加完整的double buffer. 当前一个CTA已经用了大约89 KiB shared memory，如果把下一chunk的$K,V,A$全部再缓存一份，还要增加约40 KiB. 而且chunk之间有

$$
S_{[c]}\longrightarrow S_{[c+1]}
$$

这个真正的数据依赖，下一chunk的GEMM不能提前执行. 从现有profile估算的逻辑带宽也只有约28 GB/s，所以现在的hotspot更像是GEMM和寄存器/warp资源，而不是HBM带宽. 因此这里先不增加pipeline stage，避免shared memory继续膨胀.


### warp specialization

fusion之后每个CTA同时有几个比较大的FP32 fragment：`state_fragment[128,128]`、`value_fragment[64,128]`、`output_fragment[64,128]`和`scores_fragment[64,64]`. 在256 threads下，粗略平均到每个线程分别是64、32、32和16个FP32值，还没有算编译器临时变量，所以寄存器压力比较大.

先把$QS_{[c]}$向后移动，在score写进shared memory以后再计算：

$$
KS_{[c]}\rightarrow\hat W\rightarrow QK^T\rightarrow QS_{[c]}\rightarrow O_{[c]}\rightarrow S_{[c+1]}.
$$

这不改变

$$
O_{[c]}=\mathrm{scale}\left(\Gamma QS_{[c]}+\Gamma\mathrm{Lower}(QK^T)\Gamma^{-1}\hat W\right),
$$

因为$QS_{[c]}$仍然在state update以前计算，只是让`scores_fragment`和`output_fragment`的生命周期不再重叠. 这个版本正确，但是`parallel_gva`仍为1.036 ms，说明只改变代码顺序还不够.

接着把一个CTA的线程数从256增加到512. 这里没有手写producer/consumer warp group，所以严格来说不是完整的warp-specialized pipeline；它的直接作用是把fragment分摊给更多线程，并把单CTA从8 warps增加到16 warps. shared memory仍然约89 KiB，没有增加新的buffer.

所有case均通过. `warmup=1,repetitions=2`的结果如下：

| Case | 256 threads (ms) | 512 threads (ms) | 延迟下降 |
|---|---:|---:|---:|
| `short_tail_state` | 0.301 | 0.252 | 16.3% |
| `chain_equal` | 1.992 | 1.577 | 20.8% |
| `parallel_equal` | 1.043 | 0.836 | 19.8% |
| `parallel_gva` | 1.039 | 0.831 | 20.0% |
| `long_low_gva` | 8.049 | 6.393 | 20.6% |
| `batch_split_gva` | 6.199 | 4.885 | 21.2% |
| `wide_gva_state` | 10.615 | 8.426 | 20.6% |
| `deep_gva_state` | 12.560 | 9.909 | 21.1% |

`parallel_gva`在`warmup=10,repetitions=100`下复测两次为0.815 ms和0.822 ms，相比256-thread版本的1.036 ms大约降低20.9%，即约1.26倍加速.

另外逐个试了几种组合：1024 threads为0.925 ms，额外warp的同步开销反而更大；384 threads因为fragment布局不满足要求而编译失败；BF16 $\hat W$回载、`gate_last` shared复用和warp-specialization pass都没有稳定收益. 所以最后保留512 threads，不保留这些组合.


