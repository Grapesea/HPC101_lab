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

这张图只能用于区分程序阶段，不能直接当成student kernel的GPU利用率：

* CUDA行在整个时间轴上看起来很短，是因为横轴还包含reference correctness、TileLang JIT、warm-up和CPU等待；
* 灰色CPU busy区间主要对应编译、Python/PyTorch调度和profiler工作，不属于GPU event计时的student core；
* OS runtime中的密集`read`还包含动态库、JIT cache和profiler自身的文件访问，不能据此判断每次student forward都在重复IO.

因此后面必须用`cudaProfilerStart/Stop`只截取warm-up后的正式迭代，再计算kernel busy ratio. 全程图里的“CUDA面积占比”只反映端到端程序组成，不是评分区间内的GPU利用率.



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
U = ABV, W = AB\Gamma K, \hat{W} = U - WS_0 \Longrightarrow \hat W = AB(V - \Gamma KS_0)
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


### CPU等待和warm-up

前面的优化主要针对GPU kernel，接下来再看一下CPU等待. 这几项不会减少公式中的FLOPs，也不会改变$O_{[c]}$和$S_{[c+1]}$，主要影响host怎么等待GPU以及profile是否包含初始化.

第一项是在CUDA context创建以前调用：

```python
cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)
```

这里不能用`CUDA_LAUNCH_BLOCKING=1`代替，因为后者会让每次kernel launch都同步，语义不一样. 提交命令为：

```bash
hpc submit -p lab3 -t 5m -e NAME=blocking "3.sh"
```

第二项用`cudaProfilerStart/Stop`把正确性检查、JIT和warm-up放在capture range外面，只profile正式的100次迭代：

```bash
hpc submit -p lab3 -t 5m -e NAME=warmup "3.sh"
```

第三项先排队100组CUDA event和kernel，最后只同步一次，同时使用blocking sync：

```bash
hpc submit -p lab3 -t 5m -e NAME=polling "3.sh"
```

结果如下，其中kernel median来自nsys，脚本median来自`run.py`：

| 方法 | 脚本median | kernel median | `cudaEventSynchronize` | `cudaDeviceSynchronize` | 结论 |
|---|---:|---:|---:|---:|---|
| 普通`cta` profile | 0.843 ms | 0.815 ms | 100 | 1 | 对照组 |
| `blocking` | 0.834 ms | 0.804 ms | 100 | 1 | CPU进入阻塞等待，GPU时间基本不变 |
| `warmup` | 0.833 ms | 0.803 ms | 100 | 0（capture内） | capture从113个主体调用清理成100个 |
| `polling` | 0.859 ms | 0.856 ms | 0 | 2 | 同步点减少，但稳态延迟约退化3% |

`blocking`的OS runtime里出现了`sem_wait`和约4 s的`pthread_cond_timedwait`，说明等待策略确实发生了变化. 但是nsys中的`poll`没有直接消失，因为这里还包含profiler和其他runtime后台线程的系统调用，不能把所有`poll`都当成CUDA spin.

所以最后的处理是：blocking和warm-up capture作为`3.sh`中的可选profiling模式保留，不修改默认评分路径；批量同步虽然减少CPU同步次数，但会改变benchmark的提交节奏并造成小幅退化，因此不设为默认.


### Hopper warp specialization

前面的512-thread版本只是增加线程数，几个阶段仍由同一批线程串行完成. 最后参考FlashQLA的Hopper实现，把一个CTA的16个warp分成四组：

* 0--127线程维护$S_{[c]}$；
* 128--255线程计算$\hat W$；
* 256--383线程计算$O_{[c]}$；
* 384--511线程负责TMA加载和写回.

$Q,K,V,A,g,\beta$使用两组shared buffer做ping-pong. producer加载第$c+1$个chunk时，其余warp group可以继续计算第$c$个chunk；named barrier只在真正的数据依赖处同步. 数学过程仍然是：

$$
\begin{aligned}
R_{[c]}&=V_{[c]}-\Gamma_{[c]}K_{[c]}S_{[c]},\\
\hat W_{[c]}&=A_{[c]}B_{[c]}R_{[c]},\\
O_{[c]}&=\mathrm{scale}\left(\Gamma_{[c]}Q_{[c]}S_{[c]}
+\Gamma_{[c]}\mathrm{Lower}(Q_{[c]}K_{[c]}^T)\Gamma_{[c]}^{-1}\hat W_{[c]}\right),\\
S_{[c+1]}&=\gamma_\ell S_{[c]}
+K_{[c]}^T\left(\gamma_\ell\Gamma_{[c]}^{-1}\hat W_{[c]}\right).
\end{aligned}
$$

这里的重点不是减少矩阵乘法次数，而是让加载、state、value和output四条流水线重叠. 初版在`warmup=10,repetitions=100`下已经把`wide_gva_state`从8.182 ms降到2.488 ms，但是仍略慢于100分线2.426656 ms.


### 消除$A$的中间转换

初版warp-specialized kernel把$\beta$吸收到$A$的列中：

$$
\hat W_{[c]}
=\left(A_{[c]}\odot\beta_{\mathrm{col}}\right)
\left(V_{[c]}-\Gamma_{[c]}K_{[c]}S_{[c]}\right).
$$

因为$B_{[c]}=\mathrm{diag}(\beta)$，所以它和下面的写法完全等价：

$$
\left(A_{[c]}\odot\beta_{\mathrm{col}}\right)R_{[c]}
=A_{[c]}B_{[c]}R_{[c]}.
$$

但是前一种写法需要output warp group先把$A$从BF16 shared memory读到一个$64\times64$的FP32 fragment，逐列乘$\beta$，再写回shared memory，value warp group才能继续GEMM. 这会增加4096个FP32 fragment元素、一次shared memory往返和一个跨warp barrier.

因此最后恢复成原公式的计算顺序：先逐行计算$B_{[c]}R_{[c]}$，再直接使用预处理给出的$A_{[c]}$做GEMM：

$$
R_{[c]}\longrightarrow B_{[c]}R_{[c]}
\longrightarrow A_{[c]}B_{[c]}R_{[c]}=\hat W_{[c]}.
$$

这样只增加$64\times128$次逐元素乘法，但删掉了更昂贵的$A$转换、FP32 fragment和同步. `wide_gva_state`由2.488 ms降到约2.17 ms，`deep_gva_state`由2.820 ms降到约2.45 ms.

最后使用`warmup=10,repetitions=100`完整复测两轮：

| Case | 100分线 (ms) | 第1轮 (ms) | 第2轮 (ms) |
|---|---:|---:|---:|
| `short_tail_state` | 0.346048 | 0.104 | 0.103 |
| `chain_equal` | 0.497792 | 0.387 | 0.382 |
| `parallel_equal` | 0.511264 | 0.271 | 0.271 |
| `parallel_gva` | 0.491552 | 0.230 | 0.228 |
| `long_low_gva` | 1.858816 | 1.435 | 1.432 |
| `batch_split_gva` | 1.532128 | 1.196 | 1.199 |
| `wide_gva_state` | 2.426656 | 2.174 | 2.179 |
| `deep_gva_state` | 2.831104 | 2.445 | 2.453 |

两轮都是8/8正确且全部低于100分检查点. 最接近检查点的是`wide_gva_state`，仍有约10.2%的时间余量；因此这里停止继续改公共路径，避免为了很小的额外收益破坏hidden case的稳定性.


## 热点演化与全部尝试汇总

这里把前面的优化按“profile看到的热点--提出假设--只修改一个变量--决定保留或回退”的顺序重新汇总. 所有性能数字都先检查正确性；小改动主要使用`parallel_gva`比较，结构性改动再跑全部8个case.

另外，`g^{\mathrm{cumsum}}`和$A$并没有在student kernel中重新计算，而是直接使用`preprocessing/`中`chunk_local_cumsum`和`kkt_solve`的输出. 这两部分位于核心计时范围外，所以优化重点始终是已经预处理好的$Q,K,V,g^{\mathrm{cumsum}},\beta,A$如何完成$O$和final state.


### 第一阶段：多kernel和HBM中间量

baseline中三个TileLang kernel合计占GPU kernel时间73.6%，其中：

| Kernel | GPU kernel时间占比 | 主要问题 |
|---|---:|---|
| `chunk_state` | 41.5% | 顺序遍历chunk，并反复读写state和$\hat W$ |
| `chunk_output` | 23.0% | 重新读取state、$\hat W$和$Q,K$ |
| `W/U` | 9.1% | 单独产生$W,U$等全局中间量 |

因此最初的热点不是某一条逐元素指令，而是三个kernel之间物化中间张量造成的launch、HBM读写和重复GEMM.

第一步尝试数学合并：

$$
U=ABV,\quad W=AB\Gamma K,\quad \hat W=U-WS_{[c]}
$$

化为：

$$
\hat W=A B\left(V-\Gamma KS_{[c]}\right).
$$

这一步把相关矩阵乘法由6次减少到4次，并把三个kernel先合并成两个. 接着利用$O_{[c]}$和$S_{[c+1]}$共用$\hat W$，实现单个persistent kernel：

$$
QS_{[c]}\rightarrow\hat W\rightarrow O_{[c]}\rightarrow S_{[c+1]}.
$$

`hidden`和`v_new`不再写入HBM，修改被保留. 这一版全部case正确，但`parallel_gva`约1.459 ms，说明消除kernel边界以后，热点已经转移到单个CTA内部.


### 第二阶段：重复global read和重复$QK^T$

融合版最初使用`BLOCK_DIM=64`，一个value head需要两个CTA. 两个CTA各自完整读取$Q,K,A,g,\beta$，并各算一次相同的$QK^T$. 因此尝试把`BLOCK_DIM`改成128，让一个CTA覆盖完整$d_v=128$，同时用shared memory缓存输入.

按每个chunk/value head估算，主要输入的逻辑global read从97 KiB降到56.5 KiB，减少约41.8%；$QK^T$也从每个value head两次降到一次. `parallel_gva`从约1.459 ms降到1.050 ms，之后稳定在约1.036 ms，所以这个修改被保留.

这里还尝试过跨GVA value head共享raw $QK^T$. 因为同一个Q/K head会对应多个value head，理论上可以先生成：

$$
P^{\mathrm{raw}}_{[c]}=Q_{[c]}K_{[c]}^T,
$$

再由各value head施加自己的gate：

$$
P^{(h)}_{ij}=P^{\mathrm{raw}}_{ij}\exp(g^{(h)}_i-g^{(h)}_j).
$$

该版本正确，但必须增加raw score workspace的global写回、读取和调度，`parallel_gva`为0.926 ms，比当时约0.82 ms的512-thread baseline慢约12%，所以回退. 结论是：跨CTA减少一次GEMM带来的收益，小于物化$64\times64$ score和额外kernel的成本；score只能在同一CTA内部复用.


### 第三阶段：寄存器压力和CTA线程数

`BLOCK_DIM=128`以后，主要fragment包括：

$$
S:128\times128,\quad \hat W:64\times128,\quad O:64\times128,
\quad P:64\times64.
$$

256线程时每个线程持有较多FP32 accumulator，热点由global read转成寄存器压力、GEMM之间的串行依赖和有限warp数. 围绕这个热点做了以下尝试：

| 尝试 | 结果 | 处理 |
|---|---|---|
| 调整计算顺序，让score fragment和output fragment生命周期不重叠 | 正确，但`parallel_gva`仍约1.036 ms | 思路保留，但单独没有可测收益 |
| 256线程改成512线程 | `parallel_gva`约1.036 ms降到0.815--0.822 ms，全部case约快20% | 保留 |
| 512线程改成1024线程 | 约0.925 ms，warp和同步开销增加 | 回退 |
| 改成384线程 | fragment布局不满足TileLang lowering要求，编译失败 | 回退 |
| 开启TileLang warp-specialization pass | 没有稳定收益 | 回退 |
| 把$\hat W$回载为BF16以减少FP32 fragment压力 | shared往返和舍入抵消收益，没有稳定提升 | 回退 |
| 把`gate_last`放进shared memory复用 | 只减少每chunk一个标量读取，没有稳定提升 | 回退 |

512-thread persistent kernel的两次完整复测基本一致：

| Case | 第1轮 (ms) | 第2轮 (ms) |
|---|---:|---:|
| `short_tail_state` | 0.231 | 0.233 |
| `chain_equal` | 1.543 | 1.522 |
| `parallel_equal` | 0.820 | 0.826 |
| `parallel_gva` | 0.816 | 0.826 |
| `long_low_gva` | 6.152 | 6.228 |
| `batch_split_gva` | 4.713 | 4.720 |
| `wide_gva_state` | 8.182 | 8.177 |
| `deep_gva_state` | 9.635 | 9.617 |

此时只有`short_tail_state`超过100分检查点. 长序列耗时近似随chunk数增长，说明每个chunk内部仍有大量串行阶段，继续增加普通线程已经不能解决问题.


### 第四阶段：逐元素数学和因果区域微调

这一阶段针对`exp`、首chunk和因果mask做了几组单变量实验，希望减少非GEMM指令：

1. 把gate ratio从

   $$
   \exp(g_i-g_j)
   $$

   改成

   $$
   \gamma_i/\gamma_j,\qquad \gamma_i=\exp(g_i).
   $$

   直接版本约0.817 ms，与0.815--0.822 ms baseline相同；把$1/\gamma_j$再放进shared memory后约0.893 ms，除法、shared访问和同步反而更慢. 同时拆成商还会扩大overflow/underflow风险，因此全部回退，保留`exp(g_i-g_j)`.

2. 因果矩阵对角线恒有$\exp(g_i-g_i)=1$，尝试直接写1以跳过对角线`exp`. 对角线只有64个元素，而下三角共有$64\times65/2=2080$个元素，只减少约3.1%的因果`exp`，端到端没有稳定收益，回退.

3. 当没有initial state时，第一个chunk满足$S_{[0]}=0$，理论上可跳过$K S_{[0]}$和$Q S_{[0]}$两个GEMM. 两种调度版本测得约0.829 ms和0.811 ms，均落在baseline波动内. 原因是只省第一个chunk，长序列中很快被摊薄，而且分支会打断统一流水，因此回退.

4. 尝试把完整$64\times64$ causal score GEMM拆成$16\times16$下三角block. 理论上只计算16个block中的10个，可减少37.5%的raw score乘加；实际TileLang编译报错`The offset of the first dimension of A must be 0`，而且小GEMM也可能降低Tensor Core利用率，因此没有进入计时并回退.

这些实验说明，当前热点不是少量标量`exp`或第一个chunk的特殊情况，而是每个chunk都重复发生的大GEMM、数据搬运和阶段间等待.


### 第五阶段：真正的warp-specialized流水线

根据上面的热点，最终不再继续做局部删指令，而是把一个CTA拆成四个固定warp group，并给$Q,K,V,A,g,\beta$建立double buffer：

```text
TMA producer: load chunk c+1
state group : maintain S_[c] and S_[c+1]
value group : compute R_[c] and W_hat_[c]
output group: compute QK^T, QS_[c] and O_[c]
```

这样把“单CTA内所有阶段由同一批线程串行执行”的热点改成producer/consumer重叠. 初版正式10/100结果为：

| Case | 普通512线程 (ms) | warp-specialized (ms) |
|---|---:|---:|
| `short_tail_state` | 0.231 | 0.114 |
| `chain_equal` | 1.543 | 0.454 |
| `parallel_equal` | 0.820 | 0.292 |
| `parallel_gva` | 0.816 | 0.259 |
| `long_low_gva` | 6.152 | 1.691 |
| `batch_split_gva` | 4.713 | 1.335 |
| `wide_gva_state` | 8.182 | 2.488 |
| `deep_gva_state` | 9.635 | 2.820 |

这时7/8个case超过100分检查点，剩余热点集中在`wide_gva_state`：它有64个value head，重复执行每chunk的$A$转换和跨warp同步；`deep_gva_state`也因为chunk最多而对每chunk固定开销敏感.

因此最后消除：

$$
(A\odot\beta_{\mathrm{col}})R
$$

所需的FP32 `a_fragment`、shared回写和barrier，恢复成：

$$
A(BR).
$$

这一步直接针对每个chunk、每个value head都会发生的同步热点，最终让`wide_gva_state`稳定在2.17--2.18 ms，`deep_gva_state`稳定在2.45 ms左右，并得到前表中的两轮8/8 100分结果.


### Host侧尝试为什么没有进入最终评分路径

nsys的OS runtime曾显示大量`poll`、`read`和CPU busy区间，但隔离warm-up后，正式100次调用的核心仍是同一个GPU kernel. 因此又分别测试：

| 尝试 | 观测 | 决定 |
|---|---|---|
| `cudaDeviceScheduleBlockingSync` | 脚本median 0.834 ms，kernel median 0.804 ms；CPU出现`sem_wait`，GPU时间不变 | 仅保留为profile选项 |
| warm-up后再开始capture | 脚本median 0.833 ms，kernel median 0.803 ms；排除JIT和初始化 | 保留为测量方法，不改student路径 |
| 批量提交后统一同步，关闭逐次polling | 同步点减少，但median退化到约0.859 ms | 不采用 |

所以CPU等待是profile可读性和多进程CPU占用的热点，不是student GPU kernel得分的主要热点. 最终提交只保留改变GPU稳态执行时间的数学重排、fusion、shared-memory复用和Hopper warp-specialized pipeline.


### 最终保留与回退清单

| 优化手段 | 状态 | 原因 |
|---|---|---|
| 使用`preprocessing/`封装结果 | 保留 | 避免在计时kernel中重复cumsum和三角求解 |
| $\hat W=AB(V-\Gamma KS)$数学合并 | 保留 | 减少GEMM和全局中间量 |
| 单persistent kernel fusion | 保留 | 消除kernel边界和HBM往返 |
| `BLOCK_DIM=128`与shared-memory输入复用 | 保留 | 减少约41.8%逻辑global read，并避免重复$QK^T$ |
| 512线程 | 被Hopper四warp-group结构继承 | 分摊fragment并提供流水并行度 |
| double buffer、TMA和named barrier | 保留 | 重叠加载、state、value和output |
| $A(BR)$替代$(A\odot\beta_{\mathrm{col}})R$ | 保留 | 删除A转换、FP32 fragment和barrier |
| raw $QK^T$全局workspace | 回退 | global流量和额外调度大于GVA复用收益 |
| $\gamma_i/\gamma_j$及reciprocal shared | 回退 | 无收益或退化，并增加数值风险 |
| 首chunk跳过零state GEMM | 回退 | 只影响一次，长序列被摊薄 |
| $16\times16$ causal block | 回退 | TileLang lowering失败，小GEMM收益也不确定 |
| 1024/384线程、BF16回载、`gate_last` shared、对角exp跳过 | 回退 | 编译失败、退化或无稳定收益 |
| blocking、warm-up capture、关闭polling | 不进入默认student路径 | 只改变host等待或测量方式 |


## 继续优化：低grid下的自适应value tile

前面的kernel固定令`block_dv=128`，一个CTA负责一个完整value head. 这样可以最大程度复用$Q,K,A,g,\beta$，但是CTA数量只有：

$$
N_{\mathrm{CTA}}=B\times H_v.
$$

评分设备有14个SM. `chain_equal`中$B=1,H_v=4$，所以整个persistent kernel只有4个CTA，第一轮只能使用4个SM；这里的新热点不是单CTA内部的FLOPs，而是CTA grid太小导致的SM利用率不足.

如果把value维拆成宽度为$d_b$的tile，则：

$$
N_{\mathrm{CTA}}=B\times H_v\times\frac{128}{d_b}.
$$

但拆分也会让每个value head的$Q,K,A,g,\beta$和$QK^T$被重复$128/d_b$次，所以需要在“增加CTA并行度”和“重复计算/读取”之间实测平衡点.


### 三种`block_dv`实验

首先只在低grid case上比较：

| `block_dv` | `chain_equal` CTA数 | 延迟 | 结论 |
|---:|---:|---:|---|
| 128 | 4 | 约0.382 ms | CTA太少，SM没有铺满 |
| 64 | 8 | 0.293 ms | 比128快约23%，并行收益大于两次读取 |
| 32 | 16 | 0.452 ms | 四次重复$Q,K,A$和$QK^T$，反而回退 |

所以不能直接照搬“CTA数至少等于SM数”的规则. 对当前kernel，8个较大的CTA比16个小CTA更快，因为Tensor Core tile、TMA传输和shared-memory复用效率更高.

接着曾尝试对所有$B H_v<10$的case使用`block_dv=64`：

| Case | 固定128 (ms) | 低grid使用64 (ms) | 结果 |
|---|---:|---:|---|
| `short_tail_state` | 0.103 | 0.126 | 短序列中额外CTA启动和同步占比过高 |
| `chain_equal` | 0.382 | 0.294 | 提升明显 |
| `long_low_gva` | 1.432--1.470 | 2.138 | 长序列重复读取和GEMM被chunk数放大 |

因此最后采用更保守的调度：

```python
grid_size = batch_size * num_heads_v
block_dv = 64 if grid_size <= 4 else 128
```

这里`grid_size<=4`才认为CTA不足是第一热点；其余情况继续优先避免重复读取. 最终完整10/100复测两轮：

| Case | 第1轮 (ms) | 第2轮 (ms) |
|---|---:|---:|
| `short_tail_state` | 0.103 | 0.103 |
| `chain_equal` | 0.295 | 0.291 |
| `parallel_equal` | 0.275 | 0.273 |
| `parallel_gva` | 0.231 | 0.230 |
| `long_low_gva` | 1.470 | 1.464 |
| `batch_split_gva` | 1.215 | 1.206 |
| `wide_gva_state` | 2.176 | 2.174 |
| `deep_gva_state` | 2.457 | 2.466 |

两轮均为8/8正确. 只有`chain_equal`改变执行路径并获得约24%的稳定提升，其余case继续使用已经验证的128-wide kernel.


### 删除`value_decay_shared`的尝试

state update中还存在一块`value_decay_shared`：

$$
S_{[c+1]}=\gamma_\ell S_{[c]}+K^T(D\hat W),
\qquad D=\gamma_\ell\Gamma^{-1}.
$$

利用对角矩阵的结合关系，可以改写为：

$$
K^T(D\hat W)=(DK)^T\hat W.
$$

实验版在output warp group使用完K以后，直接在shared memory中原地计算$DK$，从而删除`value_decay_shared`和一次$D\hat W$写入. 该版本正确，但`wide_gva_state`从约2.18 ms退化到2.305 ms，最终回退.

热点原因是原版本在value warp group中计算$D\hat W$，能与output/state的其他阶段重叠；改写后$DK$落到state update的关键路径上，而且固定需要缩放$64\times128$个K元素. 这说明shared memory容量不是当前第一热点，不能为了少一块buffer破坏warp group之间的流水重叠.


### 当前热点和下一步方向

经过这一步以后，公共case可以分成三类：

| 类型 | 当前主要热点 | 后续可行方向 |
|---|---|---|
| 极低grid，如`chain_equal` | CTA级并行度 | 继续按$B H_v$和chunk数选择64/128 tile，但不再使用32 |
| 长序列，如`long_low_gva`、`deep_gva_state` | 每chunk固定GEMM和named barrier | 合并barrier或让下一chunk TMA更早发射，不能增加重复读取 |
| 多GVA head，如`wide_gva_state` | 不同value head重复读取同一Q/K | 研究cluster/TMA multicast，在不写raw score workspace的前提下共享Q/K |

下一步最值得尝试的是Hopper CTA cluster的TMA multicast：同一Q/K head对应多个value head时，让多个CTA共享一次Q/K global-to-shared传输，但仍在各CTA内应用自己的gate和计算output. 这样避开了已经失败的raw $QK^T$ global workspace. 不过它需要TileLang稳定支持cluster barrier和multicast mask，改动风险明显高于本轮的dispatch优化.

另一个较安全的方向是进一步缩短named barrier关键路径，例如检查`bar_start`、`bar_state`和`bar_finish`是否能通过更精确的producer/consumer依赖合并. 这里必须逐个barrier验证，因为任何提前覆盖ping-pong buffer都会产生跨chunk数据竞争. 当前100分最小余量仍超过10%，所以后续实验应继续采用单变量修改和两轮完整回归.


## CUDA时间轴占比分析

### 为什么全程热点图中CUDA只占很少一部分

原始nsys命令从Python进程启动一直记录到退出，其中包括：

```text
输入和reference正确性
    -> preprocessing
    -> TileLang JIT/动态库加载
    -> correctness调用
    -> warm-up
    -> 100次正式迭代
    -> profiler导出
```

student core只占其中最后一小段，所以把整个进程缩放到一屏以后，CUDA kernel必然显示成很窄的蓝色竖条. 正确的观察方式是使用：

```bash
GDN_PROFILE_AFTER_WARMUP=1 nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python run.py --case parallel_gva
```

这样capture中只剩warm-up后的100次正式调用.


### 稳态GPU busy ratio

隔离后的旧512-thread profile中，100个student kernel总计80.150 ms，第一个kernel开始到最后一个kernel结束的跨度为86.500 ms：

$$
\mathrm{GPU\ busy\ ratio}
=\frac{80.150}{86.500}\approx92.7\%.
$$

说明旧图中CUDA面积很小主要是观察范围问题，而不是稳态GPU只工作了很短时间.

当前warp-specialized kernel已经更快. 从`new.sqlite`中只取最后100个`kernel_kernel`，得到：

$$
T_{\mathrm{kernel}}=21.641\text{ ms},\qquad
T_{\mathrm{span}}=27.421\text{ ms},
$$

所以：

$$
\mathrm{GPU\ busy\ ratio}
=\frac{21.641}{27.421}\approx78.9\%.
$$

这里busy ratio下降不是kernel退化，而是单次kernel已经从约0.80 ms降到约0.22 ms；每次Python循环、CUDA event创建/销毁、output分配和kernel launch的固定间隙没有同比缩短，所以它们的相对占比变大. 100次调用间总空隙约为：

$$
27.421-21.641=5.780\text{ ms},
$$

即平均每次约58 $\mu$s. 其中一部分位于两对计时event之间，不会全部计入单次event latency.


### 能否通过增加GPU调用比重优化

可以提高吞吐率，但不能简单地增加kernel数量. 当前一次forward已经只launch一个融合kernel；再拆kernel会增加launch和HBM中间量，正好回到最初的热点.

测试过先连续提交100次、最后统一同步. 这种方式能让profile中的kernel busy ratio达到约99.4%，因为CPU不再在每次调用后等待；但是当时单kernel median从约0.803 ms退化到约0.856 ms，而且评分程序仍按单次forward延迟同步测量. 所以它只让时间轴看起来更满，没有提高评分路径性能.

对于真实推理服务，如果同时存在多个独立请求，可以在调用方使用异步提交、多个stream或CUDA Graph，让下一个forward的launch开销和当前GPU计算重叠. `gdn_prefill_forward`内部没有主动`synchronize`，本身已经允许上层这样调度. 但在student接口内部缓存并复用output，或者把多次forward合成CUDA Graph，会改变返回tensor的生命周期和单次调用语义，不能安全用于当前评分.


### 这个方向真正可做的优化

在不改变benchmark语义的前提下，应区分两类目标：

| 目标 | 方法 | 是否直接提高当前得分 |
|---|---|---|
| 让热点图准确 | warm-up后开始capture，排除JIT/reference | 否，只修正测量范围 |
| 提高多请求吞吐 | 上层异步提交、multi-stream、CUDA Graph | 单请求评分不收益 |
| 降低单次host开销 | 减少Python对象、event和allocator操作 | `run.py`大部分不在student提交范围内 |
| 提高单kernel设备利用率 | 自适应CTA grid、TMA流水、减少barrier | 是，当前应继续的方向 |

所以本项目不能以“增加CUDA调用数量”为目标，而应继续提高唯一融合kernel内部的SM/Tensor Core利用率. 已完成的低grid自适应`block_dv`就是这个方向：它没有增加每次forward的kernel launch数，只把单kernel的CTA从4个增加到8个，使`chain_equal`稳定提升约24%. 对wide/deep则应继续研究named-barrier关键路径和不落HBM的TMA multicast，而不是人为批量提交来美化CUDA时间轴.


## 提前启动state update：拆分`bar_finish`

### 当前`bar_finish`的同步过程

当前一个CTA有四组参与chunk流水的线程：

```text
state group : 0--127
value group : 128--255
output group: 256--383
store group : 480--511
```

`bar_finish`的`arrive_count=416`，正好要求上面四组线程全部到达：

$$
128+128+128+32=416.
$$

当前每个chunk末尾的过程可以画成：

```text
state group : gamma_l S_[c] ------------------------┐
value group : value_decay_shared = D W_hat ---------┤
output group: O_[c] = Gamma QS + P W_hat -----------├─> bar_finish
store group : store O_[c-1] ------------------------┘
                                                       |
                                                       v
state group : K^T value_decay_shared -> S_[c+1]
```

也就是说，state group先计算$\gamma_\ell S_{[c]}$，然后在`bar_finish`等待value、output和store三组全部结束，最后才执行：

$$
S_{[c+1]}
=\gamma_\ell S_{[c]}
+K_{[c]}^T\left(\gamma_\ell\Gamma_{[c]}^{-1}\hat W_{[c]}\right).
$$

记：

$$
D_{[c]}=\gamma_\ell\Gamma_{[c]}^{-1},
\qquad
\hat W^{\mathrm{decay}}_{[c]}=D_{[c]}\hat W_{[c]}.
$$

则state update实际只需要：

$$
K_{[c]},\quad
\hat W^{\mathrm{decay}}_{[c]},\quad
\gamma_\ell S_{[c]}.
$$

它不依赖$O_{[c]}$的最终GEMM，也不依赖$O_{[c-1]}$是否已经写回global memory. 因此让state group等待整个`bar_finish`包含了两个不必要的依赖：

```text
state update  --X-->  output GEMM完成
state update  --X-->  上一个output写回完成
```


### 修改方法

新增一个只负责“decayed value已经生成”的named barrier：

```python
bar_decay_ready = T.alloc_barrier(arrive_count=128)
```

value group完成下面的shared-memory写入后立刻通知state group：

```python
for row, dim_v in T.Parallel(CHUNK_SIZE, block_dv):
    value_fragment[row, dim_v] *= gate_decay_shared[row]
T.copy(value_fragment, value_decay_shared)
T.barrier_arrive(bar_decay_ready)
```

state group不再参与`bar_finish`，而是等待`bar_decay_ready`：

```python
T.barrier_wait(bar_decay_ready, chunk % 2)
T.gemm(
    k_shared[chunk % 2, :, :],
    value_decay_shared,
    state_fragment,
    transpose_A=True,
    clear_accum=False,
)
```

修改后的数据流变成：

```text
value group : D W_hat -> value_decay_shared -> bar_decay_ready
                                                   |
state group : gamma_l S_[c] -----------------------┘
                         |
                         v
                  K^T(D W_hat) -> S_[c+1]

output group: P W_hat -> O_[c] ---------------------> bar_finish
store group : store O_[c-1] ------------------------> bar_finish
value group : 后续buffer生命周期同步 ----------------> bar_finish
```

这样state GEMM可以和output GEMM以及上一个chunk的output store重叠.

state group退出`bar_finish`后，原barrier只剩：

$$
128\text{(value)}+128\text{(output)}+32\text{(store)}=288,
$$

所以需要同时把：

```python
bar_finish = T.alloc_barrier(arrive_count=416)
```

改成：

```python
bar_finish = T.alloc_barrier(arrive_count=288)
```

`bar_start`仍保持416，因为进入下一个chunk以前，state、value、output和store四组仍需要在同一个chunk phase上对齐.


### 为什么不会破坏数学顺序

这里需要检查四个可能的数据竞争.

第一，output必须使用更新前的$S_{[c]}$. state group在更新`state_fragment`以前已经执行：

```python
T.copy(state_fragment, state_shared)
```

output和value读取的是`state_shared`快照，而state update修改的是`state_fragment`，所以提前更新不会覆盖$QS_{[c]}$和$KS_{[c]}$的输入.

第二，state update和output是否会同时读取或修改K. output的$QK^T$在等待`bar_state`以前已经完成；后续output只读取`scores_shared`、`state_shared`和`value_shared`，不再使用`k_shared`. state group此时只读`k_shared`，所以没有写冲突.

第三，`value_decay_shared`必须完整写入以后才能开始state GEMM. 这一点由`bar_decay_ready`保证，不能只依赖warp group的执行先后.

第四，producer不能在state GEMM结束前覆盖当前ping-pong K buffer. state、value和output三组仍分别在完成当前chunk后执行：

```python
T.barrier_arrive(data_free[chunk % 2])
```

而producer需要等待`data_free`的384个线程全部到达，所以state group没有完成$K^T(D\hat W)$以前，下一轮TMA不会覆盖`k_shared`.

因此新的偏序关系为：

$$
\begin{aligned}
S_{[c]}&\longrightarrow state\_shared
\longrightarrow QS_{[c]},KS_{[c]},\\
\hat W_{[c]}&\longrightarrow D\hat W_{[c]}
\longrightarrow S_{[c+1]},\\
\hat W_{[c]}&\longrightarrow O_{[c]},\\
S_{[c+1]}\text{完成}&\longrightarrow data\_free
\longrightarrow \text{覆盖chunk buffer}.
\end{aligned}
$$

数学依赖没有改变，只删除了state update对output路径的伪依赖.


### 预计收益和限制

该优化不减少FLOPs，也不减少global memory读取，收益完全来自关键路径重叠：

$$
T_{\mathrm{old}}
\approx T_{\mathrm{value/output尾部}}
+T_{K^T(D\hat W)},
$$

理想情况下变为：

$$
T_{\mathrm{new}}
\approx\max\left(
T_{\mathrm{value/output尾部}},
T_{K^T(D\hat W)}
\right).
$$

实际不会达到完全重叠，因为两个warp group仍共享同一个SM上的Tensor Core、shared-memory bandwidth和调度资源. 如果output GEMM与state GEMM同时发射造成Tensor Core竞争，也可能只得到很小收益，甚至轻微回退.

这个方向更可能改善`long_low_gva`和`deep_gva_state`，因为每个chunk都会少一次不必要等待；`short_tail_state`中chunk较少，收益容易被固定启动开销淹没.


### 实验过程与结果

该修改作为单变量实验完成，没有同时修改output双缓冲：

1. 增加`bar_decay_ready`，将state从`bar_finish`迁出，并把`bar_finish`计数从416改成288；
2. 静态编译通过，`deep_gva_state`首先通过正确性，单case测得2.427 ms；
3. `wide_gva_state`通过正确性并测得2.181 ms，与原来的2.17--2.18 ms基本持平；
4. 完整运行8个case两轮，均为8/8正确，没有出现deadlock或named-barrier phase错误.

两轮完整10/100结果如下：

| Case | 拆分前第1轮 (ms) | 拆分前第2轮 (ms) | 拆分后第1轮 (ms) | 拆分后第2轮 (ms) |
|---|---:|---:|---:|---:|
| `short_tail_state` | 0.103 | 0.103 | 0.104 | 0.104 |
| `chain_equal` | 0.295 | 0.291 | 0.291 | 0.294 |
| `parallel_equal` | 0.275 | 0.273 | 0.273 | 0.271 |
| `parallel_gva` | 0.231 | 0.230 | 0.233 | 0.231 |
| `long_low_gva` | 1.470 | 1.464 | 1.430 | 1.424 |
| `batch_split_gva` | 1.215 | 1.206 | 1.198 | 1.190 |
| `wide_gva_state` | 2.176 | 2.174 | 2.177 | 2.172 |
| `deep_gva_state` | 2.457 | 2.466 | 2.455 | 2.454 |

收益最大的`long_low_gva`从1.464--1.470 ms降到1.424--1.430 ms，稳定提升约2.8%；`batch_split_gva`也有约1.4%的提升. `deep_gva_state`小幅改善，其余case基本落在正常波动内.

这个结果和热点判断一致：barrier拆分没有减少任何GEMM或global load，只让$K^T(D\hat W)$更早进入调度. chunk越多，删除的伪依赖累计越明显；短序列主要受固定启动开销影响，所以不会获得同等比例的收益.

最终保留该修改. 新的同步关系同时满足：

```text
value_decay_shared ready -> state update
state update complete     -> data_free
output/value/store        -> bar_finish
all four groups           -> next bar_start phase
```

因此既缩短了state关键路径，也没有放松producer覆盖ping-pong buffer以前必须等待的真实依赖.


## wide/long 20%专项分析

### 正确的NCU热点

为了避免NCU再次对整个程序采样超时，这里按`kernel_kernel`名称和launch config过滤，只采集一个`block=(512,1,1),grid=(64,1,1)`的`wide_gva_state` student kernel. 关键结果为：

| 指标 | 数值 |
|---|---:|
| DRAM throughput | 78.42% |
| Compute throughput | 39.63% |
| L1/TEX throughput | 61.14% |
| L2 throughput | 26.72% |
| L2 hit rate | 36.85% |
| Scheduler no eligible | 68.81% |
| Registers/thread | 128 |
| Dynamic shared memory/CTA | 204.80 KiB |
| Waves/SM | 4.57 |

这说明wide的第一热点是global-memory等待，不是Tensor Core算力. 每个CTA使用约204.8 KiB dynamic shared memory，已经接近233 KiB配置上限，所以每个SM只能驻留一个CTA；当TMA或barrier没有准备好时，没有第二个CTA用于隐藏延迟.

`wide_gva_state`和`long_low_gva`还有一个共同点：

$$
G=H_v/H_q=4.
$$

当前一个CTA负责一个value head，因此同一Q/K head对应的4个CTA都会读取相同$Q,K$并计算相同raw $QK^T$. 这正好解释了DRAM利用率高而compute利用率只有约40%.


### 本轮逐项实验

| 实验 | wide (ms) | long (ms) | 结果 |
|---|---:|---:|---|
| state barrier拆分后的稳定参考 | 约2.17--2.20 | 约1.42--1.45 | baseline |
| output shared ping-pong，`bar_finish` 288降到256 | 2.179 | 1.464 | wide中性，long回退；output store已被隐藏 |
| 2-CTA TMA multicast | 未进入计时 | 未进入计时 | 课程TileLang没有`T.ClusterKernel`，TIR构建失败 |
| 删除`gate_fragment`，直接把gate乘进score | 2.176 | 1.454 | exp进入score关键依赖链，long回退 |
| internal state改为$[d_v,d_k]$布局 | 2.193 | 1.451 | WGMMA转置布局更慢 |
| Q/K `evict_last`，V/A `evict_first` | 2.170 | 1.427 | 落在正常波动内，没有稳定收益 |
| 将同Q/K的CTA分隔到后续wave | 3.154 | — | 破坏V/A/output相邻head局部性，严重回退 |

上面所有无收益或回退分支均已从最终代码删除. 最终只保留此前已经两轮验证的`bar_decay_ready`和自适应`block_dv`.


### 为什么局部修改无法达到20%

wide从约2.17 ms再降低20%的目标约为1.74 ms. 当前DRAM已经达到78.4%峰值，理论上即使把剩余带宽余量完全利用，也刚好接近这个目标. output双缓冲、寄存器布局和cache hint都没有减少必须传输的字节数，因此不可能稳定带来20%.

TMA multicast本来可以让4个CTA共享一次Q/K global load，但课程环境中的TileLang版本没有cluster kernel API. raw $QK^T$ global workspace虽然不依赖cluster API，但前面已经实测因为额外写回和kernel调度而慢约12%.

所以真正可能接近20%的路径必须同时满足：

```text
Q/K只加载一次
raw QK^T只计算一次
不产生global workspace
CTA数量不能减少
```


### 下一版结构：一个CTA处理4个GVA head slice

可以把当前“一CTA处理一个head的128个value列”改成：

```text
一个CTA负责一个Q/K head
一个CTA仍负责总计128个value列
但拆成4个head x 32列
```

即令：

$$
\mathrm{CTA}(h_q,b_v)
\longrightarrow
\left\{(h_v=4h_q+r,\;d_v=32b_v:32(b_v+1))\right\}_{r=0}^3.
$$

对wide：

$$
H_q\times(128/32)=16\times4=64\text{ CTAs},
$$

和当前$H_v\times1=64$个CTA完全相同；对long也是$2\times4=8$个CTA，与当前相同. 所以并行度不下降.

每个CTA需要同时维护4份$128\times32$ state slice：

$$
4\times128\times32=128\times128,
$$

state fragment总元素数也和当前完全相同. 但是一个chunk的Q/K和raw score可以被4个value head共享：

```text
load Q/K once
compute raw QK^T once
for r = 0..3:
    load V/A/g/b of value head r
    update its 128x32 state slice
    apply its own gate to raw score
    write its 64x32 output slice
```

按每个Q/K group估算，当前4个CTA需要读取4份Q/K，共128 KiB/chunk；新结构只读取一份32 KiB，Q/K global read减少75%. V、A、gate和output仍按value head分别处理，不改变数值.

这条路径有机会达到20%，但需要新增一个专门的GVA kernel，而不是继续修改当前barrier：

1. grid从value-head映射改成`(qk_head, 32-wide value tile)`；
2. state fragment需要保存4个独立slice；
3. Q/K buffer以chunk为生命周期，V/A/g/b buffer以内层value-head为生命周期；
4. 需要两套`data_free`：一套保护Q/K到4个head全部结束，另一套保护V/A ping-pong；
5. raw score必须留在fragment或shared memory，不能写回HBM；
6. 每个head仍使用自己的$A,\Gamma,B$，不能错误共享gated score.

因此当前代码已经完成了20%目标前所需的热点确认和方案排除，但本轮局部实验没有达到20%，不能把节点波动写成有效收益. 若继续追求该目标，应单独实现上述4-head-slice GVA kernel，并保留当前kernel作为equal-head、tail和回退路径.


### 4-head-slice GVA kernel实际实现结果

随后按上面的结构实现了实验kernel：grid为$B\times H_q\times4$，每个CTA维护四个$128\times32$ state fragment；Q/K和raw $QK^T$每chunk只生成一次，V/A/g/b在CTA内按4个value head处理. 为保持Q/K预取，还给Q/K、四份V和四份A建立了chunk ping-pong buffer.

TileLang能够完成TIR和CUDA编译，但kernel一启动就触发：

```text
CUDA error: an illegal instruction was encountered
```

围绕错误逐项尝试：

1. 最初使用一个$128\times128$ state fragment和一个$128\times32$ delta fragment，怀疑寄存器超过warp-group分配上限；
2. 尝试让WGMMA直接累加到state的32列slice，TileLang明确报`Fragment output C must be a full region`；
3. 改成四个独立的$128\times32$完整state fragment，仍为illegal instruction；
4. 将output fragment先写shared memory再写global，排除fragment直接写global问题，错误不变；
5. 调整state/value/output/producer的`set_max_nreg`为232/96/96/24，错误不变；
6. 完全删除动态`set_max_nreg`，错误仍然存在.

因此问题不是数学、输出布局或单纯的寄存器额度，而是当前TileLang/Hopper lowering无法稳定支持“同一warp group在内层head循环中切换多份WGMMA accumulator，同时让多个TMA事务共享chunk barrier”的组合. 该实验kernel已从最终代码和dispatch中完整删除，恢复后的wide/long重新通过正确性.

结合前面的cluster API缺失，当前课程TileLang版本阻断了两条能真正消除跨CTA Q/K重复读取的路径：

```text
TMA multicast: 缺少 T.ClusterKernel
4-head CTA fusion: 生成kernel运行时 illegal instruction
```

所以在不升级TileLang、不编写独立CUDA/CUTLASS kernel、也不改变student输入接口的约束下，没有证据支持wide和long还能稳定再降20%. 当前保留版本仍远低于100分检查点；后续若必须追求这个目标，应把4-head-slice设计迁移到原生CUDA/CUTLASS，以显式控制WGMMA accumulator、mbarrier和TMA transaction，而不是继续依赖当前TileLang自动lowering.


### deep NCU：long scoreboard才是当前主等待

继续用`per-launch-config`过滤到student的`(512,1,1)` kernel，并补采`WarpStateStats`、`InstructionStats`、`SourceCounters`、`MemoryWorkloadAnalysis_Tables`和`Occupancy`：

| 指标 | `wide_gva_state` | `long_low_gva` |
|---|---:|---:|
| grid | 64 CTA | 8 CTA |
| 实际occupancy | 24.58% | 24.91% |
| DRAM throughput | 78.66% | 54.88% |
| DRAM read/write | 289.52/136.89 MB | 136.46/65.65 MB |
| warp cycle / issued inst | 12.81 | 10.47 |
| long scoreboard | 7.58 | 5.17 |
| named barrier | 1.14 | 1.09 |
| wait | 0.84 | 0.82 |

两者每个CTA使用约204.8 KiB dynamic shared memory和128 registers/thread，所以一个SM只能驻留一个CTA，理论occupancy固定为25%. wide有$64/14=4.57$个wave，能够铺满设备，但DRAM已接近80%；long只有8个CTA，最多只用8个SM，所以DRAM比例较低并不代表访存充足，而是grid先欠填充.

PC sampling进一步表明，最高的`long scoreboard`样本大多落在：

```text
SYNCS.PHASECHK.TRANS64.TRYWAIT
@!P0 BRA ...
```

以及`LDG`之后紧接的`STS`上. 即这里的`long scoreboard`同时包含named-barrier等待和global load依赖，不能简单理解为某条Tensor Core指令慢. `WARPGROUP.DEPBAR`对应的barrier等待明显更少，所以继续拆`bar_finish`已经不是第一优先级.

long虽然欠填充，但前面已实测把`block_dv=64`扩展到8-CTA case会把Q/K/A读取和GEMM复制两份，使延迟从约1.43 ms回退到2.14 ms. 因此不能用增加重复工作来填满剩余6个SM.


### gate/beta非合并读取实验

wide的普通global load访问了1,179,648个32-byte sector，约37.75 MB. 其中$g,\beta$的有效数据只有：

$$
2\times BTH_v\times4
=2\times1\times8192\times64\times4
=4\text{ MiB}.
$$

原布局为$[B,T,H_v]$，一个CTA固定一个head并沿token读取；wide中相邻元素跨$64\times4=256$ bytes，warp读取无法合并. 这解释了producer中`LDG`后的`STS`成为采样热点.

先尝试直接把单head列切片交给TMA：

```python
T.tma_copy(g[bb, left:right, bh], g_shared[phase, :], barrier=...)
```

TileLang在编译期拒绝该descriptor，因为源切片最内层只有4 bytes，而TMA要求最内层至少16-byte对齐，因此该方法不可用.

随后只对$BH_v\ge64$的wide路径，先把两者合并转置为：

$$
P=\operatorname{stack}(g^T,\beta^T)
\in\mathbb{R}^{2\times B\times H_v\times T},
$$

再由persistent kernel对连续的64个token使用TMA. 数值正确，但包含packing开销后的10次中位数为2.203 ms，稳定版约2.174 ms，回退约1.3%；long保持原路径也没有收益. 额外kernel、分配及约16 MB packing流量抵消了kernel内部减少的sector，因此该实现已完整删除.

这个方向只有在preprocessing本来就遍历完整$[T,H_v]$时顺手产出head-major sidecar才可能获益，例如让`chunk_local_cumsum`同时写出$g^T$、让`kkt_solve`同时写出$\beta^T$. 但当前forward接口固定，且提交只收集`student/`，单独在forward里重新packing没有价值.


### 当前可继续尝试的优先级

1. **最高收益：跨4个GVA head共享Q/K和raw $QK^T$**。需要可用的CTA cluster/TMA multicast，或把4-head-slice版本迁移到原生CUDA/CUTLASS；这是唯一同时减少DRAM、避免workspace且不降低CTA数的路径.
2. **若允许改preprocessing接口：免费产出head-major $g,\beta$ sidecar**。不能在forward内额外转置；必须融合进已有cumsum/KKT kernel的写回过程.
3. **次要方向：减少寄存器local traffic**。wide有103,432个local-load sector、49,808个local-store sector，但大部分命中L1，远小于Q/K/A/V主流量；需要以SASS spill数量下降为验收条件，不能只调`set_max_nreg`.
4. **不再优先：继续拆named barrier或盲目提高occupancy**。barrier仅约1.1 cycle/issued inst，而wide的long scoreboard为7.58；long的`block_dv=64`也已经证明重复读取成本大于填充收益.

所有实验分支回退后又做了`warmup=5, repetitions=100`验证：`wide_gva_state=2.183 ms`，`long_low_gva=1.423 ms`，两项均PASS，说明最终源码已回到稳定性能区间.


