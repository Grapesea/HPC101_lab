# ABE 热点循环的 OpenMP 块级并行与 threads×processes 扫描

> ⚠️ **重要更正（真实节点实测后）**：本文第 7、9、10 节最初基于**被 cgroup
> 限流到 4 核的开发容器**的扫描数据，得出了“rank 越少、thread 越多越快”的结论。
> 该结论是**错的**。在未限流的 `lab4` 分区（30 物理核）实测：
>
> | 配置 | 单步 `Computer used`（rank0 CPU 时间） |
> |---|---:|
> | 30 rank × 1 thread（基线） | **≈ 37.3 s** |
> | 15 rank × 2 thread | ≈ 51.5 s |
> | 10 rank × 3 thread | ≈ 52.5 s |
>
> 即 **30×1 最快**，块级 OpenMP 在真实节点上**没有收益**（甚至更慢）。原因见
> §9 与 §10：MPI 的块划分已经按 rank 数铺满并行度，块级 OpenMP 没有额外的
> intra-rank 并行可挖；减少 rank 只会让每个 rank 的块更大、串行工作更多。
> 最终建议：**使用 30 rank × 1 thread**。以下正文保留原始实现记录，但结论以
> 本节更正为准。

本文记录给 `ABE` 演化主热点循环（`bssn_class::Step()` 中遍历
`(Patch, Block)` 并调用 `compute_rhs_bssn` + RK4 子步 + 边界的块循环）加入
**粗粒度 OpenMP 并行** 的尝试过程，以及 `MPI processes × OMP threads`
组合扫描的数据与（修正后的）结论。

## 1. 背景：为什么是“块级”OpenMP

`src/bssn_class.C` 的 `Step()` 是 CPU 演化的核心热点。profiling 显示单步时间
主要由 `compute_rhs_bssn_`（约 27%）、`polint_`、`fdderivs_`、`lopsided_`、
`kodis_` 等构成，它们都在 `Step()` 的块循环里被反复调用。

在本次改动之前，代码里已经存在**细粒度** OpenMP：`diff_new.f90` 的
`fderivs`/`fdderivs`、`kodiss.f90` 的 `kodis`、`lopsidediff.f90` 的 `lopsided`
内部都有 `!$omp parallel do collapse(2) ... if(ex(1)*ex(2)*ex(3) >= 2048)`。
这种写法的问题是：

- 每个 RHS 调用内部会触发几十次 stencil 子程序，每次子程序都各自 fork/join
  一个线程组，**线程组创建/销毁开销被放大几十倍**；
- 而 `compute_rhs_bssn` 里大量整数组运算（`gxx_rhs = ...` 等）并没有被线程化，
  只靠编译器自动向量化。

因此真正的收益点是把并行粒度抬到**块（Block）**这一层：一个 OpenMP 线程组
处理一个 RK4 子步内所有属于本 rank 的块，每个线程独占一个块，线程组创建次数
从“每 stencil 一次”降到“每 RK4 子步一次”。

## 2. 代码改动

### 2.1 `src/bssn_class.C` —— `Step()` 块循环并行化

`Step(lev, YN)` 原本用两层 `while` 遍历 `GH->PatL[lev]` 的 patch/block 链表，
并用 `if (myrank == cg->rank)` 只处理本 rank 的块。改动如下：

1. 在 predictor 之前一次性收集本 rank 在该 level 拥有的块：

   ```cpp
   std::vector<std::pair<Patch *, Block *> > myblocks;
   {
     MyList<Patch> *Pp = GH->PatL[lev];
     while (Pp) {
       MyList<Block> *BP = Pp->data->blb;
       while (BP) {
         if (BP->data->rank == myrank)
           myblocks.push_back(std::make_pair(Pp->data, BP->data));
         if (BP == Pp->data->ble) break;
         BP = BP->next;
       }
       Pp = Pp->next;
     }
   }
   const int nb = (int)myblocks.size();
   ```

2. predictor / corrector 两个 RK4 块循环改为：

   ```cpp
   #pragma omp parallel for schedule(static) reduction(|:ERROR) if(nb > 1)
   for (int b = 0; b < nb; b++) {
     Patch *pp  = myblocks[b].first;
     Block *cg  = myblocks[b].second;
     { /* 原 f_enforce_ga + f_compute_rhs_bssn + RK4 子步 + f_lowerboundset */ }
   }
   ```

3. 两次 `swapList` 循环（corrector 内与 Step 末尾）改为：

   ```cpp
   #pragma omp parallel for schedule(static) if(nb > 1)
   for (int b = 0; b < nb; b++)
     myblocks[b].second->swapList(...);
   ```

要点：

- `schedule(static)`：同一 level 内块大小基本一致，静态调度开销最小。
- `reduction(|:ERROR)`：`f_compute_rhs_bssn` 的 NaN 检测标志跨块做按位或归约，
  保持与原 `MPI_Allreduce(..., MPI_SUM, ...)` 的语义一致。
- `if(nb > 1)`：块数 ≤1 时不进入并行区，避免在粗网格（level 0 只有 9 个块，
  高 rank 数下每 rank 只有 0~1 个块）空开线程组。
- 块之间完全独立（每个 `Block` 的 `fgfs[]` 数组互不重叠），无数据竞争；块循环
  内部不再调用 MPI，MPI 只在并行区之外由主线程执行。

### 2.2 `src/ABE.C` —— MPI 线程级别

```cpp
int mpi_thread_provided = 0;
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_provided);
```

并行区只在块循环内、MPI 调用只在并行区之间由主线程执行，因此
`MPI_THREAD_FUNNELED` 是正确且最安全（不会因 OpenMPI 进度引擎被误用）的级别。
同时加入一行诊断输出，打印 `OMP_NUM_THREADS` 生效后的最大线程数。

### 2.3 `run.sh` —— OpenMP 工作线程栈

```bash
export OMP_STACKSIZE="${OMP_STACKSIZE:-1G}"
```

`compute_rhs_bssn` 在栈上分配约 35 MB 的 Fortran 自动数组。`ulimit -s unlimited`
只影响主线程；OpenMP 工作线程的栈由 `OMP_STACKSIZE` 控制。若不设置，块级并行
跑 2 个以上线程时工作线程会栈溢出（SIGSEGV）。这是本次调试中踩到的关键坑。

## 3. 线程安全分析

- 所有被块循环调用的 Fortran 子程序（`f_enforce_ga`、`f_compute_rhs_bssn`、
  `f_sommerfeld_rout*`、`f_rungekutta4_rout`、`f_lowerboundset`、`fderivs`、
  `fdderivs`、`lopsided`、`kodis`、`polint` 等）都只使用**栈上自动数组**；
  已 grep 确认源文件中没有 `common` / `save` / `module` / `data` 全局状态。
- 块内的 `fgfs[]` 只属于该块，`swapList` 只交换块内指针，跨块无共享写。
- 嵌套并行：块级并行区内，`fderivs` 等内部的 `!$omp parallel do` 因 OpenMP
  默认关闭嵌套并行而退化为 1 线程串行执行，不会造成线程超卖。

## 4. 构建

```bash
env AMSS_ENABLE_OPENMP=ON AMSS_ENABLE_GPU=OFF AMSS_ENABLE_SANITY_CHECK=OFF \
    AMSS_OPT="-Ofast -ffast-math" AMSS_ARCH_FLAGS="-mcpu=native" \
    AMSS_BUILD_DIR="$PWD/build_omp" ./compile.sh
```

`CMakeLists.txt` 已通过 `OpenMP::OpenMP_CXX` / `OpenMP::OpenMP_Fortran` 给
`ABE` 的 C++ 与 Fortran 源都加上 `-fopenmp`（Fortran 侧的 `!$omp` 与 C++ 侧的
`#pragma omp` 都生效）。代码同时保留了 `#ifdef _OPENMP` 保护与裸 `#pragma omp`
（无 `-fopenmp` 时仅告警不报错），因此 `AMSS_ENABLE_OPENMP=OFF` 的基线构建仍可
编译通过（已用 `build_noomp_check` 验证）。

## 5. 正确性验证

块级并行是**逐块独立**的，每个块内的运算顺序与串行完全一致，因此理论上结果应
与串行**逐位一致**。用 12 ranks 分别跑 `OMP_threads=1` 与 `OMP_threads=4`（其余
输入完全相同，1 个演化步），diff 输出文件确认：

<!-- 由扫描结果回填 -->
| 输出文件 | 12×1 vs 12×4 是否一致 |
|---|---|
| `bssn_BH.dat` | 一致（仅首行 `# File created` 时间戳不同） |
| `bssn_constraint.dat` | 一致（同上） |
| `bssn_ADMQs.dat` | 一致（同上） |
| `bssn_psi4.dat` | 一致（同上） |

结论：块级 OpenMP（12 rank × 4 thread）与串行（12 rank × 1 thread）的演化结果
**逐位一致**，说明逐块并行的数值结果与原串行实现完全相同。

此外用 `./check.sh <输出目录> golden` 对 12×1 与 12×4 各做一次 checker：

| 配置 | Trajectory | Level-0 约束最大值 | 结论 |
|---|---|---|---|
| 12×1 | matched 1/100, RMS = 0 | Ham=0.19611147, Px=0.0072621593, Py=0.00321809, Pz=0.0032643355 | PASS |
| 12×4 | matched 1/100, RMS = 0 | Ham=0.19611147, Px=0.0072621593, Py=0.00321809, Pz=0.0032643355 | PASS |

两者与 golden 的轨迹 RMS 均为 **0**、约束最大值逐位一致且 `FINAL: PASS`，证明块级
OpenMP 没有引入任何数值偏差。

## 6. 测试环境与重要限制

开发容器是 Kunpeng 920B / TaiShan-v120（aarch64），`lscpu` 报告 256 核，但容器
cgroup 限制为：

```text
cpuset: 0-63            # 允许调度到 64 个逻辑核
cpu.max: 400000 100000  # CPU 配额 = 4 个核
```

也就是说本容器**被限流到约 4 个核**（`cpu.stat` 中 `nr_throttled` 非零、累计
`throttled_usec` 巨大）。后果是：不管 ranks×threads 怎么配，单步**墙钟时间都被
总 CPU 配额卡死**（约 450 s/步），无法在本容器得到有意义的墙钟加速比；真正能
代表 OJ 60 核节点的是本仓库此前在**未限流**环境下留下的
`results/omp-sweep/`（见 §8）。

因此本次扫描数据只用于：(1) 证明块级 OpenMP 能正确跑通各组合；(2) 记录每步
CPU 时间（`Computer used`，rank0 `clock()`，不受限流影响）与墙钟，作为方向性
参考。最终结论需要结合未限流的旧数据 + 结构分析给出。

## 7. 本次扫描数据（限流容器，1 个演化步）

扫描脚本：`bench_omp_block.sh`。组合与结果：

| ranks | threads | 总核数 | Computer used (s/步) | Total Evolve Time (s/步) | 备注 |
|---|---:|---:|---:|---:|---|
| 12 | 1 | 12 | 106.01 | 397.30 | |
| 12 | 4 | 48 | 60.72 | 195.41 | 与 12×1 做正确性 diff |
| 30 | 1 | 30 | 67.84 | 554.30 | 课程基线 |
| 30 | 2 | 60 | 70.17 | 616.30 | 纯 SMT 超卖 |
| 15 | 2 | 30 | 69.51 | 249.09 | |
| 10 | 3 | 30 | 59.39 | 155.99 | 限流假象，真实节点最慢 |

（`Computer used` 是 rank0 进程 `clock()` 的 CPU 时间，**含 MPI 自旋等待**，因此
rank 越多、等待越多，该值反而越大——它不是纯“每 rank 工作量”，而是 rank 数
上升时自旋等待被放大的体现。`Total Evolve Time` 是墙钟，在本容器被 4 核配额
卡死，绝对数字不可外推。）

> ⚠️ 下表是在**被限流到 4 核的开发容器**里测得的，其“rank 越少越快”的趋势是
> **限流假象**：30 个 MPI 进程争抢 4 个核导致灾难性争用，才显得 10×3 最快。
> 在未限流的 `lab4` 分区（30 物理核）实测，**30×1 才是最快**（37.3s/步 vs
> 10×3 的 52.5s/步）。此表仅保留作过程记录，不可用于选配置。

限流容器内（仅作记录，不代表真实节点）的相对趋势：

1. 限流下 rank 越少墙钟越快（30×1 554s → 10×3 156s），这是 4 核配额下的争用
   假象，**在真实节点上反转**。
2. 12×4 比 12×1 快约 2.0×，只能说明块级 OpenMP 的代码路径正确、有并行能力，
   但真实节点上每个 rank 分不到足够多的块，这一能力用不上。

## 8. 历史未限流数据（`results/omp-sweep/`，2 个演化步，块级 OpenMP 之前）

| ranks | threads | mean Computer used (s/步) | wall/步 (s) | evolve/步 (s) |
|---|---:|---:|---:|---:|
| 30 | 1 | 38.01 | 40.88 | 38.12 |
| 30 | 2 | 54.44 | 43.08 | 40.46 |
| 40 | 1 | 48.36 | 51.26 | 48.53 |
| 48 | 1 | 47.56 | 50.45 | 47.74 |
| 60 | 1 | 46.52 | 49.47 | 46.69 |

该数据说明：在没有块级 OpenMP 时，**30 rank 已经是最优**；再增加 rank（40/48/60）
单步反而变慢——因为固定规模问题上 MPI 通信/opal 自旋等待随 rank 数增长（
`lab4.md` 中 perf 显示 `libopen-pal` 自旋占 60%+）。当时据此提出“减少 rank、用
OpenMP 线程补足并行度”的假设，**但真实节点实测推翻了这一假设**（10×3/15×2 都
慢于 30×1，见顶部更正与 §9），因为减少 rank 并不会让每 rank 分到更多块。

## 9. 网格块分布（决定块级 OpenMP 收益上限）

`Parallel::distribute` 把每个 level 的 patch 切成块后**轮转**分配给 ranks。从各
rank 数日志中的 “uses essencially X processors” 可知各 level 的块数：

| level | 块数 |
|---|---:|
| 0 | 9 |
| 1–4 | 32（每层） |
| 5–8（moving，时间细化，2 倍子步） | ≥ 40（每层） |

关键点（这也是本尝试最终失败的原因）：

- `Parallel::distribute` 的划分粒度是 `split_size = max(min_size, block_size/nodes)`，
  其中 `nodes = rank 数`。也就是说**每层的块数大致随 rank 数同比例缩放**：rank
  多 → 块多且块小；rank 少 → 块少且块大。无论 10、15 还是 30 rank，每个 rank
  在每个 level 上大致都只分到 **1 个块**（最细层偶有 2 个）。
- 因此，MPI 的块划分**已经把块级并行度铺满到每个 rank 一个块**，块级 OpenMP
  （`#pragma omp parallel for` 遍历本 rank 的块）几乎没有可并行的块（`nb≈1`，
  `if(nb>1)` 恒假）。减少 rank 不会让每个 rank 分到更多块，只会让唯一的块变大、
  串行工作变多，而线程又无法切分一个块。
- 真实节点实测也印证了这一点：10×3 与 15×2 的单步 `Computer used` 都比 30×1
  高约 40%（37.3s → 52s），说明 3 个线程只是空转/带开销，并不能补偿少掉的那
  20 个 rank 的 MPI 并行度。

所以结论是：**块级 OpenMP 对这个程序的粒度选错了**；正确结论回到 30 rank ×
1 thread 的纯 MPI 基线（见 §10）。

## 10. 结论与建议（已按真实节点实测修正）

### 10.1 结论

1. 已给 ABE 热点循环（`bssn_class::Step()` 的 predictor/corrector RK4 块循环）
   加入粗粒度 OpenMP 块级并行，并用 `MPI_THREAD_FUNNELED` 与 `OMP_STACKSIZE`
   补齐线程安全与工作线程栈两个前提。**代码正确（12×1 与 12×4 逐位一致、通过
   `check.sh`），但对性能没有帮助。**
2. **块级 OpenMP 是错的粒度**：MPI 块划分已按 rank 数把每层铺成 ~1 块/rank，
   块级 OpenMP 无块可并；减少 rank 只会放大单 rank 的串行工作，线程无法切分
   单个块。真实节点实测：30×1 ≈ 37.3s/步，15×2 ≈ 51.5s/步，10×3 ≈ 52.5s/步。
3. 早期“限流容器内 10×3 最快”是**假象**：开发容器被 cgroup 限流到 4 核，
   30 个 MPI 进程抢 4 核导致灾难性争用，才让“rank 更少”显得快；未限流的
   30 核节点上该假象消失，30 rank 是最优。

### 10.2 对 lab4 分区的最终建议

- **直接使用基线 30 rank × 1 thread**（`AMSS_ENABLE_OPENMP` 可保持 ON，因为
  `nb≈1` 时 `if(nb>1)` 会跳过并行区，几乎无开销；也可回退 `AMSS_ENABLE_OPENMP=OFF`）。
- 不要在 10×3 / 15×2 / 30×2 上浪费时间：它们都慢于 30×1。
- 本仓库 `4.sh` 里 `AMSS_MPI_PROCESSES`/`AMSS_OMP_THREADS` 现已为 10/3，**提交前
  请改回 30/1**。提交时确保 `AMSS_NCKU_Input.py` 的 `MPI_processes`/`OMP_threads`
  与之一致（driver 只读 `.py`，不读环境变量）。
- 若还想在这条线上继续优化，正确方向不是块级 OpenMP，而是：细粒度地并行
  `compute_rhs_bssn` 内部那些整数组语句（`bssn_rhs.f90`），或从 `lab4.md` 的
  perf 结论出发减少 MPI opal 自旋等待——但这两者收益都受限于程序本就
  MPI-bound，需以 `This Program Cost` 端到端为准。
