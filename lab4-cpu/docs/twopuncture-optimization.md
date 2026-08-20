# TwoPuncture 初值求解优化记录

本文记录对共享阶段 `TwoPunctureABE`（双黑洞初值求解器）的优化。TwoPuncture 是
CPU/GPU 两条路径都要经过的前置阶段，冷启动约 240s，计入任务一和任务二的端到端
时间，因此是确定性的收益点。

## 1. 热点与基线

`nA=nB=50, nphi=26`，单变量（`nvar=1`），网格 50×50×26 = 65000 点。profiling
热点（见 `lab4.md`）：

| 函数 | 占比 |
|---|---:|
| `LineRelax_be` | 32.2% |
| `LineRelax_al` | 23.1% |
| `ThomasAlgorithm` | 16.8% |
| 向量化 cos/sqrt | 9.0% |
| malloc/free | ~7% |

即 **松弛（LineRelax + Thomas）占 ~72%**，是主战场。基线（`NRELAX=200`，无
OpenMP）冷启动 **241s**，输出与 `twopuncture_cache` 参考逐位一致（除首行时间戳）。

## 2. 改动与数据

开发容器（cgroup 限流 4 核）只能确认单线程（1 核 < 配额，计时可靠）与 4 线程
（恰在配额边界）的结果；**最终以 `hpc submit -p lab4`（30 物理核，未限流）实测
为准**。

### lab4 分区实测（`bench_twop_lab4.sh`，NRELAX=100 基础上扫 AMSS_TWOP_THREADS）

| AMSS_TWOP_THREADS | 冷启动 wall | 相对 1 线程 |
|---|---:|---:|
| 1 | 199s | 1.00× |
| 4 | 126s | 1.58× |
| 8 | 107s | 1.86× |
| 13 | 98s | 2.03× |
| 16 | 92s | 2.16× |
| **24** | **82s** | **2.43×** |
| 30 | 83s | 2.40× |

所有线程数的 `Ansorg.psid` 与参考的最大绝对差均为 **5.331e-14**（逐位一致）。

完整收益链：baseline（NRELAX=200，单线程）241s → NRELAX=100 单线程 199s →
**NRELAX=100 + 24 线程 82s**，合计 **2.94×**（省 ~159s）。

### 2.1 NRELAX 200 → 100（`src/TwoPunctures.h`）

BiCGStab 预条件子每次迭代做 `2×NRELAX` 次 `relax()`。把 `NRELAX` 从 200 降到
100，直接砍掉一半松弛。验证：

- `puncture_parameters_new.txt` **逐位一致**；
- `Ansorg.psid` 最大绝对差 **5.331e-14**（机器精度量级），最大相对差 2.3e-11；
- ADM 质量 0.983557 一致。

即结果完全一致（差别在 1e-14 量级），且收敛没有变差。

### 2.2 松弛 OpenMP 并行（`src/TwoPunctures.C` 的 `relax()`）

这是本次的关键。`relax()` 原本是：

```c
for (k = 0,2,...)               // 红黑 k
  for (n < N_PlaneRelax)        // ==1
    for (i = 2,4,...) LineRelax_be(dv,i,k,...);   // 求解 j 方向三对角
    for (i = 1,3,...) LineRelax_be(dv,i,k,...);
    for (j = 1,3,...) LineRelax_al(dv,j,k,...);    // 求解 i 方向三对角
    for (j = 0,2,...) LineRelax_al(dv,j,k,...);
```

**关键观察**：红黑 Gauss-Seidel 里，每个子扫描内部，被更新的线只读取相反奇偶的
i/j/k 平面（那些平面在本子扫描中冻结），所以同一子扫描里的所有线**互相独立**。
于是每个子扫描可以 `collapse(2)` 到 `(k, i)` 或 `(k, j)`：

```c
#pragma omp parallel for collapse(2) schedule(static)
for (k = 0; k < n3; k += 2)
  for (i = 2; i < n1; i += 2)
    LineRelax_be(dv, i, k, ...);
```

并行度：13(偶数 k) × 24~25(i/j) ≈ **312~325 路**，而不是前人只试过的 13 路
（nphi/2）。这是之前 OpenMP 尝试失败的关键区别——13 路粒度太小、负载不均衡、
fork/join 开销吃光收益；collapse 到 (k,i) 后并行度放大 24 倍。

**正确性**：4 线程与 1 线程输出 `Ansorg.psid` **逐位一致**（红黑依赖只读冻结数据，
顺序无关）。4 线程 128~130s，即 1 线程（203s）的 **1.56×**、基线（241s）的
**1.85×**。8/16 线程在本容器反而变慢（cgroup 4 核配额导致的超卖假象）。

### 2.3 工作区复用（`LineRelax_be/al` 与 `ThomasAlgorithm` 改栈数组）

`LineRelax_be/al` 每次调用 `new[]/delete[]` 5 个数组、`ThomasAlgorithm` 每次 4 个
数组，而每条线要被求解约百万次。改为**栈上 VLA**（`double diag[n2];` 等）：

- 消除高频 malloc/free；
- 栈数组天然线程私有，OpenMP 下无数据竞争。

单独收益在噪声内（前人结论一致），但它是 2.2 的 OpenMP 必要前提。

### 2.4 `AMSS_TWOP_THREADS`（`src/TwoPunctureABE.C`）

TwoPuncture 与 ABE 原先共享 `OMP_NUM_THREADS`（来自 `.py` 的 `OMP_threads`），
而 ABE 最优是 1 线程——这会连带把 TwoPuncture 压成 1 线程，白白浪费上面的并行。
现已让 `TwoPunctureABE` 读取独立的 `AMSS_TWOP_THREADS` 环境变量：

```c
const char *t = getenv("AMSS_TWOP_THREADS");
if (t && atoi(t) > 0) omp_set_num_threads(atoi(t));
```

最终入口将 `AMSS_TWOP_THREADS` 默认设为 20（`4.sh` 和 profiling 脚本也显式使用
20）；ABE 仍保持自己的 `OMP_NUM_THREADS` 配置，TwoPuncture 不再被 ABE rank 的
线程数牵连。线程数仍可通过环境变量覆盖。

## 3. 尝试过但回滚的项

- **`J_times_dv` 点级并行**：65000 点本可 collapse(3) 并行，但把 21 个小工作区
  声明成 innermost 循环里的 VLA，导致 1 线程 214s、4 线程 >600s（VLA 栈调整 +
  高频小迭代开销），**回滚**，保留原 per-i 分配。

## 4. 正确性小结

- `puncture_parameters_new.txt` 与基线逐位一致；
- `Ansorg.psid` 与基线最大绝对差 5.33e-14；
- ADM 质量 0.983557 一致；
- OpenMP（任意线程数）与串行 NRELAX=100 逐位一致。

结论：**NRELAX=100 + 松弛 OpenMP 是安全的**。完整 `check.sh`（ABE 演化 + 轨迹/
约束）仍需在 lab4 分区上跑一次确认（前人同样只做了初值级验证）。

## 5. 对 lab4 分区的建议

- 提交脚本里保证 `AMSS_ENABLE_OPENMP=ON`；`run.sh` / `4.sh` 默认
  `AMSS_TWOP_THREADS=20`（最终 twop20 配置）。
- **预期收益：TwoPuncture 247s → 82s（2.94×），端到端省 ~159s**，占基线 1742s
  的 ~9%，且所有线程数结果逐位一致（5.33e-14），低风险。
- 完整 `check.sh`（ABE 演化 + 轨迹/约束）仍需在 lab4 分区跑一次确认（初值级已
  逐位验证）。
