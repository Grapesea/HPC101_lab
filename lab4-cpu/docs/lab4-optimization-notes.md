# Lab 4 CPU 优化记录

本文严格按 `lab4.md` 的章节顺序记录本次 CPU 路径中实际尝试过的方案。所有计时均来自同一套 `GW250118` 输入；开发容器的 CPU/核数与 OJ 的鲲鹏节点不同，因此 OpenMP 和绑核结果只能作为方向性参考。每个纳入方案都必须重新运行正确性检查。

## 回归核对：`4-90943.log`

后续测试显示，不能直接把不同运行配置的端到端时间相减。`4-90943.log` 的 ABE 单步约 37.1--37.8 s、总时间 1731.12 s，但日志明确显示它实际执行了 `TwoPunctureABE`；`4-91085.log` 使用相同输入的 TwoPuncture cache，ABE 单步约 38.0--38.8 s、总时间反而只有 1535.34 s。因此前者多出的约 220 s 是初值生成成本，不是 ABE 优化导致的负收益。比较演化性能时必须同时报告 `TwoPuncture cache hit/miss` 和 ABE step 平均值。

本轮针对 `J_times_dv` 的额外并行和临时结构复用没有完成可靠的长程验证，已回滚 `src/TwoPunctures.C`、`src/TwoPunctures.h` 和 `src/TwoPunctureABE.C` 到仓库稳定版本（保留其他允许文件中的诊断开关）。回滚版使用 `NRELAX=200`、无新增 TwoPuncture OpenMP。用回滚版重新构建成功；在相同 `TwoPunctureinput.par` 上运行 30 s 短测时，Newton/BiCGStab 前 11 次残差序列与 `4-90943.log` 逐项一致（初始 `|F|=9.566257e-03`，第 1 次 BiCGStab 残差 `1.064e+00`，第 11 次 `4.014e-04`），说明回滚没有引入收敛漂移。由于完整 TwoPuncture 约需数分钟，未在开发容器重复完整 OJ 作业。

数值稳定性证据：`4-90943.log` 和 `4-91085.log` 都完成 40 个演化时间组，轨迹 RMS 均为 0，约束最大值均为 Ham=0.27739667、Px=0.028132512、Py=0.031488238、Pz=0.026503396，并得到 `Trajectory: PASS`、`Constraints: PASS`、`FINAL: PASS`。这表明已完成的优化/缓存路径与稳定基线在 checker 精度内一致；尚未完成长程 checker 的新改动不应提交。

## 准备工作

基线流程为 `compile.sh` 构建、`run.sh` 启动 `TwoPunctureABE + mpiexec ABE`，最后使用 `check.sh` 检查轨迹和约束。CPU 的正式输入为 `Final_Evolution_Time=40`，MPI rank 数为 30，ABE 的 `OMP_NUM_THREADS=1`。此前平台报出的 `syntax error near unexpected token '('` 来自提交脚本/参数被 shell 错误解析，并非数值程序；运行脚本使用 `set -euo pipefail` 并通过绝对路径和 `mpiexec` 参数避免该类解析问题。

## 性能分析

完整 CPU baseline 约 1496--1514 s（冷启动包含初值求解时约 247 s），端到端总时间接近 30 min。`perf` 热点如下：

| 阶段 | 热点 | 采样占比（约） |
|---|---|---:|
| ABE | `compute_rhs_bssn_` | 27.25% |
| ABE | `polint_` | 16.36% |
| ABE | `fdderivs_` | 9.96% |
| ABE | `lopsided_` | 7.33% |
| ABE | `prolong3_` | 6.26% |
| ABE | `kodis_` | 5.13% |
| TwoPuncture | `LineRelax_be`/`LineRelax_al` | 55.35% |
| TwoPuncture | `ThomasAlgorithm` | 16.77% |

ABE 还存在较多 `memset/memmove`、MPI 进度和分配开销；因此优先优化 TwoPuncture 的重复分配和求解器迭代，并关闭只用于调试的全数组扫描。

## 共享优化阶段：TwoPuncture 初值求解

### 编译优化

保留 `-Ofast -ffast-math` 与目标节点适配的 `-mcpu=native`（位置在 `CMakeLists.txt`，可由 `compile.sh` 的 `AMSS_OPT`、`AMSS_ARCH_FLAGS` 覆盖）。相比 `-O2/-O3`，在开发节点上单步演化约由 44 s 降至 38 s。`-ffast-math` 可能改变浮点重排，故每次都比较轨迹、约束和初值文件。

尝试过 `CXX=armclang++ FC=armflang`：能够完成 CMake 配置和编译，但没有在完整 OJ 硬件上完成可重复的端到端计时，因此没有把它写死进脚本，记录为未采用方案。

### 初值求解优化

1. **工作区复用。** 原实现每次 `LineRelax_be/al` 和 `ThomasAlgorithm` 都 `new[]/delete[]`。在 `src/TwoPunctures.C/.h` 中改为每个 OpenMP 线程建立一次 `std::vector` 工作区，并将 Thomas 算法的临时数组作为参数传递。单独复用内存的实测收益很小（约 247 s，分配开销不是唯一瓶颈），但减少了高频分配和碎片化。
2. **聚合 relax 调用。** 将 `NRELAX` 次调用合并到一次 `relax(..., sweeps)`，并在一次 OpenMP 并行区中完成 red/black 两组平面，减少 fork/join。不同 `k` 平面使用 `schedule(static)` 并行；每线程工作区保证无数据竞争。`run.sh` 设置 `AMSS_TWOP_THREADS`（默认 13），`src/TwoPunctureABE.C` 在启动时调用 `omp_set_num_threads`，不会改变 ABE 的 MPI rank 线程数。
3. **减少迭代次数。** `src/TwoPunctures.h` 将 `NRELAX` 从 200 调为 100。TwoPuncture 冷启动约 247 s 降至 205 s（约 17%），`puncture_parameters_new.txt` 完全一致，`Ansorg.psid` 最大绝对差约 `5.33e-14`，ADM 质量仍收敛。该方案纳入最终 CPU 构建，但必须在目标平台再次跑 checker。
4. **预条件子探索。** 只分析了 `LineRelax`/Thomas 热点，没有替换预条件子；替换可能改变收敛步数和稳定性，暂无证据表明总时间更低，未采用。

5. **并行 Jacobian-vector 计算。** 进一步尝试在 `J_times_dv` 中按 `i` 平面并行，并把 `values`、导数结构体工作区改为线程私有，避免原先每个平面反复分配/释放。该路径是 BiCGStab 每次迭代的热点，理论上比只并行线松弛更接近端到端收益。已在 OpenMP 构建中编译通过；当前开发容器无法在 30 秒级完整 case 上稳定计时，且尚未完成长程 checker，因此暂列为候选方案，提交前必须用基线和优化版分别比较 `Ansorg.psid`、`puncture_parameters_new.txt` 及最终轨迹。

### TwoPuncture GPU 化（Bonus）

未迁移到 GPU。当前热点包含大量按平面三对角求解和不规则系数访问；GPU 化需要重新组织 host/device 缓冲区、同步和收敛判定，开发节点没有完成可验证实现。没有改变物理参数来换取速度。

## ABE CPU 演化优化

在 `src/bssn_rhs.f90` 中，原有每次 RHS 调用对 22 个三维数组执行 `sum()` 的 NaN sanity scan。该代码只用于诊断，却会扫描大量内存；现由 `CMakeLists.txt` 的 `AMSS_ENABLE_SANITY_CHECK` 控制，生产默认 `OFF`，调试时可用 `AMSS_ENABLE_SANITY_CHECK=ON ./compile.sh` 打开。关闭扫描不改变 RHS 数学结果。

尝试过调整 MPI rank 数和绑核；开发容器资源受限，且程序提示粗网格可有效利用的处理器有限，无法得出比课程基线 30 rank 更可靠的结论，因此未强制写入运行器。ABE 没有强行链接 Fortran OpenMP，避免 MPI rank 产生线程膨胀。

## ABEGPU GPU 演化优化

本记录只覆盖 CPU 允许范围内的实验；没有修改 GPU 路径，也没有把 CPU 的 OpenMP 修改传播到 GPU 数值 kernel。

## 思考题与正确性结论

主要热点同时包含规则 stencil（`compute_rhs_bssn_`、`fdderivs_`）和 MPI/内存调度；TwoPuncture 则由线松弛和 Thomas 三对角求解主导。所有初值优化均比较输出文件；完整验收应运行：

```bash
./check.sh GW250118/AMSS_NCKU_output golden_cpu
```

目标输出为 `Trajectory: PASS`、`Constraints: PASS`、`FINAL: PASS`。若 OJ 仍失败，应优先检查提交脚本的 shell 语法、工作目录、MPI 启动参数和超时，而不是根据单个轨迹图形差异直接改变物理参数。
