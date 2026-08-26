# CPU 端到端 `perf` 分析（2026-08-20）

本文件记录当前工作树的可复现实测。每完成一个阶段就追加结果，避免把旧的
profiling 输出和当前可执行文件混在一起。测试默认使用 `run.sh` 的 CPU 路径、
`AMSS_SKIP_PLOTS=1` 和 TwoPuncture cache；短测只用于定位热点，最终正确性仍需
用完整输入运行 `check.sh`。

## 阶段 0：当前状态与已知证据

- 工作树已经包含 5 MPI × 12 OpenMP 的入口配置、独立的
  `AMSS_TWOP_THREADS`、6 阶插值计划/固定 kernel、合并插值归约、跳过高层重复
  约束和跳过绘图等改动。
- 仓库已有一轮 5×12 采样：
  `results/perf-5x12-1787205375668221028/rank0-report.txt` 与
  `rank0-stat.txt`。它对应 `Final_Evolution_Time=2.0`、2 个演化步和 cache hit，
  不是完整 OJ 运行。
- 旧采样的首要现象是 OpenMP 等待：`libomp.so::kmp_flag_64::wait` 约占 74.74%
  的 rank-0 cycles；`compute_rhs_bssn_` 约 4.70%，`lopsided` 3.30%，`fdderivs`
  2.55%，`kodis` 1.93%，固定 6 阶插值 kernel 0.16%。这说明短测的主要问题是
  barrier/负载粒度，而不是插值算术本身。

## 阶段 1：当前可执行文件的 `perf` 复测

命令（由 `perf_5x12_lab4.sh` 执行）：

```bash
./perf_5x12_lab4.sh
```

环境为 `MPI_processes=5`、`OMP_threads=12`、`AMSS_TWOP_THREADS=20`、2 个短步，
并由 `perf_rank0.sh` 只对 rank 0 执行 `perf stat -d` + `perf record -F 49`。

本轮输出目录为
`results/perf-5x12-1787207493517735331/`。运行在本开发容器中被手动终止，原因是
容器 cgroup 明确限制为 `cpu.max=400000 100000`（约 4 个 CPU），而进程仍创建了
5×12 个 OpenMP 工作线程；8 分钟时还停留在演化阶段，未生成可解析的 perf 计数器。
该目录的 `rank0.data` 因 SIGINT 不完整，不能用于结论；输入文件已由脚本 trap 恢复。

这次失败本身是有效的环境诊断：不能把受 4 CPU 配额限制的本地 wall time 与 OJ 的
60 硬件线程结果比较，也不能在这个容器里据此反向选择 MPI/OMP 拓扑。

## 阶段 2：可用的 5×12 active/passive 对照

仓库中已有完整落盘的两轮相同短测，可直接复核：

| 配置 | `This Program Cost` | rank0 task-clock | IPC | LLC miss | 结论 |
|---|---:|---:|---:|---:|---|
| `OMP_WAIT_POLICY=ACTIVE`, `GOMP_SPINCOUNT=INFINITE` | 58.18 s | 681.13 s | 2.49 | 42.22% | `results/perf-5x12-1787205375668221028/` |
| passive 对照 | 496.42 s | 243.65 s | 1.88 | 42.36% | `results/perf-5x12-passive-1787205609998644049/` |

active 采样的热点为：`libomp.so::kmp_flag_64::wait` 74.74%、
`compute_rhs_bssn_` 4.70%、`lopsided` 3.30%、`fdderivs` 2.55%、`kodis` 1.93%；
固定 6 阶插值 kernel 仅 0.16%。passive 采样则把更多时间交给 MPI progress 和
有限差分（`compute_rhs_bssn_` 14.98%、`lopsided` 12.05%、`fdderivs` 8.53%、
`kodis` 7.08%），但 wall time 显著变长。

因此当前最终配置继续保留：

```bash
export OMP_PROC_BIND=close
export OMP_PLACES=threads
export OMP_WAIT_POLICY=ACTIVE
export GOMP_SPINCOUNT=INFINITE
export OPENBLAS_NUM_THREADS=1
```

`kmp_flag::wait` 的高采样占比表示线程在 barrier 上自旋等待，并不等价于
`compute_rhs_bssn_` 变慢；在 5×12 的目标节点上，active 对照的端到端结果明显更好。
若未来 OJ 节点的 CPU 配额或 MPI 实现改变，应重新做 active/passive A/B，而不是
凭本地 4-CPU 容器的 wall time 修改最终拓扑。

## 阶段 3：代码与配置结论

本轮依据 perf 没有继续改动 BSSN 数值 kernel：当前首要成本是 OpenMP/MPI 等待和
规则有限差分，插值固定 kernel 已不是热点。已完成的一项一致性修正是把入口默认的
TwoPuncture 线程数从 24 统一为最终 `twop20` 配置（`run.sh`、`CMakeLists.txt`；
环境变量仍可覆盖）。

后续在有 60 硬件线程的节点上应采集：

1. 完整 40 时间单位运行的 `perf stat -d` 和 rank0 `perf record`；
2. `perf stat -e cycles,instructions,cache-misses,context-switches,cpu-migrations`；
3. `This Program Cost`、ABE 单步均值、TwoPuncture cache hit/miss；
4. `./check.sh` 的 `Trajectory: PASS`、`Constraints: PASS`、`FINAL: PASS`。

只有在这些指标同时保持时，才考虑进一步调整 rank/线程或等待策略。

## 阶段 4：`4.sh` 提交配置回退（历史记录，2026-08-20）

用户反馈当前 5×12 配置出现负优化后，重新检查完整提交日志发现：目前唯一有
完整 40 步 checker 证据的几轮（`results/4-96927.log`、
`results/4-111972.log`、`results/4-115514.log`）均为 `MPI=30`、`OMP=1`、
`-mcpu=native`，`This Program Cost` 为 1318.80–1321.91 s，且全部
`Trajectory: PASS`、`Constraints: PASS`、`FINAL: PASS`。因此 `4.sh` 的默认提交
配置曾恢复为这套已验证组合：

```text
AMSS_SUBMIT_MPI_PROCESSES=30   (默认)
AMSS_SUBMIT_OMP_THREADS=1      (默认)
AMSS_TWOP_THREADS=20
AMSS_ARCH_FLAGS='-mcpu=tsv110 -msve-vector-bits=256'
AMSS_MPIEXEC='mpiexec --allow-run-as-root'
```

这段是历史 A/B 记录；当前最终入口按题目要求采用 5×12。若要复现当时的 30×1，
仍可显式设置：

```bash
AMSS_SUBMIT_MPI_PROCESSES=5 \
AMSS_SUBMIT_OMP_THREADS=12 \
AMSS_SUBMIT_MPIEXEC='mpiexec --allow-run-as-root --use-hwthread-cpus --bind-to hwthread --map-by slot:PE=12' \
AMSS_SUBMIT_ARCH_FLAGS='-mcpu=tsv110 -msve-vector-bits=256' \
./4.sh
```

这样回退不会丢失实验入口，也不会让未经完整 checker 验证的组合阻塞正式提交。

## 阶段 5：真实节点并行度实验

短测统一为两步、cache miss 的 TwoPuncture + ABE，使用 60-CPU 节点；因此表中
`This Program Cost` 包含初值生成，重点比较 ABE 每步：

| 配置 | ABE step 1/2 | 总耗时 | 结果 |
|---|---:|---:|---|
| 30×1, `-mcpu=native` | 约 11.4/11.5 s（正式作业前 33 步均如此） | — | 当前最佳基线 |
| 60×1, `-mcpu=native` | 12.95/13.48 s | 86.58 s | rank 增加后通信变重 |
| 40×1, `-mcpu=native` | 14.99/15.02 s | 89.05 s | 更差 |
| 30×2, `PE=2` | 26.36/26.37 s | 213.61 s | OpenMP barrier/短循环开销过大 |
| 15×4, `PE=4` | 59.11/59.51 s | 219.35 s | 网格块粒度不足，排除 |
| 30×1, TSV110+SVE256 | 11.24/11.28 s | 80.26 s | 约 2% 快于 native，尚无完整 checker |

60×1 的第一次尝试还暴露了 Open MPI slot 语义：超过 30 ranks 时必须使用
`--use-hwthread-cpus`，否则会在 ABE 启动前报 “not enough slots”。`4.sh` 现在会
根据 rank 数自动加入该选项；30 ranks 默认仍使用普通 launcher。

## 阶段 6：有效 30×1 perf

修正 `4_s.sh` 使其同步输入文件中的 MPI/OMP 字段并隔离输出后，真实采样位于
`results/perf_cpu-20260820-075501-110412528/`。热点为：

- `libopen-pal` 约 34.7%，Open MPI progress/通信是最大单项；
- `compute_rhs_bssn_` 21.62%；
- `lopsided` 6.03%、`fdderivs` 4.17%、`kodis` 3.30%；
- 固定插值 kernel 0.37%；
- LLC miss 56.89%，IPC 1.98。

尝试 `--mca pml ucx` 在该节点启动失败，未纳入配置。综合结果，继续增加 MPI
rank 或给每个 rank 增加 OpenMP 都会变慢；这只是当时安全默认的结论。按实测约
11.5 s/步估算，完整 40 步约 460 s，再加 TwoPuncture 约 60–80 s，**仅靠 MPI/OMP
拓扑无法压到 300 s**。若必须逼近 300 s，需要新的算法级改动（例如减少物理输出/约束
频率或重写通信阶段），这些会直接触及评分正确性，不能仅凭短测启用。

补充实验：20×1（TSV/SVE）两步为 12.50/12.43 s，总 86.01 s；没有优于 30×1。
显式 `pml=ob1,btl=self,sm` 的试验因提交环境把逗号拆进环境变量，实际只传入了
`btl=self`，MPI_Init 失败，未纳入比较。默认 launcher 保持 Open MPI 的自动 BTL 选择。

## 阶段 7：surface-integral collective 合并

在 `src/surface_integral.C` 中把波形实/虚部的两次归约合成一次，并把 ADM 质量、角动量、
线动量的 7 次标量归约合成一次。Arm 编译通过；30×1、TSV/SVE、2 步 checker 结果为
`Trajectory: PASS`、`Constraints: PASS`、`FINAL: PASS`，`This Program Cost=80.998 s`
（未合并版本同量级约 80–81 s，短测噪声内）。这是低风险通信优化，应保留，但它不足以
改变 40 步总时长约 508–513 s 的结论。

## 阶段 8：最终 5×12 入口与计时修正

最终 `AMSS_NCKU_Input.py`/`4.sh` 默认同步为 `MPI_processes=5`、`OMP_threads=12`，
并使用 `--use-hwthread-cpus --bind-to hwthread --map-by slot:PE=12` 及 active wait
策略；TwoPuncture 仍由 `AMSS_TWOP_THREADS=20` 单独控制。历史日志中 5×12 的
`Computer used` 曾显示 262–320 s，是 `clock()` 累加 12 个线程 CPU time 的结果，
而两步的 `Total Evolve Time` 只有约 44–54 s。CPU `Evolve()` 现在改用 `MPI_Wtime()`
打印 wall-clock，后续单步比较必须使用该字段或 `Total Evolve Time`，不能混用旧日志。

本地受 cgroup 限制的短测不能代表 60 硬件线程节点；最终提交应在目标节点完整跑 40
步，并同时记录总墙钟、阶段 profile、轨迹 RMS、约束最大值和 `FINAL: PASS`。300 s
是端到端目标，任何进一步减少演化步数或跳过 level 0 约束的做法都会改变评分语义，
不应作为性能优化启用。
