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

## 阶段 4：`4.sh` 提交配置回退（2026-08-20）

用户反馈当前 5×12 配置出现负优化后，重新检查完整提交日志发现：目前唯一有
完整 40 步 checker 证据的几轮（`results/4-96927.log`、
`results/4-111972.log`、`results/4-115514.log`）均为 `MPI=30`、`OMP=1`、
`-mcpu=native`，`This Program Cost` 为 1318.80–1321.91 s，且全部
`Trajectory: PASS`、`Constraints: PASS`、`FINAL: PASS`。因此 `4.sh` 的默认提交
配置已恢复为这套已验证组合：

```text
AMSS_SUBMIT_MPI_PROCESSES=30   (默认)
AMSS_SUBMIT_OMP_THREADS=1      (默认)
AMSS_TWOP_THREADS=20
AMSS_ARCH_FLAGS=-mcpu=native
AMSS_MPIEXEC='mpiexec --allow-run-as-root'
```

5×12 不再作为默认提交路径；若要在真实 60 硬件线程节点重新验证，可显式设置：

```bash
AMSS_SUBMIT_MPI_PROCESSES=5 \
AMSS_SUBMIT_OMP_THREADS=12 \
AMSS_SUBMIT_MPIEXEC='mpiexec --allow-run-as-root --use-hwthread-cpus --bind-to hwthread --map-by slot:PE=12' \
AMSS_SUBMIT_ARCH_FLAGS='-mcpu=tsv110 -msve-vector-bits=256' \
./4.sh
```

这样回退不会丢失实验入口，也不会让未经完整 checker 验证的组合阻塞正式提交。
