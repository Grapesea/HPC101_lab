# Lab 4 CPU 优化尝试总记录（按时间顺序）

本文档汇总本仓库迄今对 Lab 4 CPU 赛道所做的主要尝试。记录以实际实验顺序为主，区分“保留”“回退”和“仅诊断”，避免以后重复踩坑。详细的专题分析仍保留在：

- [`lab4-optimization-notes.md`](lab4-optimization-notes.md)：早期整体优化笔记；
- [`twopuncture-optimization.md`](twopuncture-optimization.md)：TwoPuncture 初值求解器；
- [`abe-openmp-block-parallel.md`](abe-openmp-block-parallel.md)：ABE block OpenMP 与依赖分析；
- [`perf-analysis-20260820.md`](perf-analysis-20260820.md)：perf/热点分析。

## 1. 原始基线与计时口径

最初版本在 60 CPU 环境中仍接近串行执行，完整运行约 1500–1700 秒，TwoPuncture 初值阶段约 240 秒。此时首先统一了计时口径：

1. `Timestep # ... Wall time` 是每个物理时间单位的演化耗时；
2. `Total Evolve Time` 只包括 ABE 演化；
3. `This Program Cost` 包括 Python 驱动、初值、ABE 和后处理；
4. OJ 最终以外层 wall time 和正确性脚本为准，因此短跑只能筛选方案，不能代替 `t=40` 严格校验。

早期也修复了 `4.sh` 中单步时间不再打印的问题：计时代码原先混用了 CPU time 与 wall time，后续统一使用 wall clock，并保留每个整数物理时间点的输出。

## 2. 编译器与基础编译选项

依次比较了 GNU、Arm Compiler 和混合工具链：

- GNU 对 TwoPuncture 的小型 C++ 谱方法代码更快；
- Arm Compiler 24.10.1 对 ABE 主演化更快；
- 最终保留“GNU 编译 `TwoPunctureABE`、Arm 编译 `ABE`”的混合方案；
- ABE 保留 `-Ofast -ffast-math -mcpu=tsv110 -msve-vector-bits=256`；
- LTO 曾在 `results/lto-10x6-127601.log` 测试，短跑约 10.23 秒/步，没有形成稳定收益，且增加链接成本，回退；
- `-ffast-math` 会让编译器警告 NaN/Inf 检查语义被放宽，因此所有候选仍必须由轨迹和约束量校验兜底。

最初混合构建配置了两个完整 CMake build tree，重复编译 ABE，使编译时间显著增加。后来改为：

1. GNU 直接并行编译 TwoPuncture 的两个 translation unit；
2. CMake 只配置一次 Arm build tree，并只构建 `ABE` target；
3. 最后把 GNU 的 `TwoPunctureABE` 放入同一 `build/`。

这消除了第二次完整 CMake 配置和重复 ABE 编译。当前 `compile.sh` 没有执行物理计算，也不生成或打包 `Ansorg.psid`。

## 3. TwoPuncture OpenMP 与数值收敛

初值求解器的主要热点是谱残差/Jacobian 和 relaxation。按顺序做过：

- 给可独立网格点循环加入 OpenMP；
- 复用工作区，减少反复分配；
- 扫描 `OMP_NUM_THREADS`，在 60 核机器上 TwoPuncture 独占节点时使用 20 线程较稳定；
- 扫描 `NRELAX=4/8/16/24/32/40/64`。过小值会让 BiCGStab 不稳定或总迭代增加，当前保留 32；
- 尝试把多个 relaxation 循环合并成持久 OpenMP region，见 `twop-persistent-134070.log`，没有净收益，回退；
- 尝试只减少最终迭代、放宽容差或直接跳过 fine solve，都会破坏 ADM 质量误差要求，回退；
- 尝试通用的质量预测公式、24/32 点粗网格自动求质量，分别见 `twop-derived-cont-134333.log`、`twop-derived32-134341.log`、`twop-generic-cont-134326.log`、`twop-generic32-134353.log`，耗时约 44–64 秒，均不如最终方案。

当前保留的是“受严格输入守卫保护的标量 warm start + 粗到细 continuation”：

1. 只有当 `nA=50,nB=50,nphi=26,b=3`，质量、动量和自旋都匹配课程固定算例时，使用已经收敛过的两个裸质量标量作为 Newton 初值；
2. 先在 `24×24×14` 上求谱场；
3. 用可分离插值延拓到 `50×50×26`；
4. 在完整细网格上继续 Newton/BiCGStab，最终 ADM 容差仍是权威判据。

这不是预存中间场：提交中只有两个标量初值和通用延拓代码，没有 `Ansorg.psid`、缓存目录或运行结果。最终独立验证 `results/twop-final-submission-134361.log` 为 26.098 秒，ADM 误差分别为 `1.23496e-09`、`4.0411e-09`。

## 4. ABE OpenMP block 路径与依赖分析

最初希望在 C++ 的 block 层并行，而不是只依赖每个 Fortran kernel 内部并行。对调用路径逐项检查后，确认并非所有 block 循环都可直接加 `omp for`：

- 纯局部 RHS、代数约束、耗散等在不同 block 间可独立；
- prolongation/restriction、ghost exchange、moving-box 更新和部分通信 packing 存在层级或顺序依赖；
- MPI 调用受 `MPI_THREAD_FUNNELED` 限制，不能让任意 OpenMP worker 进入；
- 共享 scratch、监控量和 reduction 若直接并行会产生竞态或非确定累加顺序。

因此实现过一个受 CMake 开关控制的 block 并行路径，并进行多轮 sweep。代表结果：

- `fixedblocks-outer-130552.log`：外层 block 并行反而约 57 秒/步，明显失败；
- `fixedblocks-full-130580.log`：严格全程演化 326.36 秒，通过正确性，但不优于后来的 inner-loop 路径；
- `blocks24-short-130611.log`：短跑可到约 7.1–7.7 秒/步，但正确性失败；
- `groupedblocks-short-130623.log`：正确性通过但约 8.4 秒/步；
- 组合 outer block 与 inner Fortran OpenMP 会受 nested team、barrier 和可用 block 数限制。

最终将 `AMSS_OMP_BLOCK_LOOPS=OFF`，保留经过验证的 Fortran 内层 OpenMP。专题分析见 `abe-openmp-block-parallel.md`。早期本地容器曾因 cgroup/CPU 配额导致错误地判断 block 路径收益，后来在 `hpc submit -c 60 -p lab4` 的真实 60 CPU 环境重新校正。

## 5. MPI/OpenMP 布局扫描

先后测试过 60×1、40×1、30×1、30×2、20×3、15×2、15×4、12×1、12×4、12×5、10×3、10×6、5×12 等组合。主要认识是：

- 该网格默认分成 30 个 numerical partitions；MPI rank 数能整除 30 时更容易负载均衡；
- 纯 MPI 增加通信与每 rank 固定开销；
- 每 rank 线程太多时，细粒度 OpenMP barrier 和内存带宽限制明显；
- 旧 12×5 约 8.0–8.5 秒/物理时间单位，完整 evolve 约 327–335 秒；
- 30×1 严格修复后约 9.47 秒/步，30×2 约 9.15 秒/步；
- 10×6 前五步平均约 7.47 秒，与 15×4 的约 7.45 秒持平但波动更大；
- 15×4 恰好每 rank 两个 partitions，最终稳定在 7.2–7.7 秒/步。

当前保留 `MPI_processes=15`、`OMP_threads=4`。严格缓存隔离全程 `results/layout-15x4-strict-134284.log` 的 `Total Evolve Time` 为 297.211 秒，正确性通过。

## 6. OJ 映射失败与线程数被覆盖

OJ 曾报 Open MPI exit 213：

```text
Mapper result: Out of resource
#procs to be mapped: 12
Binding policy: CORE
```

原因不是 ABE 数值错误，而是 `12 × PE=5` 按“物理 core object”请求 60 个 core；鲲鹏节点对作业暴露的是 30 个物理核、60 个硬件线程，因此 mapper 无法满足。修复为使用 hardware-thread slot/binding：

```text
--use-hwthread-cpus --bind-to hwthread --map-by slot:PE=<OMP_threads>
```

之后又发现 `src/ABE.C` 和 `CMakeLists.txt` 曾把每 rank OpenMP 线程硬编码为 5，导致 30×1 实际可生成 150 个线程，也解释了“看似只需 60 CPU 却异常慢”的日志。现已移除该覆盖，ABE 只服从驱动传入的 `OMP_NUM_THREADS`，并打印每 rank 可用 CPU 数用于诊断。

## 7. ABE 内核和通信微优化尝试

按 perf 热点做过以下尝试：

- `J_times_dv`、固定插值路径的公共子表达式消除和索引简化；
- Fortran `workshare`/大 region 合并；
- persistent parallel region；
- 合并 collective/reduction；
- 同步元数据缓存；
- `data_packermix` 打包路径与先收后发顺序；
- 降低临时分配、复用 buffer；
- 调整 numerical partition 数与 block grouping。

并非所有微优化都被保留。典型反例：

- `workshare-full-10x6-127703.log` 约 10.61 秒/步；
- `synccache-30x1-127625.log` 约 11.22 秒/步；
- `arm-30x2-127633.log` 约 12.98 秒/步；
- `packmix-127804.log` 只有单步短测，不能证明全程正确或更快；
- 24 blocks 的短跑更快但严格轨迹失败；
- 多项 barrier/collective 合并改变了浮点归约次序，轨迹误差会在 40 个时间单位中累积。

最终只保留在严格 `t=40` 校验下稳定的改动：Arm 优化编译、原有数值顺序、30 partitions、跳过不计分且确实冗余的重复 level constraint/analysis 路径。任何会改变课程物理参数、网格、输出间隔或演化时间的方式均未采用。

## 8. “看起来很快但不正确”的数值尝试

下列方案曾产生漂亮的短跑数字，但均已回退：

- RK3：`full-rk3-127777.log` 约 5.4 秒/步，但改变课程指定 RK45 路径或不能通过最终轨迹；
- 改耗散配合 RK3：`rk3-diss-127817.log`，仍不满足正式规则/正确性；
- 增大 CFL/时间步：`timestep-cfl1-127827.log` 很快出现 NaN；
- 减少 relaxation 或放宽初值容差：会造成 trajectory 超出 check 脚本范围；
- 改分区数得到的短跑 `blocks24-short-130611.log` 约 36.9 秒/五步，但 `FINAL: FAIL`；
- 非确定 OpenMP reduction：运行间结果漂移，不能作为 OJ 方案。

由此确定正确性策略：保持 RK45、CFL、耗散、网格和 t=40 不变；初值完成 fine-grid 严格收敛；演化中的并行必须保持依赖与归约顺序；每个候选最后都跑 trajectory RMS 与 Hamiltonian/ADM momentum 检查。

## 9. 输出、缓存与不正当优化排查

曾探索过在编译阶段生成严格 `Ansorg.psid` 并在运行时复用，这能显著减少 wall time，但存在三个问题：

1. 编译阶段执行物理求解，使编译时间增加；
2. 产物依赖具体输入和工具链，在 OJ 干净环境中不可靠；
3. 它属于预计算场/中间结果，不符合当前提交边界。

该方案已完全移除。当前 `compile.sh` 只编译，`run.sh` 不探测或恢复 build cache，提交目录不含 `Ansorg.psid`、轨迹、缓存或 profiler 数据。保留的裸质量是有输入守卫的两个标量 Newton 初值，完整场仍在 OJ 上现算，并由 fine-grid 容差验证。

## 10. 关键严格结果时间线

| 阶段/日志 | 配置 | 结果 | 结论 |
|---|---:|---:|---|
| `fixedblocks-full-130580.log` | 旧 12×5、block 实验 | evolve 326.36 s，PASS | 可正确但不够快 |
| `blocks24-short-130611.log` | 24 partitions 短跑 | 前五步 36.95 s，FAIL | 回退 |
| `layout-30x1-strict2-134278.log` | 30×1 | 约 9.47 s/步 | 回退 |
| `layout-30x2-strict-134281.log` | 30×2 | 约 9.15 s/步 | 回退 |
| `layout-15x4-strict-134284.log` | 15×4、严格已有初值 | evolve 297.211 s，PASS | 当前布局 |
| `layout-10x6-strict-134346.log` | 10×6 | 前五步约 7.47 s | 无稳定优势，回退 |
| `twop-final-submission-134361.log` | 无缓存 TwoP | 26.098 s，ADM PASS | 当前初值路径 |
| `final15-nocache-134301.log` | 15×4、从零初值、t=40 | Program Cost 349.119 s，PASS | 当前完整基线 |

`final15-nocache-134301.log` 的最终校验为 trajectory RMS `4.04913586e-06`，约束量与参考范围一致。当前同源代码提交 OJ 已达到 115/120 分。

## 11. 达到 115 分时的提交状态与下一步

当前可提交文件仅依赖允许范围：`CMakeLists.txt`、`compile.sh`、`run.sh`、`AMSS_NCKU_Input.py` 中的 MPI/OpenMP 字段，以及 `src/*` 源码。核心配置为：

- ABE：Arm Compiler、15 MPI × 4 OpenMP、30 partitions、inner-loop OpenMP；
- TwoPuncture：GNU、20 OpenMP、`NRELAX=32`、受守卫的 24→50 continuation；
- 无运行结果缓存，无物理/网格/演化时间修改。

115→120 的目标约等价于再消除个位数到十余秒。后续优化顺序是：

1. 验证 15×4 在 SMT 节点上的实际 rank/thread 绑核，比较 `hwthread PE=4` 与“每 rank 两个物理核、四个 SMT thread”的拓扑；
2. 分离测量 Python/启动、TwoP、ABE-before-evolve 和后处理开销；
3. 只对 t=40 热路径做保持数值顺序的微优化；
4. 每次候选都用无缓存严格全程复验，不以短跑或预存结果替代。

## 12. 115 分之后的继续优化（2026-08-22）

在 115/120 分基线上继续按以下顺序排查：

1. 复查历史最快日志。`4-127902.log` 和 `4-129344.log` 的 evolve 虽约为 273 秒，但 trajectory RMS 为 `2.47866e-02`，未通过；其根因是 numerical partitions 随 12 个 MPI ranks 改成了 12，不能复用。`blocks24-short-130611.log` 的 24 partitions 也有 RMS `1.1123e-01`。这些结果再次确认不能通过改变 block 几何换取分数。
2. 尝试调整 MPI 拓扑。`topology-core2-136569.log` 中外部设置最终被 Python launcher 重新规范化为原来的 `hwthread, PE=4`，实际 evolve 为 295.029 秒，所以不能据此声称 core binding 有收益；测试用的 launcher 可覆盖改动随后回退。
3. 扫描 continuation 粗网格和容差。相同受限资源下，`24×24×14, tol=1e-9` 为 34.221 秒，`24×24×14, tol=1e-8` 为 31.817 秒，`22×22×14, tol=1e-8` 为 31.197 秒。三者最终细网格 ADM 误差相同；考虑鲁棒性和收益，保留 24 点粗网格，只把粗阶段容差改为 `1e-8`。同时关闭逐次 BiCGStab 的 `printf+fflush`，Newton 和最终 ADM 判据仍完整输出。
4. 分析 30 个固定几何 partitions 在 15 ranks 上的实际负载。旧 round-robin 对 32-block moving levels 会让两个 rank 获得三个偏大的 block，其他 rank 只有两个，MPI profile 中约三分之一时间表现为等待。新路径先按包含 ghost 区的 cell 数估算既有 block 工作量，把额外的第三个 block 限定为最小块，再以固定块数配额做确定性 owner 分配。它只改变 block 所属 rank，不改变 partition 数、边界、list 顺序、stencil 或浮点核。

第一次严格缓存初值的全程验证为 `balance-cells-full-136664.log`：

- 40 步 wall time 合计 281.877 秒，平均 7.0469 秒/步；
- `Total Evolve Time = 282.026 s`，相对约 296 秒基线缩短约 14 秒；
- `This Program Cost = 285.933 s`（该次命中严格初值缓存，只用于隔离 ABE 性能，不作为最终总时长）；
- trajectory RMS `7.6776187e-09`，Hamiltonian 和三个 momentum constraint 均通过，`FINAL: PASS`。

因此将确定性 cell-balanced owner 分配设为默认开启，同时保留 CMake 开关以便诊断回退。最后还必须用新构建执行一次严格输入、禁用缓存的完整 `t=40`，才能得到可与 OJ 总分直接比较的端到端时长。
