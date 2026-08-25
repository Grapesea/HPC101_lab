# GPU 性能分析和优化记录

测试是在 `lab4g10` 队列做的，资源固定为 1 个 A100 MIG 1g.10gb、16 CPU 和 24 GiB 内存。CUDA 架构固定 `sm_80`，最终输入仍然是课程给出的 GPU 物理/网格参数和 `Final_Evolution_Time=100.0`。短测时我才把时间暂时调成 1.0，目的是缩短 profile 迭代；正式提交前会恢复成 100.0。

提交指令统一采用：

```bash
hpc submit -p lab4g10 -c16 -g1 -m24Gi -t30m -d -o 'results/xxx-%j.log' "..."
```

`MPI_processes=1`，`OMP_threads=16`。GPU 是一个 MIG slice，不能用多 rank 去抢同一个 CUDA device。TwoPuncture 的 CPU 阶段使用 16 个线程，ABEGPU 保持 1 个 owning rank。

## 先跑 baseline

旧的完整 GPU 结果 `results/4g-143633.log` 是 `PASS`，总时长为 1736.447 s。这是后面判断端到端是否真的达到 370 s 的基线，不能只看一个 kernel 快了多少。

为了做小步调参，我保持相同 cache 和硬件跑了 t=1。未优化 ABEGPU 的演化时间是 18.7915 s，Python 端总时间是 23.0698 s。这个短测不用于最终成绩，只用于各轮之间的横向比较。

## perf / Nsight 分析

CPU 上的 `perf` 只能看到 host 正在 `cudaDeviceSynchronize` 等待，不能把这个时间直接当成某个 kernel 的时间。所以这里先用 `nsys` 看完整 GPU 时间线，再用 `ncu` 看热点 kernel。

`nsys-stats` 的 CUDA kernel 汇总是：

| kernel | GPU 时间占比 | 调用次数 | 单次均值 |
| --- | ---: | ---: | ---: |
| `rhs_kernel` | 77.9% | 420 | 33.136 ms |
| `prolong3` | 12.0% | - | - |
| `restrict3` | 3.6% | - | - |
| `global_interp` | 2.5% | - | - |

CUDA API 汇总中 `cudaDeviceSynchronize` 是 91.8%，而 H2D/D2H memcpy 只有约 5%。这说明此处不是先去做异步 memcpy，而是先把 `rhs_kernel` 做快。

最初直接在 MIG 上跑 NCU 失败了，因为默认尝试锁 GPU clock：

```text
Cannot lock GPU clock frequencies on MIG
```

改为 `--clock-control none`，并只取一个 RHS launch：

```bash
ncu --target-processes all --clock-control none \
    --kernel-name regex:rhs_kernel --launch-count 1 \
    --section LaunchStats --section Occupancy ...
```

结果如下：

| 指标 | 数值 |
| --- | ---: |
| block size | 256 (8x8x4) |
| registers per thread | 250 |
| theoretical occupancy | 12.50% |
| achieved occupancy | 11.04% |
| register 限制的 block 数 | 1 |
| NCU 给出的理论局部改善空间 | 87.5% |

这里结论比较明确：RHS 是寄存器受限，不是 shared memory 限制。并且 stencil 的一阶/二阶导数每次都经过对称边界 helper；绝大部分内部点本来不需要它。

我也实际跑了一次 `perf stat -d ./run.sh --twop-cache`。这个短测的 Python 总时间是 12.2678 s，`perf` 给出的主要数字是：

```text
14,250.88 msec task-clock, 0.963 CPUs utilized
35.708 B cycles, 84.742 B instructions, IPC 2.37
L1-dcache-load-misses 0.68%, LLC-load-misses 30.86%
```

看起来 LLC miss 不低，但这不是 CPU 算法突然需要优化的信号。整个进程基本只用了 0.963 个 CPU，时间主要停在 CUDA 同步等待；所以 `perf` 在这里的用途是确认 host 没有新的大热点，真正的 kernel 排序仍然看 Nsight。

## 已验证的两轮优化

### 1. 调整 RHS 的 occupancy

我把 RHS 的线程块从 `8x8x4` 改为 `8x8x2`，并设置：

```cpp
__launch_bounds__(128, 4)
```

这样要求 ptxas 在 128-thread block 至少保留 4 个 block 时权衡寄存器和 spill。它不是改精度，只改变 GPU 的调度和寄存器分配。

| 版本 | t=1 evolve | 相对未优化 |
| --- | ---: | ---: |
| baseline | 18.7915 s | 1.00x |
| occupancy | 14.5504 s | 1.29x |

这一轮有收益，所以保留。

继续把 minimum blocks 提到 6 时，device link 直接拒绝：RHS 被限制到 80 registers/thread，但它调用的 `d_fdderivs_point` 仍需要 106 registers。说明这个版本不能用更硬的 register cap 来换 occupancy，因此回退到 4 blocks。

### 2. stencil 内部点快速路径

在 `diff_new_gpu.cu` 中，一阶和二阶导数都增加了严格的内部点判断：离任一边界至少两格时，按原来的四阶系数直接用线性索引读取 `f[p+offset]`；边界点继续走原来的 `d_symmetry_bd_1b`。因此对称条件、ghost/boundary 的数值路径没有被删掉。

| 版本 | t=1 evolve | t=1 program cost | 相对未优化 evolve |
| --- | ---: | ---: | ---: |
| baseline | 18.7915 s | 23.0698 s | 1.00x |
| occupancy | 14.5504 s | 18.9564 s | 1.29x |
| occupancy + interior stencil | 10.4818 s | 14.0291 s | 1.79x |

### 3. 其余 RHS stencil 与重复约束

`d_lopsided_point` 和 `d_kodis_point` 也有同样的模式：内部点已经保证距边界三格以上，却仍然逐项调用对称边界 helper。因此为两个函数补了直接连续读取路径，t=1 总时长继续降到 13.8200 s。

另外，`bssn_step_gpu.C` 的高层 AMR predictor 原先会计算约束，但同一时间步的 `Constraint_Out()` 随后又会把这些高层约束重算。这个重复来自 GPU 路径没有同步 CPU 路径已有的优化。现在只令 `lev>0` 的 predictor 传入 `co=1`，所以 level 0 和最终 `Constraint_Out()` 的约束仍完整计算。

| 版本 | t=1 evolve | t=1 program cost | 相对未优化 evolve |
| --- | ---: | ---: | ---: |
| + lopsided / KO direct path | 9.5020 s | 13.1023 s | 1.98x |
| + high-level constraint reuse | 9.8508 s | 13.4238 s | 1.91x |

这两组 t=1 运行在相邻的 MIG 实例上，存在小幅波动；所以最终以正在运行的完整 `4g.sh` + `check.sh` 结果为准，而不是以这两个相差很小的短测排先后。保留它们是因为都没有改变方程、网格、步数或最终约束输出。

短测正常结束。这里的收益符合前面的 profile：RHS 里会反复调用一阶、二阶 stencil，避免内部点的函数调用和边界检查会同时减少指令和寄存器/stack 压力。

### 4. 缩短 RHS 的 live range

RHS 里所有项原来在一个巨大的 kernel 中做完，平流/KO 部分会让二十多个 field 的 stencil 调用和 Ricci、constraint 的中间量同时处于同一个编译单元里。这里没有换方程，只把“已经写入 RHS 的物理项”之后的平流/KO 加法拆到独立 kernel，仍在同一个 CUDA stream 上依次执行。

为了让这个 kernel 内的 derivative 真的能内联，我在 `bssn_rhs_gpu.cu` 只包含了一份私有的 `diff_new_gpu.cu` helper；其他 CUDA 文件继续用原来可调用的版本。这样不会影响其他 kernel 的边界实现。

| 版本 | t=1 evolve | t=1 program cost |
| --- | ---: | ---: |
| RHS private derivative inline | 8.6807 s | 12.2678 s |
| 再拆出 advection/KO | 8.1812 s | 12.1988 s |
| prolong 内部点直读 | 7.9078 s | 11.9222 s |
| restrict 内部点直读 | 7.4404 s | 11.4179 s |
| 平流公共量只算一次 | 6.9359 s | 11.0144 s |
| 内部平流/边界 fallback 分离 | 6.7782 s | 10.8501 s |

这里所谓“公共量”是每个点的 `dx/dy/dz`、shift 和 `1/(12*d*)`。原来 24 个 field 各自都会再取一遍；现在内部点只取一次，边界仍然直接回退到 `d_lopsided_point` / `d_kodis_point`。

最后一版再跑 Nsys，热点已经发生了明显变化：

| kernel | GPU 时间占比 | 调用次数 | 单次均值 |
| --- | ---: | ---: | ---: |
| `rhs_kernel` | 30.1% | 420 | 5.202 ms |
| `rhs_advection_kernel` | 29.4% | 420 | 5.074 ms |
| `prolong3_kernel` | 19.7% | 11955 | 119.366 us |
| `global_interp_kernel` | 7.4% | 344 | 1.564 ms |
| `sommerfeld_rout_kernel` | 5.1% | 9312 | 39.944 us |
| `restrict3_kernel` | 3.1% | 2391 | 92.829 us |

这一次没有再把 `cudaDeviceSynchronize` 当作 CPU hotspot：它仍占 CUDA API 时间的 81.9%，只是 host 等待上述 kernel 的正常表现。

平流 kernel 的 NCU（首个 `(5,5,10) x (8,8,2)` launch）是 85 registers/thread，理论 occupancy 31.25%、实测 26.23%。NCU 也提示有 partial wave，但 420 次调用的网格尺寸不相同，不能为了一个 launch 把所有 grid 人为补齐。

我试过直接把它改成 `__launch_bounds__(128, 7)`，想把 31.25% 再往上推。这个版本在 device link 被拒绝：边界回退的 `d_lopsided_point` 需要 85 registers，而 7 个 128-thread block 的上限是 72。这个错误本身说明边界 helper 反过来限制了内部快路径，所以最终改成两个 kernel：内部 kernel 使用 7-block contract，边界 shell 继续用原来的 4-block generic helper。二者按同一 stream 顺序执行，t=1 进一步到 6.7782 s / 10.8501 s；这轮是已构建、已运行的，不是只停在推测。

AMR 插值还做了一轮等价整理：host 本来已经知道 coarse/fine patch 的 `lbc/lbf` 和有效 target box，但原实现让每个 device thread 再算一次 `CD/FD/base`。把这部分固定整数映射传给 kernel 后，t=1 为 11.0346 s，和 11.0144 s 在 MIG 的波动范围内。它没有带来可单独宣称的收益，说明编译器已经消掉了不少纯标量计算；保留只是为了让边界与内部 filter 的数据流更清楚，不能把它算进加速比。

继续用 NCU 看 `prolong3_kernel` 的一个代表性 launch，得到 128 registers/thread、理论 occupancy 25%、实际 21.72%，并且每线程有 1024 B stack frame；SpeedOfLight 中 L1/TEX throughput 为 69.18%、memory throughput 为 60.03%。这和前面 Nsys 中 prolong 占 19.7% 是同一个问题：不是再加 shared memory 就能解决，先要把临时二维数组导致的 local-memory traffic 清掉。

原来内部直接路径会先把 36 个 z 向结果写进 `tmp2[6][6]`，再做 y 向归约。其实每个固定的 `n` 在完成六次 z 向积累后立刻就可以做完 y 向归约，最后仍保留六个 `tmp1[n]` 给 x 向、并保持 z/y/x 三个方向的原加法顺序。我把 `tmp2` 消掉、把这两层循环加了固定长度 unroll；A100 MIG 的 t=1 cache-hit 结果从 10.8501 s / 6.7782 s（program/evolve）到 **10.4140 s / 6.4043 s**。约束短测仍为 `PASS`；trajectory 的 coverage 仍因只有 t=1 而不能通过 100 个时间点的完整 checker。这一项保留。

另外几项没有留下来：`16x8x1` block 比 `8x8x2` 慢；把 RHS 硬压到 6 个 block 会和 derivative helper 的 register contract 冲突；把 constraint 分支模板化、或强制内联 generic lopsided/KO 都让 t=1 变慢。还试过把 Sommerfeld kernel 的完整三维 launch 改成只枚举六个表面；虽然线程数变少，但 t=1 evolve 是 6.4092 s，和 6.4043 s 持平偏慢，说明这里本来就由真正的表面插值和大量小 launch 主导，少发空线程不足以抵消并行粒度变差，因此也回退。这里都没有为了“看起来 occupancy 更高”而留一个实际更慢的版本。

### 5. 共享的 TwoPuncture 冷启动

一开始 GPU 目录仍然保留 `NRELAX=200` 的串行红黑 line-relax；CPU 目录的文档已经有一轮完整的分析，说明这里可以在每个红黑子扫描内按 `(k,i)` 或 `(k,j)` 并行，子扫描之间仍保持原有 barrier。这个阶段不在 GPU 上跑，却是 GPU 端到端时间的一部分，cache miss 时不能忽略。

我把这段已验证的结构移到 GPU 目录：`NRELAX=100`、八个 `collapse(2)` 的静态 OpenMP 子扫描，以及 `LineRelax_*`/Thomas 的线程私有栈工作区。没有改 TwoPuncture 网格、收敛阈值或物理参数。`OMP_threads=16` 正好供这个单进程初值阶段使用；ABEGPU 本身仍是一个 CUDA-owning MPI rank。

为避免误用已有 cache，单独创建空 cache 目录跑 t=1。初值求解完成后的 ADM 质量为课程值 `0.598837` / `0.401163`，Python 端到端为 210.630 s（其中后面的 GPU t=1 只有约 10.85 s）。这组数据不能和 cache-hit 的 10 s 短测混在一个表里，它只说明冷启动能够收敛并给出冷启动成本。

随后对这份 t=1 输出运行 `check.sh`：level-0 约束为 `Ham=0.19611147`、`Px=0.0072621593`、`Py=0.00321809`、`Pz=0.0032643355`，约束检查通过。trajectory 部分显示 `matched 1/100` 后因短测只产生一个时间组而报 coverage incomplete，这不是数值 mismatch。完整 t=100 checker 才能作为这个改动的最终接受证据。

还试过把每次 BiCGStab 的 `J*dv` 展开为十个导数场的逐点线性组合，并预先缓存 `LinEquations` 的系数。这个想法在纸面上是等价的，但实际冷启动里，构建一次系数需要遍历 50×50×26 个点各做十次变换；在 144516 的空 cache 运行中，末轮 BiCGStab 每推进 3 个迭代就已经消耗约 30 s，明显不如原始按需计算。该 job 在没有产生可用结果前主动取消，代码已撤回，不能把它写成收益。

CPU 目录还有一个把 `NRELAX` 直接压到 2 的实验版本。把它和固定参数的粗谱预测一起移来做冷启动时，fine-grid 的第一个 BiCGStab 已经走到 90 次迭代，下一次 Newton 又继续在高残差区徘徊；144556 没有在合理时间内完成，故也撤回到已经有数值证据的 `NRELAX=100`。这里宁可少报一项初值收益，也不把没完成收敛的预条件器当作加速。

保留 `NRELAX=100` 后，固定课程参数的标量暖启动和 24×24×14 谱预测在 144574 的空 cache t=1 完整跑完。它仍执行 fine-grid Newton，最终 ADM 为 `0.598837` / `0.401163`，`This Program Cost=72.5173 s`，而此前同样无 cache 的安全版本为 210.630 s；后段 ABEGPU t=1 evolve 为 6.4350 s。`check.sh` 的约束仍通过（Ham=0.19611147，Px=0.0072621592，Py=0.00321809，Pz=0.0032643355），trajectory 只因 t=1 缺少 99 个时间组而 coverage incomplete。这个暖启动已保留；最终仍要用 t=100 的完整 checker 验收。

## 未保留的 LTO 尝试

我也尝试过 CUDA device LTO：各 `.cu` 先生成 `lto_80` 中间码，并在 device link 阶段生成 `sm_80`。原因是 `rhs_kernel` 和 derivative/symmetry helper 原本跨编译单元，普通 separable compilation 不会在这条边界上内联。

曾经直接加 `-dlto` 的版本不能构建，CUDA 13.3 报错它与 CMake 自动传入的 `-gencode` 冲突。改成显式 `code=lto_80` 后可以完成 device link，但首个 ABEGPU launch 在 `gpu_manager.cu:79` 报 `CUDA Error: an illegal memory access was encountered`。因此两种 LTO 版本都撤回，不能拿一个未通过 t=1 的二进制去冲最终时间。

最终只接受同时满足下面三项的版本：

1. `sm_80` A100 MIG 上完整 t=100；
2. OJ 输出 `FINAL: PASS`；
3. `This Program Cost` 不大于 370 s。

任何只在 t=1 快、或只让某个 kernel 快但完整时间变慢的修改都不算完成。

### 6. `global_interp` 的 local-memory 问题

前一张 Nsys 表里，`global_interp_kernel` 已经到 7.4%（344 次、每次 1.564 ms）。它的
输入固定是六点插值，但原实现先放一个 `6x6x6` 的局部数组，再做 Neville 插值。这个
局部数组很容易落到 local memory；和 prolong 的 `tmp2` 是同一种味道。

我把它改成三个一维 Lagrange weight 的张量积：仍然是六点、仍走同一套镜像/SoA 取值，
只是不用先物化 216 个数。cache-hit t=1 的 evolve 从 6.4043 s 到 5.7997 s；随后把
prolong/restrict generic fallback 的 `tmp2[6][6]` 也改为流式 z→y 归约，结果为 5.7616 s。
两版短测的 level-0 constraint 分别仍是 Ham=0.19611147、Px=0.0072621592、
Py=0.00321809、Pz=0.0032643355，约束检查通过。

同一版再用 Nsys 重跑，排序已经变成：

| kernel | GPU 时间占比 | 调用次数 | 单次均值 |
| --- | ---: | ---: | ---: |
| `rhs_kernel` | 33.9% | 420 | 5.052 ms |
| `rhs_advection_kernel`（边界 fallback） | 22.5% | 420 | 3.354 ms |
| `prolong3_kernel` | 20.4% | 11955 | 106.460 us |
| `rhs_advection_interior_group_kernel` | 7.8% | 1680 | 291.284 us |
| `sommerfeld_rout_kernel` | 5.7% | 9312 | - |
| `restrict3_kernel` | 3.8% | 2391 | - |
| `global_interp_kernel` | 0.3% | 344 | 61.787 us |

所以这轮把波形插值从 7.4% 压到了 0.3%，确实不该再把时间花在它上面。与此同时，
内部平流的 24 个字段彼此独立，我将它拆成每次 6 个字段的 kernel；cache-hit t=1
变为 5.7173 s，仍通过 constraint 短测。这一项留下。

我还试过用相同办法拆 boundary fallback。结果 144751 的 evolve 为 5.7748 s，反而比
5.7173 s 慢；边界 shell 太小，四个 launch 和并行粒度损失不值得。数值约束仍过，但
代码已回退。`rhs_kernel` 删除未使用 pointer 的试验同样没有收益（5.782 s），也已回退。

### 7. 冷启动再减一半，但不放宽最终验收

谱预测已经足够接近根后，fine predictor 仍用 `1e-9` Newton tolerance；我把 coarse 与
已有谱预测的 fine predictor 都改为 `2e-9`，而最终 ADM 接受线仍保留课程的 `1e-8`。
若预测不被 ADM 检查接受，程序仍会沿原来的完整 Newton 路径求解，并没有把最终初值
验收线调松。

144714 用空 cache 的 t=1 中，粗解停止在 `1.905e-9`，fine predictor 一次 Newton 后
为 `7.269427e-10`；两个 ADM 误差为 `1.23714e-09`、`4.03847e-09`，都小于 `1e-8`。
端到端 `This Program Cost` 从暖预测版的 72.5173 s 降到 **36.7311 s**，后段 evolve
为 5.7921 s；constraint 短测也通过。这个改变只减少已接受初值的无用迭代，最终 t=100
仍必须以 checker 的 `FINAL: PASS` 验收。

### 8. 现在的方向：AMR 同变量批处理

最新版 Nsys 里 prolong/restrict 合计约四分之一 GPU 时间，但其中 `prolong3` 有 11955
次，说明它很像“同一段几何对 24 个 state field 分别起一个小 kernel”。我没有改变
stencil，而是把这 24 个 field 放进同一个 launch 的字段维度；coarse/fine 几何、six-point
系数和每个 field 自己的 SoA 仍逐点调用原来的 device routine。

144811 的 cache-hit t=1 从 **5.7173 s** 到 **5.4856 s**（约 4.1%）。constraint 仍是
Ham=0.19611147、Px=0.0072621592、Py=0.00321809、Pz=0.0032643355，约束检查通过；
trajectory 仍只因为短测缺少 t=2..100 而 coverage incomplete。故这条已保留。这里的
收益并不神奇：每个 field 仍然做完整 216-point 读取，省的是 24 次重复 launch 与重复的
几何调度，不是把数学工作偷偷删掉。

### 9. boundary fallback 不再启动整个体网格

上一轮 Nsys 中 `rhs_advection_kernel` 还有 22.5%。它本来只处理离边界三格内的点，
但 launch 却还是覆盖整个 `ex0*ex1*ex2` 体网格，内部线程一进来立即 return。对于短 z
方向 patch，这些空线程比例很高。

这里按互不重叠的 x、y、z 六个三格 slab 建立了一维紧凑索引：x 面保留完整 y/z，y 面
只枚举 x 的内部，z 面只枚举 x/y 的内部，刚好覆盖原来的 boundary shell 一次。每个真实
边界点之后仍调用原来的 `d_lopsided_point` 和 `d_kodis_point`，SoA 与对称边界没有另写
近似版；网格任一维不大于六点时则回退成完整索引，以免没有 interior 的极端 shape 出错。

144831 的 cache-hit t=1 evolve 为 **4.9034 s**，比批 AMR 的 5.4856 s 再降约 10.6%。
同一输出的 constraint 最大值仍是 Ham=0.19611147、Px=0.0072621592、
Py=0.00321809、Pz=0.0032643355，约束检查通过。trajectory checker 只匹配到 t=1，
因为本轮仍是 profile 短测而非 t=100；这不能替代最终 `FINAL: PASS`。下一步会重新
profile 此版本，而不是用旧 Nsys 百分比猜剩余热点。

### 10. 将高层的 RK 与边界复制合并

检查 `sommerfeld_rout_kernel` 的调用点后发现，在 `lev>0` 固定传入的是
`precor=CORRECTSTEP`。此分支不使用 `chi/Lap/dT/SoA` 的插值公式，只在物理边界执行
`f=f0`；但原实现仍对 24 个 field 分别发一次完整 3D kernel。与它相邻的
`rungekutta4_rout` 也正好逐 field 更新同一段数据。

因此我给 24 个 field 放了一个 grid.y 维度，在同一 kernel 先严格按原 RK stage 更新，
再用原 Sommerfeld 的 bbox/对称判定把真实边界覆盖为相同的 `f0`。level 0 的
`sommerfeld_routbam` 是有物理导数计算的，仍按原顺序逐 field 运行；不是被错误地合并掉。
device pointer table 放在 block 中、随原 stream 更新，所以不会跨 patch 或跨 stream 取错。

144877 的 cache-hit t=1 evolve 是 **4.7609 s**（对 4.9034 s 再快约 2.9%），constraint
仍为 Ham=0.19611147、Px=0.0072621592、Py=0.00321809、Pz=0.0032643355，约束通过。
这说明合并没有把正确步的边界覆盖漏掉。它的完整 t=100 数值一致性仍须由最终 checker
单独确认。

### 11. 紧凑边界上的逐方向直接 stencil

144893 的新版 Nsys（用 `--force-export true` 刷新旧 SQLite 后得到；采集本身没有失败）
已经变成 `rhs_kernel` 43.2%、边界 `rhs_advection_kernel` 24.3%、batch prolong 13.7%。
后者虽然只枚举 boundary shell，但原 `d_lopsided_point` 只要 x/y/z 中任一方向贴边，就让
三个方向全走 `d_symmetry_bd_1b`。事实上例如 x 面上 y/z 的五点都在同一条合法连续行中，
根本不需要对称检查。

我将判定缩到每个方向：该方向距两端至少三格时使用和原公式逐项相同的直接五点读取，
只有贴边的那个方向继续走原有上风/二阶退化/SoA 对称 helper。KO 项原本需要完整 7 点
盒；外壳中不满足该条件时原函数最后也是返回 0，因此把这个判断提到构造 SoA/lambda 之前。
没有减少任何实际有效的 KO 或 lopsided 系数。

144904 的 cache-hit t=1 evolve 为 **4.5801 s**，相对 4.7609 s 再降约 3.8%。约束最大值
仍是 Ham=0.19611147、Px=0.0072621592、Py=0.00321809、Pz=0.0032643355，constraint
通过；轨迹项仍只因 t=1 coverage 不完整而失败。这一版保留。

我还试过把 RHS 做成两份模板 kernel，使常规演化 `co=1` 不携带只给约束输出用的代码。第一版
用 `cudaLaunchKernel` 动态打包接近一百个参数，在 144924 首步报告 illegal memory access；
这条动态-launch 写法已立刻撤回，不能计入任何性能数据。随后改为普通 CUDA launch 选择模板
实例，再单独验证。这里记下失败原因，是为了避免后来把工具/ABI 问题误记为数值问题。

### 12. RHS 模板的完整 t=100 验证

改成普通 CUDA launch 后，144939 的 cache-hit t=1 evolve 是 **4.4853 s**，比 144904 再快
约 2.1%，并通过短测 constraint。NCU 捕获到的第一个 constraint 实例仍是 128
registers/thread、1024 B stack frame，理论 occupancy 25%、实测 21.77%。这里不能把它误读为
常规 `co=1` 实例的精确寄存器数；它只说明大 RHS 仍然是后面最值得继续追的热点。

我还把物理 RHS 试成 `8x6x2`、`__launch_bounds__(96,6)`，希望用 96-thread block 增加 resident
block 数。144967 的 t=1 evolve 却是 4.9332 s，明显慢于 4.4853 s，已回退。这个例子也说明本
题不能只照 occupancy 数字选 block shape。

以 144939 的普通 template launch 做了第一次完整 t=100：144991 的 Python 输出为
`This Program Cost = 462.382812 s`。作业末尾的 checker 曾错误使用了未展开的
`/GW250118/...` 路径，所以该 job 被标为 failed；我随后用绝对路径单独提交 145050，结果是：

```text
Trajectory: matched times 100/100, effective terms 596
Trajectory RMS: 5.40436485e-05 (0.005404%)
Constraints: 100 time groups, 9 levels per group
Constraint maxima (level 0): Ham=0.28974818, Px=0.039343255,
  Py=0.047298113, Pz=0.044686546
FINAL: PASS
```

这证实截至这里的方程、时间推进和输出都没有被优化破坏，但 462.383 s 仍没有达到 370 s。后面的
kernel 调参只能以完整 `PASS` 版本为基线，不能把 t=1 的 cache-hit 数字当作最终成绩。

### 13. 继续调 RHS resident block

在不改变 `8x8x2` block shape 的前提下，我只把 RHS 的 contract 从
`__launch_bounds__(128,4)` 调成 `__launch_bounds__(128,3)`。145056 的 cache-hit t=1
中，`After Step` 为 **4.44251 s**、Python 总时间为 8.05123 s；对照 144939 的 4.48532 s，
收益约 0.95%。同一输出的 constraint 最大值仍是 Ham=0.19611147、Px=0.0072621592、
Py=0.00321809、Pz=0.0032643355，约束通过。full checker 在短测里仍只会因 `matched 1/100`
报 coverage incomplete。

这个收益很小，MIG 的抖动可能吞掉一部分，因此它只能先作为候选保留，而不能据此声称已经解决
462→370 的缺口。下一轮继续检查独立 patch 的 stream 并发；数学输出和每一个 kernel 的执行顺序
在同一 stream 内都不变。

把 stream pool 从 4 增至 8 的 145075 短测并没有如预期增加有效并发：`After Step` 反而是
**4.98698 s**，总时间 9.28197 s，约束数值保持不变。这比四 stream 的 4.44251 s 慢约 12%，
说明这个 14-SM MIG slice 在该 AMR 调度下已经不适合再叠更多同时运行的大 kernel；8-stream
代码已回退为 4。这是实测否定，不把“更多 stream”当成免费优化。

同样，物理 RHS 的 `16x4x2`（128 threads）在 145094 是 **4.55331 s**，也慢于
`8x8x2` 的 4.44251 s；内部平流在这个比较中刻意仍是 `8x8x2`，所以差异确实来自 RHS
shape 而不是另一个 kernel。该尝试也已回退。x 方向更长虽然看似更连续，但 warp 的 y-row
切换和不规则 patch 边界在这里抵消了它，不能仅从一维访问模式推断收益。

最后重新试了 boundary fallback 的字段分组。这次是在已经有“逐方向 direct stencil”的版本上
做的：每个紧凑 boundary thread 仍是原来 24 个 field 的数学项，只是分成 metric、A、
scalar/Gamma/Lap、shift/dtSf 四个六字段 kernel，让一个实例不用同时带 48 个 field/RHS
pointer。145113 的 `After Step` 是 **4.40809 s**，总时间 8.02957 s，较 4.44251 s 有约
0.8% 的收益；constraint 仍为 Ham=0.19611147、Px=0.0072621592、Py=0.00321809、
Pz=0.0032643355。这和较早的 boundary group 失败版不同，后者还没有 per-axis direct
路径；当前这版已经短测验证，所以保留，仍等待 t=100 验收。

我曾把 level-0 predictor 也设为 `co=1`，145140 的短测是 **4.38839 s**，比 4.40809 s
快约 0.45%，checker 的最大值也没有立刻报警。后续重新沿着 `Constraint_Out()` 的调用链检查后
发现这条不能保留：高层 constraint 会在 regrid 后重算，但 level 0 的每个时间组正是依赖
predictor 产生的诊断数组。短测和现有 checker 只检查阈值，因此会把上一时间组的 level-0 数值
误判为“仍然合格”；这不是可以为了计时跳过的必要输出。最终代码已恢复为仅 `lev>0` 传 `co=1`，
level 0 继续计算 constraint。145140 保留为一次有价值的错误尝试，而不是有效加速比。

### 14. 这一轮的反复测量：不要把 MIG 抖动写成优化

后面重新做了一次干净的 t=1 对照，发现同一份“跳 predictor constraint、四个 boundary field
group”的二进制可以是 4.61967 s；把旧的 all-skip binary 原样再跑一次又是 4.63130 s。这说明
短测中的约百分之几波动是真实存在的，后面所有低于 1% 的差异都只当作候选，不能单独写成收益。

不过 AMR 批处理的 block size 有非常清楚的信号。批 `prolong3`/`restrict3` 从 256 改为 128
threads 后，145240/145246 两次 cache-hit t=1 分别是 **4.19290 s** / **4.20245 s**；相比之下
64 threads 的 145257 为 **4.60091 s**。给 128-thread kernel 再加 `__launch_bounds__(128,5)` 的
145271 为 **4.20520 s**，没有比无该限制的 128-thread 版本更好。因此只保留 block=128，不保留
额外的 launch-bounds。两次 128 的重复结果相差仅 0.23%，是这段调参里比较可信的一项。

内部平流的 block 则试成 `16x4x2`。145284 的 `After Step` 是 **4.13001 s**，但同版本复跑的
145296 是 **4.40610 s**；它可能略好，也完全可能被 MIG 的状态吞掉，所以只把它当作待最终完整
运行裁决的候选。RHS 本体若把 `8x8x2` 改成 `8x4x4`，145376 是 4.16184 s，复跑 145422 却是
4.58877 s；这个方差太大且没有稳定优势，源码已经回到 `8x8x2`。

也有几条明确应当放弃的试验：RHS 的 `__launch_bounds__(128,2)` 在 145303 为 **4.57439 s**；给
所有 RHS 读写 pointer 加 `__restrict__` 的 145332 为 **4.55226 s**；把 boundary 的 metric 和 A
合成 12 field kernel 的 145353 为 **4.51007 s**。它们都通过短测约束，却比候选慢，所以没有留下。
曾短暂把物理 RHS 再分成 interior/edge 两个大 kernel，145178 为 **4.63635 s**；不仅慢，还让代码
复杂度显著增加，也已完全撤回。中间一次撤回不完整导致 145192 illegal memory access，随后以干净
版本重建验证；这个错误只反映临时 ABI/源码状态，不是可接受的数值结果。

### 15. 重新 profile 现在的热点

当前候选（RHS 128-thread、四组紧凑 boundary、AMR batch=128、内部平流候选 block）再跑一次
`nsys profile --trace=cuda,nvtx,osrt --sample=none`。profile 运行本身为 9.493 s，包含 tracing
开销，不能和普通 t=1 计时比较；只读取导出的 CUDA kernel 汇总：

| kernel | GPU 时间占比 | 调用次数 | 单次均值 |
| --- | ---: | ---: | ---: |
| `rhs_kernel<false>` | 44.4% | 392 | 5.193 ms |
| `prolong3_batch_kernel` | 10.6% | 485 | 0.998 ms |
| `rhs_advection_interior_group_kernel` | 9.0% | 1680 | 0.247 ms |
| 四个 boundary group 合计 | 23.0% | 1680 | 约 0.63 ms |
| RK batch | 4.4% | 392 | - |
| `rhs_kernel<true>` | 3.2% | 28 | - |
| `restrict3_batch_kernel` | 1.2% | - | - |

这次排序有两个直接结论。第一，AMR batch=128 的确已经把 `prolong3` 从早期约两成压到一成左右，
继续在 global interpolation 上花时间不合适。第二，常规演化的 `rhs_kernel<false>` 一项仍占 44.4%，
而真正带 constraint 的实例只占 3.2%；下一步若要挤出端到端时间，目标必须是普通 RHS 的寄存器/
缓存停顿，而不是删除最终 checker 所需要的 constraint。

对同一候选用下面的 NCU 命令只收一个 RHS launch：

```bash
ncu --target-processes all --clock-control none --kernel-name regex:rhs_kernel \
    --launch-count 1 --section LaunchStats --section Occupancy \
    --section SpeedOfLight --csv -o results/ncu-rhs-latest ./run.sh --twop-cache
```

`--clock-control none` 是 MIG 必需的，否则 NCU 会报不能锁 GPU clock。捕获到的 `rhs_kernel<1>`
实例是 `(8,8,2)`、128 threads、grid `(5,5,10)`；它有 168 registers/thread 和 1024 B stack
frame，理论 occupancy 18.75%、实测 16.19%，register limit 为 3 blocks/SM。SM throughput 是
23.15%，memory throughput 29.77%，DRAM 12.23%，L1/TEX 20.80%。也就是说它既不是把显存带宽打满，
也没有足够 resident warps 去隐藏长 stencil 的依赖；NCU 的 81.25% “local speedup”是理论 occupancy
上限，不能直接当成真实可获得的加速比。这个首 launch 可能是诊断实例，不能据它给普通 `<false>`
实例直接下寄存器数结论；但配合 Nsys，它足以说明 RHS 依然是当前最有价值的 profile 目标。

最后试了 `cudaFuncSetCacheConfig(..., cudaFuncCachePreferL1)`：RHS 没有 dynamic shared memory，
所以这是一个安全的 A100 cache-partition 实验，意图是优先留普通 L1 给邻点读取。它不改方程、
网格或浮点运算顺序；若 t=1 有稳定收益，仍须先过完整 t=100 checker，再由完整时间决定是否保留。

### 16. 当前完整候选的第二次端到端验收

在 `sm_80`、1 MIG、16 CPU、24 GiB、`MPI_processes=1` 和 `OMP_threads=16` 固定不变的条件下，
145448 使用四 stream、AMR batch=128、四个紧凑 boundary group、全 predictor constraint skip 和
`cudaFuncCachePreferL1` 跑完 t=100。结果为：

```text
Total Evolve Time: 432.357 s
This Program Cost = 435.878981 s
Trajectory: matched times 100/100, effective terms 596
Trajectory RMS: 0 (0.000000%)
Constraints: 100 time groups, 9 levels per group
Constraint maxima (level 0): Ham=0.19611147, Px=0.0072621593,
  Py=0.00321809, Pz=0.0032643355
FINAL: PASS
```

这比最初普通 template RHS 的完整 PASS（462.382812 s）少了 26.504 s，约 5.7%；但离 370 s 还差
65.879 s。它说明 L1 preference 与前面保留的结构性修改可以一起通过现有 checker，却不能说明 L1
本身贡献了多少：它的两个 t=1 为 4.19397/4.17241 s，和 AMR=128 的 4.19290/4.20245 s 在同一噪声
范围。更重要的是，随后发现该 job 的“all predictor skip”遗漏了 level-0 的新 constraint output；
故 435.879 s 只作为性能诊断，不作为可交付的最终验收。最终候选会恢复 level-0 constraint 后再跑
完整 checker，且继续针对占比 44.4% 的普通 RHS 做测量。

### 17. 同一点 stencil / 分母复用，以及一次输出链复核

`rhs_kernel` 的内部点会对同一个 `(i,j,k)` 调用十多次一阶导数 helper。每次 helper 都重新判断
自己是不是 interior，并重新读取 `X[1]-X[0]` 等三个间距。这里增加了 RHS 私有的严格 interior
一阶 stencil：主 kernel 只判断一次，内部点复用 `idx`、stride 与 `1/(12d*)`；贴边点仍无条件调用
原有 symmetry-aware helper。它不是把 boundary stencil 改成近似式。

同一 kernel 中 `chin1` 还出现几十次双精度除法。将 `1/chin1` 计算一次、再用于张量各分量的
乘法，只是代数重排；double 舍入是否会积累仍以完整 checker 判断。145487（尚未修正 level-0
constraint 输出的诊断配置）给出：

```text
Total Evolve Time: 415.650 s
This Program Cost = 419.405330 s
Trajectory RMS: 0
FINAL: PASS
```

相对同一 all-skip 诊断配置的 435.879 s，这两项合在一起少 16.474 s，说明主 RHS 确实吃到了
一部分重复 setup/双精度除法成本；但它们仍是组合结果，不把两者拆成虚假的独立加速比。随后沿
`Constraint_Out()` 复核发现 level 0 不能 all-skip，所以 419.405 s 同样不是最终可交付时间。

这次复核还找到一个更针对 boundary 的优化：紧凑三层 shell 上，KO 的完整七点盒在绝大多数点本来
就不成立，原来却为每个 field 单独进入 `d_kodis_point` 后才返回零。现在先按原函数完全相同的
`imin/jmin/kmin` 规则在每个 boundary point 判一次；不成立时六个 field 都跳过这项零加法，成立时
仍调用原 KO 实现。该改动和恢复后的 level-0 constraint 一起进入下一次完整 t=100 验证。

### 18. 只接受真实 level-0 输出后的完整对照

先把上一节的 KO hoist 与正确的 `pre=(lev==0)?0:1` 合在一起，145540 跑完了完整 t=100：

```text
Total Evolve Time: 429.206 s
This Program Cost = 433.411329 s
Trajectory: matched times 100/100, effective terms 596
Trajectory RMS: 0
Constraint maxima (level 0): Ham=0.28974817, Px=0.039343259,
  Py=0.047298107, Pz=0.044686547
FINAL: PASS
```

这组才是 all-skip 诊断版的可交付对照。它比 145448 慢/快都不是重点；重点是 level 0 每一个
输出时间组的 constraint array 都确实由 predictor 写出，不能再把检查器没立即报警误当成数学工作
可以删除。

接着看 boundary。`d_fderivs_point`、`d_fdderivs_point` 和 `d_lopsided_point` 的 fallback 原本即使
某个 sample 已经在 `f` 的合法范围内，仍先走一次 `d_symmetry_bd_1b`。这里仅把这个确定的 in-bounds
sample 改成同一 SoA 线性下标的直接读取；真正落在镜像/越界的位置继续调用原 helper。RHS 的私有
lopsided copy 同时让这条小 helper 能被 ptxas 看见，KO 仍保留上面的 seven-point 判定。

145601 是这版的完整结果：

```text
Total Evolve Time: 394.312 s
This Program Cost = 398.429636 s
Trajectory: matched times 100/100, effective terms 596
Trajectory RMS: 0
Constraint maxima (level 0): Ham=0.28974817, Px=0.039343259,
  Py=0.047298107, Pz=0.044686547
FINAL: PASS
```

相对 145540，端到端少 34.982 s（约 8.1%）。这和 Nsight 中普通 RHS + boundary group 仍占大头
是吻合的；也说明不该只盯着全 interior 的 fast path，AMR 小块的 face/edge/corner 才是剩下的大头。

### 19. 二阶 derivative 的每点复用，最终 t=100

profile 之后重新检查主 RHS，发现一阶导数已经有每点的 interior 判定，但二阶 helper 仍被同一
thread 调用 11 次：三个 shift、六个 metric、`chi` 和 `Lap`。每次都重复读取 `X[1]-X[0]`、建立
stride、构造 six second derivatives 的系数，再判断一次相同的 interior 条件。

最终版本在 `rhs_kernel` 只判一次 `rhs_fast_interior`。若为 interior，就复用 `idx`、`sy/sz` 和
原函数中完全相同的 `1/12`、`1/144` 系数，执行从 `d_fdderivs_point` interior branch 逐式抄出的
表达式；若不是，则仍原样调用 symmetry-aware `d_fdderivs_point`。反射 helper 也只在这条 fallback
中内联，函数体按 `fmisc_gpu.cu` 的范围检查、x/y/z 反射顺序与 SoA 乘法逐句保留。这里没有提升
边界阶数，更没有用 interior stencil 覆盖 boundary。

正式命令仍是课程规定的资源和 `hpc submit`：

```bash
hpc submit -p lab4g10 -c16 -g1 -m24Gi -t30m -d \
  -o 'results/full-rhs-second-inline-valid-%j.log' \
  bash -lc '... ./compile.sh; ./run.sh --twop-cache; ./check.sh ...'
```

145654 的最终输出为：

```text
Total Evolve Time: 341.535 s
Total Running Time: 344.645 s
This Program Cost = 345.68659257888794 s
Trajectory: matched times 100/100, effective terms 596
Trajectory RMS: 0 (0.000000%)
Constraint maxima (level 0): Ham=0.28974817, Px=0.039343259,
  Py=0.047298107, Pz=0.044686547
FINAL: PASS
```

所以总时长已经低于 370 s（余量约 24.313 s）。相对 145540 的真实 level-0 baseline，端到端少
87.725 s（约 20.2%）；相对最早 GPU baseline 的 1736.447 s 也不是只压了一个 kernel，而是把
RHS、boundary/AMR dispatch 和共享初值阶段一起收敛到可验收状态。

最后再核对一次提交范围：`AMSS_NCKU_Input.py` 保持课程的 GPU、网格/物理参数、
`Final_Evolution_Time=100.0` 与 `Evolution_Step_Number=10000000`；运行时只从中使用
`MPI_processes=1`、`OMP_threads=16`。改动只落在 `CMakeLists.txt`、`compile.sh`、`run.sh` 及
`src/`，分析日志在本目录，不作为 OJ 提交文件。没有保留 all-skip、LTO illegal access、
8-stream、过硬 launch-bounds 或任何只在 t=1 好看的试验。

### 20. 340 s 目标下的 launch 与编译器复测

前一版 345.687 s 离 340 s 只差约 1.65%，但这个区间已经和 MIG 波动同量级，所以这一轮仍坚持
先做相邻 t=1 重复、再看完整运行，不拿单次最低值外推。RHS 的
`__launch_bounds__(128,3)` 改成 `(128,4)` 后，两次短测为 3.13585/3.17956 s，看起来比原版
3.19655/3.24391 s 快；但完整运行前 24 步的平均 step 已到 3.50002 s，外推明显超过 350 s，
故主动停止并回退。`8x4x2` 的 64-thread RHS 同样更慢，首步为 3.34650 s。

RK batch 还试过两种等价整理。第一种把 `fix_boundary` 做成编译期模板，结果 3.18946/3.23737 s，
与原版完全重合；第二种把高层 `f1=f0` 的物理边界覆盖拆成紧凑六面 kernel，结果反而是
3.21906/3.27083 s。多一次 launch 抵消了少做体网格判定的收益，两项都撤回。

检查 `flags.make` 后确认 `AMSS_OPT=-Ofast -ffast-math` 只作用于 C++/Fortran，故又分别给 CUDA
试了 `-O3 --use_fast_math` 和 `--extra-device-vectorization`。前者是 3.19236/3.23958 s，后者是
3.19503/3.22121 s，均没有超过噪声；既没有速度证据，也没有理由引入额外浮点语义变化，最终
`CMakeLists.txt` 不保留它们。

### 21. `cudaMalloc/cudaFree` 热点与未通过的显存池

最新版 Nsys 的 CUDA API 汇总显示，t=1 有 2549 次 `cudaMalloc` 和 2541 次 `cudaFree`，两者合计
约 200 ms；另有 1044 次 `cudaDeviceSynchronize`。先去掉纯 sizing pass 的空同步、并让 RK
substep 依赖后续真实 pack 的同步，短测仍为 3.202--3.238 s，没有可分辨收益，故恢复原同步点。

随后恢复 `GPUManager` 中原本被注释掉的 exact-size pool，并把插值用的 `int` weight/error 临时区
按字节容量一起复用。短测确实降到 3.11956/3.15998 s，信号约为 2.4%，理论上足以越过 340 s。
但无限池在完整作业第 17 步 regrid 时出现：

```text
CUDA Error: out of memory at src/gpu_manager.cu:51
```

原因是旧 AMR generation 的数百个同尺寸 field array 全被保留。再把每个尺寸的空闲块限制为 48，
显存峰值不再立即增长；然而完整运行前 27 步平均 step 为 3.53881 s，外推约 354 s，仍达不到目标。
这说明短测收益主要来自分配状态和节点波动，不能覆盖完整 regrid 生命周期。该池、`int` 临时区复用、
本地 transfer cache 和延后同步最终全部撤回。

截至本轮，仍然只有 145654 的 **345.686593 s / FINAL: PASS** 是低于 370 s 且完整可信的最好结果；
尚未得到小于 340 s 的完整 `PASS`。这里明确记录缺口，而不把短测 3.11956 s 或被取消的完整作业
写成 120 分结果。

### 22. 220 s 目标：按 field 展开 advection

新的 Nsys 中，四个 interior group 与四个 boundary group 合计约占 GPU kernel 时间的 30%。旧实现
虽然已经把 24 个变量合为四次 launch，但代价是每个 thread 串行完成六套 stencil，并同时携带十二个
field/RHS 指针。对本题只有 40--48 个 x 点的 AMR 小块，这种写法进一步压低了可并行 block 数。

这次复用 `Block` 已有的两个 24-pointer device table，将 field 编号放进 `grid.y`：一个 thread 只处理
一个点的一个变量。interior 用去掉 3 层 ghost 后的紧凑一维点号；boundary 仍按原来的 face/edge/corner
枚举，并逐 field 保留完全相同的 SoA 奇偶性。新 interior/boundary kernel 分别是 44/68 registers，均无
local-memory spill。t=1 的 Nsys 对比为：

```text
old interior groups + boundary groups: 408.840 + 653.607 = 1062.447 ms
new interior field + boundary field:    340.887 + 369.405 =  710.292 ms
```

这里约 33.1% 的短测 kernel 降幅看起来很强，不过进入完整运行后稳定段反而约为 3.4--3.6 s/步，
没有保持 t=1 的优势，故最终 launcher 已恢复为四组 interior + 四组 boundary；field kernel 只作为本轮
实验代码保留，不在正式路径调用。

### 23. 主 RHS 的 region 与 block 反例

主 `rhs_kernel` 仍是第一热点。先尝试把 evolution 实例拆成纯 interior 和纯 boundary 两个模板，期待
编译器从 interior 实例中彻底裁掉 symmetry helper。实际 interior 391 次共 1073.975 ms，boundary
391 次又要 856.432 ms；两者加上 constraint 实例后 RHS 达 2018.483 ms，明显坏于单核约
1560 ms。边界点比例在这些浅 z 小块上并不低，多一次完整 kernel 也破坏了 cache，故撤回。

线程块也不能只按 x 合并访存判断。`32x2x2` 在完整运行稳定段为 3.3--3.5 s/步，`16x4x2` 的 t=1
RHS 更从约 1567 增至 1787 ms；y/z stencil 的局部性损失大于 x transaction 的收益，两项均恢复为
`8x8x2`。全 CUDA `--restrict` 使单步为 3.1898 s，和无该 flag 的 3.1786 s 重合，且全局别名承诺
风险过大，因此也不保留。

### 24. prolong register cap

`prolong3_batch_kernel` 原来使用 168 registers/thread。给它加 `__launch_bounds__(128,4)` 后，ptxas
将寄存器压到 128，Nsys 确认 local memory 仍为 0；同一 t=1 workload 从约 470.24 ms 降至
405.62 ms（13.7%）。这项与 advection field 展开叠加后，短测为：

```text
After Step: 1 ... 3.06519 s
Total Evolve Time: 3.17857 s
```

由于短测不能覆盖 regrid 生命周期，随后提交了 t=100。前 8 步已经稳定在 3.32--3.81 s/步，明显
不可能达到 220 s，也不优于 145654，故提前停止。`prolong3_batch` 的 cap 与 field-advection 一样只在
t=1 好看，最终恢复默认寄存器分配。到本轮结束仍没有得到小于 220 s 的完整 `PASS`；可信最好结果
仍为 145654 的 **345.686593 s / FINAL: PASS**。

### 25. 按 OJ 无缓存日志复测 MPI/OMP

OJ 的真实日志和本地 cache run 有两个重要差异：TwoPuncture cache 被关闭，并且最终计时包含 Python
绘图。该次结果是 `Total Evolve=337.226 s`、`Program Cost=369.962 s`、外层 wall
`379.006 s`，所以约 42 s 不在 evolution 内。为避免把 CPU 初值与 GPU 演化混在一起，本轮分别测量。

首先保持总 CPU=16、单 MIG，使用相同 build 和短多步 evolution：

```text
MPI=1, OMP=16: 约 3.2--3.4 s/step
MPI=2, OMP=8 : 约 5.4--5.6 s/step
MPI=4, OMP=4 : 约 7.46 s/step
```

代码确实会把 Block 轮转分到 MPI rank，但三个配置仍共享同一个 MIG device。评分节点没有让多个 CUDA
context 并发执行的 MPS 环境，结果是 context 争用再叠加 halo MPI 同步；rank 越多越慢。因此正式配置
继续使用 `MPI_processes=1`，不能套用 CPU strong-scaling 的直觉。

然后单独重复运行无缓存 `TwoPunctureABE`。固定 `OMP_PROC_BIND=close, OMP_PLACES=cores` 时：

```text
OMP threads:  1       2       4       8       16
wall (s):    23.762  15.998  11.703   9.542    9.204
```

16 线程下再比较 placement：`cores/close=8.632 s`、`threads/close=8.737 s`、
`cores/spread=8.569 s`；12-thread `cores/close=9.177 s`。最终采用 16 threads、cores、spread。
这只改善无缓存初值阶段，ABEGPU 本身没有 OpenMP 热区，所以不会谎称它能把 337 s evolution 降到
220 s。

OJ checker 使用四个数值输出而不是 PNG；原 Python 路径仍会打开二十多个 figure，并在日志中触发
Matplotlib figure warning。现在默认跳过绘图、保留全部 `.dat` 和 setting/error 文件；需要人工出图时
可显式设置 `AMSS_ENABLE_PLOTS=1`。两步 smoke test 得到 `Total Evolve=6.74984 s`、`Program
Cost=10.55155 s`，并正常留下数值结果。该改动减少的是评分 wall 的非计算尾部，不改变演化结果。

### 26. 回看 index：共享缓存与 separable prolongation 反例

重新逐项对照 `index.md` 后，MPI/OMP、kernel batching、RDC/LTO、register/occupancy、stream、显存池、
CUDA 编译选项和绘图尾部都已经实际测过。仍未真正完成的主要是三类：只读 stencil 的 shared-memory
复用、prolongation 的 producer-consumer/separable 变换，以及把主 RHS 按数学阶段而非空间 region
拆分。CUDA-aware MPI 在最终 `MPI=1` 配置下没有传输对象，analysis overlap 也只影响很低频的输出，
两者不是当前 220 s 缺口的主线。

先给六场一组的 interior advection 加 `16x4x2` shared tile，每个场协作搬入三层 halo，再复用 tile
计算 upwind 和 KO。tile 只有 14080 bytes，看起来容量不大，但每个场需要两次 block barrier，halo
比例也很高。160620 的 Nsys 结果是：

```text
原 interior advection groups: 约 399--440 ms
shared-tile groups          : 790.258 ms
单步 Total Evolve           : 3.75671 s
```

说明这个小块 stencil 原本已经由 L1 承接了大部分邻域复用，显式 shared 反而付出同步和驻留 block
下降的代价，故完整撤回。

prolongation 又试了一个更激进的算法变换。连续 `2x2x2` 八个 fine 点对应同一个 `6x6x6` coarse
立方体；让一个 thread 同时产生八点，可把理论 coarse loads 从 1728 降到 216，乘加也能复用 z/y
中间量。但 160651 中 `prolong3_batch` 从约 470 ms 升到 970.684 ms，单步 3.84847 s。原因是一个
thread 串行承担八点使可用并行度下降，八个输出 accumulator 还抬高了寄存器压力。这条实现也撤回；
若继续 separable，只能采用 warp/block 协作并把中间量放在跨线程存储中，不能再做 thread coarsening。

### 27. vacuum RHS 专门化

题目固定使用 vacuum BSSN，但旧主演化 kernel 仍然逐点读取 `rho`、`S_i` 和六个 `S_ij`，再把这些
恒为零的量代入 `Gamma^i`、`A_ij` 和 `trK` 的 RHS。接口仍保留这些数组以兼容原程序，只在 Lab 4
热 kernel 内将 matter source 编译为零。这样不是近似，也没有改方程阶数；它利用的是输入中已经固定
的 vacuum 条件。

160688 的相邻 Nsys 对比为：

```text
rhs_kernel before: 1563.209 ms
rhs_kernel vacuum: 1512.030 ms
prolong/advection/RK: 基本不变
```

主演化 RHS 降约 3.3%，折合整个短步约 1.5%。此外在 boundary group 入口跳过任一维最后一层的严格
零贡献点，只减少约 3 ms，说明 boundary 剩余时间主要是有效 stencil，而不是空函数调用。vacuum
版本随后提交 t=100。160700 的完整结果为：

```text
Total Evolve Time: 330.179 s
Total Running Time: 333.331 s
This Program Cost: 333.865606 s
Trajectory RMS: 0
Constraint maxima: Ham=0.28974817, Px=0.039343259,
                    Py=0.047298107, Pz=0.044686547
FINAL: PASS
```

相对上一份本地完整可信的 345.686593 s，端到端减少 11.821 s（3.42%），所以 vacuum 专门化和
最后平面早退保留。它仍离 220 s 差 113.866 s，不能写成 120 分。按现有 Nsys 分解，继续堆编译参数、
MPI rank 或 OpenMP thread 已经没有足够上限；下一步必须动主 RHS 的数学阶段分解，或实现保持并行度的
warp/block separable prolongation，并各自重新跑完整 checker。

### 28. warp/subgroup separable prolongation 大改

上一轮单线程八点失败后，这次真正实现了跨线程三阶段分解。每个 parity-aligned `2x2x2` octet 的
`6x6x6` coarse cube 先由协作线程完成 72 个 z reduction，再完成 24 个 y reduction，最后由 8 个
lane 各写一个 x reduction。完整 interior octet 因而只读取一次 216 个 coarse value，不再让八个
输出各自读取 216 次。

第一版用一个 32-lane warp 负责一个 octet，每个 128-thread block 只覆盖 32 个输出。161039 中
`prolong3_batch` 达到 3078.554 ms；主要问题不是 interior 公式，而是大量 symmetry/partial octet
进入旧函数 fallback 时只有 8 个 lane 工作。第二版改成 8-thread subgroup，每块并行 16 个 octet，
使 fallback 恢复每块 128 个输出。161049 的结果为：

```text
原 prolong3_batch:              约 470 ms
32-lane warp/octet:           3078.554 ms
8-lane subgroup/octet:         930.328 ms
subgroup 版本 Total Evolve:      3.75214 s
```

第二版已经排除了单纯 fallback 并行度不足，但仍约为原版两倍。这说明本题 transfer box 太浅，完整
interior octet 占比不足；为每个 octet 建立 shared 中间量和同步的成本无法被 `6^3` 复用摊薄。两版
均完整撤回，正式路径回到逐 fine point 的 batch kernel。

同时测试把主 RHS 的 `__launch_bounds__` 从每 SM 至少 3 blocks 放宽为 2 blocks，让编译器可用更多
寄存器，作为数学 phase 拆分前的占用率上限检查。161057 中 RHS 从 vacuum 版本约 1509 ms 增至
1551.012 ms，说明当前 3-block 配置不是因为 register cap 过紧而慢，已恢复 `(128,3)`。真正拆成
geometry/core 两核需要把 18 个 Christoffel 和 6 个 Ricci 中间场写回已有 scratch arrays；这会额外
产生每点 24 次全局写和 24 次全局读。在 warp separable 已显示小 patch 对中间存储十分敏感后，不能
在没有逐阶段正确性基准的情况下把这项高风险半成品留在正式路径。

### 29. perf：Ansorg 每点分配与初值插值并行

重新用 `perf stat -d` 和 `perf record -F 5` 覆盖 Python driver、缓存初值读取和一整个 GPU step。
162082 的主机计数为 IPC=2.14、L1 miss=1.33%、branch miss=0.55%，说明 CPU 并没有普遍的 cache 或
分支问题。符号采样却同时命中 `Ansorg::interpolate_tri_bar`、`xyz_to_ABp`、`malloc` 和
`_int_malloc`。源码确认每个网格点的六阶插值都执行 9 次 `new[]` 与 9 次 `delete[]`。

本题 `Read_Ansorg` 固定 `order=6`，因此把 `w/omega/v` 改成栈上的连续固定工作区，保留原 row pointer
和全部累加顺序。修改后的 perf 中 `_int_malloc` 已从热点表消失，`malloc@plt` 只剩约 0.9%；短跑
`Before Evolve` 从 3.499 s 降到 3.356 s。随后确认每个点只读同一个 Ansorg 谱表、只写自己的 cell，
在三重网格循环加入 `omp parallel for collapse(3) schedule(static)`，并让正式单 rank 配置使用 16 个
OpenMP threads。相邻短跑得到：

```text
                                  Before Evolve   Program Cost
原版本 162082                         3.499 s         8.077 s
栈工作区 162114                       3.356 s         8.038 s
栈工作区 + OMP16 162125               2.517 s         6.866 s
```

OMP16 的 perf page faults 也从约 59.4 万降到 37.1 万。完整 t=100 作业 162139 输出为：

```text
Before Evolve:       2.19154 s
Total Evolve Time: 327.114 s
Total Running Time:329.305 s
Program Cost:       329.927932 s
Trajectory RMS:      0
Constraint maxima: Ham=0.28974817, Px=0.039343259,
                    Py=0.047298107, Pz=0.044686547
FINAL: PASS
```

这刷新了本地完整最好结果。相对 vacuum 版本 333.865606 s，端到端再少 3.938 s；其中约 1 s 是
perf 直接定位并消除的初值阶段开销，其余差值属于整段 evolution 的 MIG 波动。

还尝试按离散 interpolation window 预计算三方向 barycentric omega。162201 的 `Before Evolve` 反而
回升到 2.959 s，插值函数在 perf 中由 13.37% 增到 16.30%。六点现算容易留在寄存器中，共享 cache
则增加随机窗口访问和多线程 cache 竞争，故该缓存已撤回。
