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
