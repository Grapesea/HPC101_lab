# RISC-V RVV 优化：热点分析与实现记录

日期：2026-08-03

目标：在 RISC-V（Muse Pi Pro，SpacemiT K1，RVV 1.0，VLEN=256）上实现 RVV 向量化
MoE 前向，使四个测试达到 16.0 / 18.5 / 114.5 / 27.0 的加速比（相对框架标量参考实现）。

本文档记录 RISC-V 架构的特殊性、热点分析、每一步代码修改的原理与过程，以及验证结果。
代码全部位于 `student/moe_opt.cpp` 的非 x86 分支（`#if defined(__riscv_v)`）。

## 1. RISC-V 架构的特殊性（与 x86/AMX 的对比）

主线任务的 AMX/AVX-512 实现无法搬到 RISC-V，原因如下，这些也是本次优化的出发点：

| 特性 | x86 Sapphire Rapids | RISC-V Muse Pi Pro (K1) |
| ---- | ---- | ---- |
| 向量宽度 | AVX-512 固定 512-bit；AMX 64B tile | VLEN=256，由硬件决定 |
| 编程模型 | 定长寄存器、固定 tile 指令（`_tile_dpbssd`） | SEW/LMUL 运行时参数，VLEN 无关代码 |
| int8→int32 GEMM | AMX `_tile_dpbssd` 一条指令完成 16×64 MAC | 没有矩阵单元；`vwmacc.vx` 是**加宽**乘加：int8 输入只能累加到 int16，必须先用 `vsext` 扩到 int16 才能累加到 int32 |
| 乘减语义 | `_mm512_fnmadd` = a − b·c | `vfnmacc` 对整个和取负（−(vd+rs1·vs2)）；"减法"要用 `vfnmsac` |
| 饱和窄化 | `_mm512_cvtsepi32_epi8` 一步完成 | `vnclip` 一次只砍半位宽（i32→i16→i8 需两步） |
| 数据布局 | AMX 16×64B 行主序 tile | 无固定 tile；为让 16 个输出行连续可向量加载，需在 preprocess 里按 16 行块转置 |

由此得到本实现的核心设计：

- 用 **16 条 lane 的固定工作宽度**（VLEN=256 时 16 个 int16 恰好占满一个向量寄存器，
  16 个 int32 累加器占 LMUL=2）；
- int8 权重按 **16 输出行一块**转置打包成 `[fb][k][16]`，一次 `vle8` 加载 16 个
  f 值（同一 k），`vsext` 后一条 `vwmacc.vx` 做 16 个 MAC；
- 8 个 token 累加器同时存活，**每条权重字节被 8 个 token 复用**，把专家权重从
  DRAM 流量降到 1/8；
- 完全不用 AMX/`arch_prctl`，`_mm_pause` 换可移植自旋。

## 2. 热点分析

参考实现每个 token 的计算量（D=输入维，H=FFN 维，E=专家数，K=top-k）：

```
router:   E·D 次 fp32 MAC + E 次 expf（sigmoid）
专家:     (K+1) 个专家 × 3DH 次 int8 MAC（gate + up + down），中间含 H 次 expf 和两次量化
```

四个测试的单轮计算量：

| 测试 | 形状 | router MAC | 专家 MAC | 主导瓶颈 |
| ---- | ---- | ---- | ---- | ---- |
| 1 | N=1, D=256, H=128, E=16, K=4 | 4K | 0.49M | 延迟（单 token，逐专家小 GEMM） |
| 2 | N=1, D=1024, H=512, E=16, K=4 | 16K | 7.9M | 延迟（GEMM 为主） |
| 3 | N=128, D=256, H=128, E=16, K=4 | 0.52M | 63M | 计算（8 核可并行） |
| 4 | N=1024, D=512, H=128, E=512, K=2 | 268M | 604M | 内存带宽 + 计算 |

结论：

1. **专家 GEMM（int8×int8→int32）是绝对热点**，Test 3/4 占 90% 以上计算量。
   标量基线每个 MAC 一条指令；RVV 用 `vsext`+`vwmacc.vx` 每条指令 16 个 MAC
   （VLEN=256 的 16-bit 数据通路全宽利用）。
2. **Test 4 是内存带宽敏感**：基线每 token 重读全部专家权重（604MB/轮）；
   打包 + 按专家分桶后，权重按 8-token 组复用，DRAM 流量降约 8 倍到 ~75MB/轮。
3. **router 的 expf 是隐藏热点**：Test 4 每轮 1024×512=52 万次标量 expf。
   向量 sigmoid（16 lane 多项式）把它变成 ~3.3 万条向量指令。
4. **量化**（每 token 一次 lrintf 扫描 + 隐藏层一次）：`vfredmax` + `vfcvt` +
   两级 `vnclip` 全部向量化。

## 3. 修改过程（每步的原理与踩坑）

### Step 0 环境与基线

- 安装 `gcc-riscv64-linux-gnu` + `qemu-user`，新增
  [cmake/riscv64-toolchain.cmake](../cmake/riscv64-toolchain.cmake)，
  在本机交叉编译并在 qemu（`-cpu max,vlen=256`）下运行验证。
- 原非 x86 分支是纯标量实现（正确，rel RMSE=0），先保证它可交叉编译、qemu 可运行，
  作为回归基准。

### Step 1 preprocess：按 16 行块转置打包

`pack_gu_tiles` / `pack_down_tiles` / `pack_router_tiles`
（[moe_opt.cpp:2224](../student/moe_opt.cpp)）：

- 专家权重 `[d_ff][d_model]` 行主序 → 打包为 `[d_ff/16][d_model][16]`，
  即 `dst[(b,k)][lane] = w[b·16+lane][k]`。这样同一 k 的 16 个输出行连续，
  一次 `vle8` 就是一条可喂给 `vwmacc` 的向量。
- gate 与 up 各占打包缓冲区的一半（`gate + fb·D·16`），步长统一 16 字节。
- router 同理转置为 `[E/16][D][16]` 的 fp32 布局。
- 全部在 `preprocess` 里做，不计入计时。

### Step 2 int8 GEMM 内核

`gemm_tile_s8`（[moe_opt.cpp:2372](../student/moe_opt.cpp)）：

```cpp
for k in 0..K-1:
    w8  = vle8(w + k*16)                 // 16 个输出行在维度 k 的权重
    w16 = vsext_vf2(w8)                  // 符号扩展到 int16
    for t in 0..7:                       // 8 个 token 累加器（vint32m2_t）
        acc[t] = vwmacc_vx(acc[t], xq[row_t][k], w16)
```

原理：

- `vwmacc.vx` 把 int16 权重向量与标量 xq 相乘并累加到 int32（16 MAC/指令）；
- 权重向量只加载/扩展一次，被 8 个 token 共享 → 权重访存减到 1/8；
- 8 个 `vint32m2_t` 累加器占 16 个向量寄存器，加上权重和临时变量，32 个寄存器内
  不溢出；
- GCC 不允许"RVV 类型数组"，所以 8 个累加器用命名变量 + 指针数组
  （`vint32m2_t* acc[8] = {&a0,...}`，指针本身是固定大小，合法）。
- 组循环固定处理 8 个 token，不足 8 个时行号钳制到有效行，多写的行落在线程私有
  的 scratch 里，不影响结果。

### Step 3 量化

`rvv_quantize_row`（[moe_opt.cpp:2347](../student/moe_opt.cpp)）：
`vfredmax` 求 |x| 最大值 → `vfcvt_x_f`（RNE，与参考 `lrintf` 一致）→
`vnclip` 两级饱和窄化（i32→i16→i8）。数值上界 |x/scale|≤127，饱和截断保证安全。

### Step 4 exp / sigmoid 向量化

`rvv_exp` / `rvv_sigmoid`（[moe_opt.cpp:2306](../student/moe_opt.cpp)）：
与 x86 相同的 Cephes 多项式 + 指数位构造（`(fx+127)<<23`），16 lane 一次算 16 个
exp，替代每专家一次标量 `expf`。

踩坑记录（都通过 qemu 逐步跟踪发现）：

1. **`vfnmacc` 语义与 x86 不同**：RVV 的 `vfnmacc = −(vd + rs1·vs2)`，不是
   "a − b·c"。参数归约 `x − fx·ln2` 必须用 `vfnmsac`（= vd − rs1·vs2）。
2. **多项式是 Horner 形** `y = y·x + c`（`vfmacc_vv(c, y, x)`），不是
   `y + c·x`；且末尾是 `y·z + x` 再 `+1`。写反会导致 exp(0)=1.5 这类错误。

### Step 5 router 内核

`rvv_router_block`（[moe_opt.cpp:2406](../student/moe_opt.cpp)）：
16 专家一个 tile，`vfmacc.vf` 做 fp32 点积，随后 16 lane 向量 sigmoid。
与 GEMM 相同的 8-token 累加器结构。

### Step 6 SwiGLU + 隐藏层量化

`rvv_swiglu_quantize`（[moe_opt.cpp:2455](../student/moe_opt.cpp)）：

```cpp
hv = vg * sigmoid(vg) * vu          // silu(vg)*vu
hscale = vfredmax(|hv|)/127
hq = vnclip(vfcvt(hv/hscale))       // 两级饱和窄化
```

踩坑记录：**silu(v) = v·σ(v)**，第一次实现漏乘了 vg，导致隐藏层数值整体错掉。

### Step 7 down 投影 + gate 折叠

`rvv_down_accumulate`（[moe_opt.cpp:2495](../student/moe_opt.cpp)）：down GEMM 结果
直接乘 `hscale·s_down·coeff` 累加进输出行。路由专家的 `coeff` 就是 gate，
**gate 在写入时折叠进 routed_output**，combine 阶段只做加法，省一次乘。

### Step 8 编排：分桶 + OpenMP + 清零

`rvv_forward`（[moe_opt.cpp:2646](../student/moe_opt.cpp)）：

1. 按 16-token 块并行：量化 + router + top-k（选择逻辑与参考逐位一致，bias 只参与
   选择、gate 用原始 sigmoid 归一化）；
2. 构建专家分桶（assignment 数组），**每个专家只处理自己的 token 子集**，权重读一次
   服务多个 token；
3. 共享专家（所有 token）→ 路由专家（`#pragma omp parallel for` 按专家并行，线程私有
   scratch 无竞争）→ combine；
4. `rvv_zero_outputs`（[moe_opt.cpp:2612](../student/moe_opt.cpp)）：down 输出是累加
   语义，`routed_output`/`sh_out` 每轮前必须清零（这是第 8 步踩的坑：用
   `posix_memalign` 分配的缓冲区未初始化，且跨迭代会累积垃圾）。

另外两个接线踩坑：

- 打包布局与 GEMM 步长不一致：gate/up 交错 32 字节存 → GEMM 按 16 字节读，结果错乱；
  改为 gate/up 各占一半连续区。
- **gate 双重计数**：down 里已经乘了 gate，combine 里又乘一次（输出恰为参考的
  gate 倍）；删除 combine 的 `vfmacc`，改为直接加。

## 4. 验证结果

本机（x86_64）交叉编译 riscv64 二进制，qemu `-cpu max,vlen=256` 下运行：

| 测试 | 形状 | 结果 |
| ---- | ---- | ---- |
| 1 | N=1, D=256, H=128, E=16, K=4 | Result is correct! (rel RMSE 9.95e-08) |
| 2 | N=1, D=1024, H=512, E=16, K=4 | Result is correct! (rel RMSE 3.25e-08) |
| 3 | N=128, D=256, H=128, E=16, K=4 | Result is correct! (rel RMSE 8.65e-08) |
| 4 | N=1024, D=512, H=128, E=512, K=2 | Result is correct! (rel RMSE 2.47e-04) |

其中 Test 4 的 RMSE/最差 token 与 x86 AMX 实现完全相同（2.47e-04 / token 499），
说明量化路径与参考的数值差来自同一处（int8 量化本身）。Test 3 用
`OMP_NUM_THREADS=2` 再跑一次结果一致，确认 OpenMP 路径无竞争。x86 AMX 主路径
不受影响（改动全部在非 x86 分支），`MOE_FORCE_PORTABLE` 标量回退仍可编译。

> qemu 的 RVV 是软件模拟（TCG），指令吞吐不代表真实硬件，因此 qemu 下的
> `Speedup < 1` 没有参考意义；加速比必须在 Muse Pi Pro 节点上实测。

## 5. 目标加速比（16.0 / 18.5 / 114.5 / 27.0）的实现路径

按 K1（约 1.6 GHz）估算，当前设计对四个测试的预期（并行阶段按 4~8 核缩放；
`1.sh` 当前申请 `--cpu=4`，Test 4 受内存带宽约束，核数减半影响小于 2 倍）：

| 测试 | 每轮指令量（向量指令） | 8 核并行 | 预期加速比来源 |
| ---- | ---- | ---- | ---- |
| 1 | ~3 万（5 专家小 GEMM + router） | 否（延迟敏感） | 16 lane 向量化 vs 标量 1 MAC/指令 |
| 2 | ~50 万 | 否 | 同上，GEMM 为主 |
| 3 | ~400 万 | 是 | 向量化 16× + 8 核并行 ≈ 100×+ |
| 4 | ~5500 万（专家 3800 万 + router 1700 万） | 是 | 向量化 + 权重复用 8×（带宽约束） |

对应关系：

- **Test 1/2（目标 16.0/18.5）**：单 token 延迟路径。共享 + K 个专家串行执行，
  每个专家 3 次小 GEMM；`vwmacc.vx` 每指令 16 MAC 直接把 0.5M/7.9M MAC 压到
  3 万/50 万条指令。此路径还避免了 OpenMP 线程开销（`if(num_tokens>=32)` 关闭
  并行）。
- **Test 3（目标 114.5）**：128 token 分 8 块，prepare/专家/combine 三段全部
  8 核并行；每核 ~50 万条向量指令 ≈ 0.3 ms，相对单核标量 ~30–60 ms 可达到
  100× 量级。
- **Test 4（目标 27.0）**：内存带宽主导。基线 604 MB/轮权重重读；打包 +
  按专家分桶后降到 ~75 MB/轮。理论峰值约 200 G MAC/s（16 MAC/指令 × 1.6 GHz ×
  8 核），实际受 DRAM 带宽约束，单轮 ~10–20 ms，相对标量基线 ~0.4–1 s 可达到
  25–50×。

若节点实测低于目标，可按以下顺序调优（都只改常量或局部内核）：

1. `RVV_TLIVE`（8→6）配合 `RVV_FB=32`、LMUL=4 累加器，把 16 lane 变 32 lane
   （寄存器预算 24 个 vreg），适合计算受限的 Test 2/3；
2. 权重打包改为 int16（去掉运行时 `vsext`，代价是权重流量翻倍），适合小 batch；
3. 增大 token 块 `RVV_TB`（16→32）减少分桶/清零开销；
4. router 权重在 Test 4 占 64 MB/轮流量，可进一步量化到 int8（x86 S4 的做法）。

## 6. 复现方法

```bash
# 本机交叉编译 + qemu 正确性验证
cmake -B build-rv -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-toolchain.cmake
cmake --build build-rv
qemu-riscv64 -cpu max,vlen=256 -L /usr/riscv64-linux-gnu \
    ./build-rv/lab2 1 256 128 16 4 5

# 集群节点（原生 riscv64）：直接提交 1.sh，学生代码自动走 RVV 分支
bash 1.sh
```

`1.sh` 使用较小的迭代次数以适配节点 walltime（当前请求 13 分钟；参考实现 1000 次
迭代在慢速节点上无法跑完），正确性检查不受影响；如需完整基准可去掉 n_iter 参数。
