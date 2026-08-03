# MoE 算子 RISC-V 适配记录

日期：2026-08-03

本文档记录 `student/moe_opt.cpp` 从 Intel Sapphire Rapids（AVX-512 + AMX）专用实现
适配到 RISC-V 平台的完整过程，包括 `exit 126` 问题的根因、代码改动、验证结果以及后续
RVV 优化路线。

## 1. 问题背景：exit 126

在 RISC-V 集群节点上运行 `1.sh` 时，日志（`results/20260803_072545.log`）出现两类错误：

```text
c++: error: '-march=sapphirerapids': ISA string must begin with rv32 or rv64
...
1.sh: 15: ./build/lab2: Exec format error
```

`Exec format error` 是 shell 在尝试执行一个**错误架构的可执行文件**时返回的错误，
对应退出码 126。它不是 `moe_opt.cpp` 运行时的崩溃，而是下面这条失败链的结果：

1. `build/CMakeCache.txt` 是在 x86_64 机器上生成的（源目录还是旧的 `lab2`，日志同时报出
   cache 目录不匹配），cache 里缓存了 `CMAKE_SYSTEM_PROCESSOR=x86_64` 和已经通过的
   `COMPILER_SUPPORTS_SPR`。
2. RISC-V 节点上 cmake 复用了这份 stale cache：`CMakeLists.txt` 的
   `-march=sapphirerapids` 分支被触发，RISC-V 的 g++ 拒绝该参数，`student` 目标编译失败。
3. 编译失败后脚本继续执行 `./build/lab2`，此时运行的是 build 目录里残留的
   x86-64 ELF 二进制，RISC-V 内核无法加载它，于是 shell 报 `Exec format error`，
   退出码 126。

结论：**exit 126 的根因是跨机器/跨架构复用了 build 目录**，不是学生代码的运行错误。

## 2. 代码改动

### 2.1 `student/moe_opt.cpp`：架构隔离 + RISC-V 可移植后端

原文件是无条件面向 x86 的：342 处 `_mm512*`、112 处 `_tile_*`（AMX）、
`SYS_arch_prctl`/`XFEATURE_XTILEDATA` 权限申请等，任何一处都无法在 RISC-V 上编译；
同时文件顶部无条件 `#include <riscv_vector.h>`，导致 x86 的 g++ 也编译不过
（当时文件处于"两头都编译不了"的半移植状态）。

改动方式：

- 在 [moe_opt.cpp:22](../student/moe_opt.cpp) 用
  `#if (defined(__x86_64__) || defined(__amd64__)) && !defined(MOE_FORCE_PORTABLE)`
  把整段 x86/AMX 实现（约 2040 行）包起来，**原实现一行未动**；删掉了无用的
  `#include <riscv_vector.h>`，x86 专属头文件（`immintrin.h`、`linux/futex.h`、
  `pthread.h`、`sched.h`、`sys/syscall.h`、`thread.h`、`unistd.h`）只在该分支内引入。
- 在 [moe_opt.cpp:2064](../student/moe_opt.cpp) 追加非 x86 分支（RISC-V 及其他 ISA
  共用）：
  - `riscv_expert_ffn`：单个 SwiGLU 专家前向，数学公式与参考实现
    `src/moe_ref.cpp` 完全一致（同样的 exp/sigmoid、同样的 `lrintf` 取整、同样的
    量化尺度），因此结果与 baseline 逐位相同（下面验证中 rel RMSE = 0）。
  - `riscv_forward_scalar`：按 token 并行（`#pragma omp parallel for`）完成
    router 打分 → Top-K 选择（bias 只参与选择、不进入 gate）→ int8 量化 →
    共享专家 + K 个路由专家 → 残差合并，与参考实现的每个 token 内部计算顺序一致。
  - `preprocess` 为空操作：标量后端直接使用框架给的行主序权重，无需重打包。
- 增加了编译期测试开关 `MOE_FORCE_PORTABLE`（见
  [moe_opt.cpp:23](../student/moe_opt.cpp)）：在任何机器上编译时定义它即可强制走
  非 x86 分支，用于在 x86 主机上验证 RISC-V 后端的正确性；正常构建不受影响。

### 2.2 `1.sh`：每次从干净 build 目录开始

[1.sh:16](../1.sh) 在 cmake 配置前执行：

```bash
cmake -E remove_directory build
cmake -B build
```

彻底消除 stale cache 和残留二进制带来的两类问题（错误 `-march` 参数、`Exec format
error`）。构建产物完全可再生成，删除无风险。

## 3. 验证结果

本机（x86_64，g++ 14.2）完成如下验证：

### 3.1 x86 分支（主任务路径）

从干净 build 目录重新 configure + 构建成功；以 1000 次迭代完整运行四个测试
（日志见 `results/20260803_073652.log`），全部通过：

| 测试 | 形状 | rel RMSE | 加速比 |
| ---- | ---- | -------- | ------ |
| 1 | N=1, D=256, H=128, E=16, K=4 | 7.9e-08 | 18.4x |
| 2 | N=1, D=1024, H=512, E=16, K=4 | 4.7e-08 | 9.5x |
| 3 | N=128, D=256, H=128, E=16, K=4 | 9.0e-08 | 98.1x |
| 4 | N=1024, D=512, H=128, E=512, K=2 | 2.5e-04 | 81.5x |

### 3.2 RISC-V 分支（本机强制编译 + 运行验证）

用 `-DMOE_FORCE_PORTABLE` 在 x86 上编译出非 x86 后端的可执行文件，确认产物中不含任何
x86 内建符号（`_mm512*`/`_tile_*`），四个测试全部通过且 **rel RMSE = 0**
（与参考实现逐位一致）：

| 测试 | 结果 |
| ---- | ---- |
| 1 | Result is correct! (rel RMSE: 0) |
| 2 | Result is correct! (rel RMSE: 0) |
| 3 | Result is correct! (rel RMSE: 0) |
| 4 | Result is correct! (rel RMSE: 0) |

Test 4（N=1024, E=512, K=2）单独运行（8 线程）：Baseline 11.60 s → Optimized
3.18 s，加速比 3.65x，说明 OpenMP 按 token 并行在 RISC-V 上同样生效。

## 4. 在 RISC-V 集群节点上运行

```bash
cd /path/to/lab2rv
bash 1.sh   # 或按集群方式提交脚本
```

### 4.1 walltime 超时的调整

首次在 RISC-V 节点提交后，作业报 `walltime exceeds`（作业运行超过 `#HPC` 请求的
walltime 被调度器终止）。根因不在学生代码：框架的参考实现循环每个测试默认跑
1000 次迭代，而 Test 4 的 baseline 单测在 x86 上就要约 230 s，RISC-V 节点只会更慢，
仅 baseline 就远超 5 分钟；该循环位于框架 `main.cpp` 中，评分时会被替换，不能修改。

解决办法：`main.cpp` 支持第 6 个参数 `n_iter`，因此 [1.sh](../1.sh) 给每个测试显式
传入较小的迭代次数（Test 1/2: 200，Test 3: 100，Test 4: 10）。这样：

- 四个测试的 `check_result` 正确性检查照常执行，形状与参考实现完全一致；
- `Baseline time` / `Optimized time` / `Speedup` 仍然正常输出（迭代次数少导致测时
  精度略降，但不影响正确性判定）；
- 本地实测整个 `1.sh`（含重新构建）约 30 s，RISC-V 节点即使慢 10~20 倍也远在
  walltime 限额内（当前 `1.sh` 请求 13 分钟）。

如需在 x86 上恢复完整基准（1000 次迭代），把对应 `./build/lab2 ...` 命令行末尾的
迭代次数去掉即可。

预期行为：

1. `cmake -E remove_directory build` 清掉跨架构残留，cmake 在 RISC-V 节点上重新
   configure。`CMakeLists.txt` 检测到 `CMAKE_SYSTEM_PROCESSOR=riscv64` 后走
   `-march=rv64gcv`（若编译器不支持该参数则自动跳过），只有 student 目标会带架构参数，
   framework 保持可移植编译。
2. `student/moe_opt.cpp` 走非 x86 分支（可移植标量 + OpenMP），无需
   `immintrin.h`/AMX/RVV 头文件即可编译。
3. 四个测试与参考实现比较，结果为逐位一致（rel RMSE = 0），不再出现
   `Exec format error` / exit 126。

> 说明：本机没有 RISC-V 交叉工具链和 qemu，无法直接产出/执行 riscv64 二进制；
> 以上通过在 x86 上强制编译非 x86 分支验证了代码的可编译性和数值正确性。

## 5. 后续：RVV 内核优化路线

当前 RISC-V 后端是可移植标量实现（正确性优先）。如需在 Muse Pi Pro 上取得真正的
向量化加速，可按以下顺序把各阶段替换为 RVV 内建实现（用
`#if defined(__riscv_v)` 保护，并在 `CMakeLists.txt` 的 `-march=rv64gcv` 基础上
启用）：

1. **Router 打分**：`vle32.v` 读权重与激活，`vfmacc.vv` 做 fp32 点积，
   `vfrsqrt7.v`/多项式做 sigmoid。
2. **量化**：`vle32.v` → `vfcvt.rtz.x.f.v` → `vnclip`（饱和截断到 int8），
   每 VLEN 处理 16 个元素，天然适配不同向量长度。
3. **专家 GEMM**：`vle8.v` + `vwmacc.vv`（int8×int8→int32 加宽乘加）替换 AMX
   的 `_tile_dpbssd`；按 token×d_ff 的 tile 分块，保持 VLEN 无关写法。
4. **SwiGLU / 隐藏层量化**：同上用 `vfmacc.vv` 与 `vnclip` 处理。
5. 若节点支持进迭时空 IME 矩阵扩展，可在 RVV 基础上叠加矩阵单元加速（对应
   `docs/index.md` 的 Bonus 说明）。

替换过程中每个阶段都可用 `moe_forward_ref` 结果做差分验证（本仓库 `check_result`
自带容差检查），且可以继续用 `-DMOE_FORCE_PORTABLE` 在 x86 上做语法/逻辑预检。
