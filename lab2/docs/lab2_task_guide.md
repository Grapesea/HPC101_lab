# Lab 2 任务与代码接口说明

本文根据 `index.md`、`include/moe.h`、`main.cpp` 和标量参考实现整理，目的是说明本实验需要完成的工作以及各个代码接口参数的含义，不包含 Lab 的具体实现。

## 1. 需要完成的任务

本实验的主任务是：在 CPU 上实现并优化一个 DeepSeek-V3 风格的量化 MoE（Mixture of Experts，混合专家）层前向计算。

只需要修改：

```text
student/moe_opt.cpp
```

并实现下面两个函数：

```cpp
void preprocess(MoEWeights& w);

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens);
```

评测目标有两个：

1. 优化版本的结果与标量参考实现足够接近。
2. 在保证正确性的前提下，前向计算尽可能快。

最终性能按照端到端加速比计算：

```text
Speedup = Baseline time / Optimized time
```

本实验不需要训练模型，不需要实现反向传播。主线任务面向 Intel Sapphire Rapids CPU；文档后面的 RISC-V 内容属于 Bonus。

评测时只收取 `student/` 目录，`main.cpp`、`include/`、`src/` 和 `CMakeLists.txt` 都会被替换为原版，因此不要依赖对这些文件的修改。

## MoE 中各个维度的含义

本实验使用以下符号：

| 数学符号 | 代码名称 | 含义 |
| --- | --- | --- |
| $N$ | `num_tokens` | 一个输入批次中的 token 数量 |
| $D$ | `d_model` | 每个 token 的输入和输出向量长度 |
| $H$ | `d_ff` | 每个专家内部 SwiGLU 的中间层长度 |
| $E$ | `num_experts` | 路由专家数量，不包括共享专家 |
| $K$ | `top_k` | 每个 token 选择的路由专家数量 |

除 $E$ 个路由专家外，模型还有一个共享专家。因此每个 token 实际执行：

```text
top_k 个路由专家 + 1 个共享专家
```

## 单个 token 的前向流程

对于一个输入 token：

$$
x_t\in\mathbb{R}^D
$$

需要依次完成下面六个步骤。

### 路由打分

对每个路由专家 $e$ 计算 FP32 点积：

$$
z_e=W_{\mathrm{router},e}\cdot x_t
$$

然后通过 sigmoid 得到原始亲和度：

$$
s_e=\operatorname{sigmoid}(z_e)=\frac{1}{1+e^{-z_e}}
$$

### 选择 Top-K 专家

选择专家时使用带偏置的分数：

$$
\mathrm{score}_e=s_e+\mathrm{bias}_e
$$

从 $E$ 个路由专家中选出分数最高的 $K$ 个。

`bias` 只决定选中哪些专家，不能参与后面的路由权重计算。如果分数相同，参考实现会优先选择编号更小的专家。

### 计算路由权重

只对选中专家的原始亲和度 $s_e$ 进行归一化：

$$
g_e=\frac{s_e}{\sum_{j\in\mathrm{TopK}}s_j}
$$

这里不能使用 $s_e+\mathrm{bias}_e$。所有选中专家的 $g_e$ 之和为 1。

### 将输入 token 量化为 INT8

每个 token 单独计算量化 scale：

$$
s_x=\frac{\max_d|x_d|}{127}
$$

然后量化：

$$
x_q=\operatorname{round}(x/s_x)
$$

这种方式称为 per-token 量化。如果整个 token 都是 0，参考实现令 `s_x = 1.0f`，以避免除零。

### 执行 SwiGLU 专家

共享专家和路由专家的内部结构相同。每个专家依次完成 gate projection、up projection、SwiGLU 激活、隐藏层重量化和 down projection。

先进行两个 INT8 矩阵向量乘：

$$
h_g=(W_gx_q)s_xs_g
$$

$$
h_u=(W_ux_q)s_xs_u
$$

INT8 与 INT8 相乘，结果必须在 INT32 中累加，然后乘量化 scale 反量化回 FP32。

接着计算 SwiGLU：

$$
h=\operatorname{SiLU}(h_g)\odot h_u
$$

其中：

$$
\operatorname{SiLU}(v)=\frac{v}{1+e^{-v}}
$$

$\odot$ 表示逐元素相乘。

随后将中间结果 $h$ 再量化为 INT8：

$$
s_h=\frac{\max_f|h_f|}{127}
$$

$$
h_q=\operatorname{round}(h/s_h)
$$

如果整个隐藏向量为 0，参考实现令 `s_h = 1.0f`。

最后执行 down projection：

$$
o=(W_dh_q)s_hs_d
$$

专家的输入长度为 $D$，中间层长度为 $H$，最终输出长度重新变为 $D$。

### 合并输出

最终输出为：

$$
y_t=x_t+o_{\mathrm{shared}}+
\sum_{e\in\mathrm{TopK}}g_eo_e
$$

其中：

- $x_t$ 是残差连接。
- 共享专家对所有 token 都执行。
- 路由专家只执行被选中的 $K$ 个。
- 路由专家的输出需要乘对应的路由权重 $g_e$。
- 共享专家的输出不乘路由权重。

## 4. `MoEWeights` 结构体参数

结构体定义在 `include/moe.h` 中。

### 4.1 模型形状

```cpp
int d_model;
```

输入和输出 token 的维度 $D$。

```cpp
int d_ff;
```

每个专家 SwiGLU 中间层的维度 $H$。

```cpp
int num_experts;
```

路由专家数量 $E$，不包括共享专家。

```cpp
int top_k;
```

每个 token 需要选择的路由专家数量 $K$。

这些维度都是运行时参数，不能假设程序永远只运行某一种固定形状。

### 4.2 Router 参数

```cpp
float* w_router;  // [num_experts][d_model]
```

FP32 路由权重，采用行优先连续存储。专家 `e`、输入维度 `d` 对应的元素为：

```cpp
w_router[e * d_model + d]
```

每一行与一个 token 做点积，得到一个专家的路由 logit。

```cpp
float* bias;  // [num_experts]
```

每个路由专家的负载均衡偏置。它只用于 Top-K 选择：

```cpp
score = affinity[e] + bias[e];
```

不能把这个偏置加入最终 gate 权重。

### 4.3 路由专家权重

```cpp
int8_t* w_gate;  // [num_experts][d_ff][d_model]
int8_t* w_up;    // [num_experts][d_ff][d_model]
```

分别表示 SwiGLU 的 gate projection 和 up projection。

元素索引为：

```cpp
w_gate[(e * d_ff + f) * d_model + d]
w_up[(e * d_ff + f) * d_model + d]
```

其中：

- `e` 是专家编号。
- `f` 是中间层编号，范围为 `[0, d_ff)`。
- `d` 是输入维度编号，范围为 `[0, d_model)`。

```cpp
int8_t* w_down;  // [num_experts][d_model][d_ff]
```

表示 down projection，将长度为 `d_ff` 的隐藏向量投影回长度为 `d_model` 的输出。

元素索引为：

```cpp
w_down[(e * d_model + d) * d_ff + f]
```

```cpp
float* s_gate;  // [num_experts]
float* s_up;    // [num_experts]
float* s_down;  // [num_experts]
```

每个专家的三个权重矩阵分别对应一个 FP32 反量化 scale。例如：

```cpp
weight_fp32 = weight_int8 * s_gate[e];
```

这里是每个矩阵一个 scale，而不是每行一个 scale。

### 4.4 共享专家参数

```cpp
int8_t* sh_gate;  // [d_ff][d_model]
int8_t* sh_up;    // [d_ff][d_model]
int8_t* sh_down;  // [d_model][d_ff]
```

共享专家与一个路由专家的形状完全相同，但没有专家编号这一维。

```cpp
float sh_s_gate;
float sh_s_up;
float sh_s_down;
```

分别是共享专家三个 INT8 权重矩阵的反量化 scale。

## 5. 待实现的代码接口

### 5.1 `preprocess`

```cpp
void preprocess(MoEWeights& w);
```

该函数在性能计时开始前调用一次，可以用于：

- 转置或重排权重。
- 将权重打包成 AVX-512、VNNI 或 AMX 适合读取的布局。
- 预先生成辅助索引或其他只依赖权重的数据。
- 为优化版本分配和准备持久化缓冲区。

`MoEWeights& w` 是非常量引用，因此语法上允许修改结构体及其指针。不过需要特别注意：验证阶段会在 `preprocess` 之后再次调用参考实现 `moe_forward_ref(x, w, ...)`。

因此，不应直接破坏 `w` 中原始权重的语义或原始布局。比较稳妥的思路是保留原始权重，在 `student/moe_opt.cpp` 自己管理的缓冲区中存放打包后的副本。

### 5.2 `moe_forward_optimized`

```cpp
void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens);
```

#### `const float* x`

输入 token 数组，逻辑形状为：

```text
[num_tokens][d_model]
```

token `t` 的起始地址为：

```cpp
const float* xt = x + (size_t)t * w.d_model;
```

`const` 表示函数不能通过该指针修改输入数据。

#### `const MoEWeights& w`

模型权重和运行时形状参数。它包含：

- `d_model`、`d_ff`、`num_experts`、`top_k`。
- Router 的 FP32 权重和 bias。
- 路由专家的 INT8 权重及 scale。
- 共享专家的 INT8 权重及 scale。

这里是常量引用，前向计算期间不应修改该结构体或其中的权重。

#### `float* y`

输出缓冲区，逻辑形状为：

```text
[num_tokens][d_model]
```

token `t` 的输出起始地址为：

```cpp
float* yt = y + (size_t)t * w.d_model;
```

函数需要完整写入输出，不能依赖 `y` 原来保存的内容。

#### `int num_tokens`

本次调用中的 token 数量 $N$。该值没有存储在 `MoEWeights` 中，因此必须从这个函数参数读取。

## 6. 框架中其他接口

这些函数由框架实现，一般不需要修改。

### 6.1 `init_data`

```cpp
void init_data(float* x, MoEWeights& w, int num_tokens, uint64_t seed);
```

作用是根据固定随机种子初始化模型权重和第一批 token。

- `x`：需要写入的输入数组。
- `w`：需要初始化的权重结构体，其形状字段必须提前设置。
- `num_tokens`：输入 token 数量。
- `seed`：随机种子，相同参数与种子会生成可复现的数据。

### 6.2 `init_tokens`

```cpp
void init_tokens(float* x, int num_tokens, int d_model, uint64_t seed);
```

只生成一批新的输入 token，不修改模型权重。

### 6.3 `moe_forward_ref`

```cpp
void moe_forward_ref(const float* x, const MoEWeights& w, float* y,
                     int num_tokens);
```

标量参考实现，是正确性和基础性能的基准。它清楚展示了实验要求的精确计算流程，但不是最终需要提交的优化实现。

### 6.4 `check_result`

```cpp
void check_result(const float* y, const float* y_ref, int rows, int cols);
```

比较优化输出和参考输出：

- `y`：优化实现的输出。
- `y_ref`：参考实现的输出。
- `rows`：token 数，即 `num_tokens`。
- `cols`：每个 token 的输出长度，即 `d_model`。

## 7. 命令行参数

程序运行格式为：

```bash
./build/lab2 <num_tokens> <d_model> <d_ff> <num_experts> <top_k> [n_iter]
```

例如：

```bash
./build/lab2 128 256 128 16 4
```

表示：

- `num_tokens = 128`
- `d_model = 256`
- `d_ff = 128`
- `num_experts = 16`
- `top_k = 4`
- 未提供 `n_iter`，默认运行 1000 次计时迭代

可选参数：

```text
n_iter
```

表示计时阶段重复调用前向函数的次数。它只影响性能测量时长，不改变一次前向计算的数学结果。

## 8. 输入范围

框架限制如下：

| 参数 | 范围或要求 |
| --- | --- |
| `num_tokens` | 1 到 1024 |
| `d_model` | 64 到 1024，并且是 64 的倍数 |
| `d_ff` | 64 到 512，并且是 64 的倍数 |
| `num_experts` | 1 到 512 |
| `top_k` | 1 到 4，并且不能大于 `num_experts` |
| `n_iter` | 至少为 1 |

这些上界也在 `include/moe.h` 中定义：

```cpp
#define MAX_NUM_TOKENS 1024
#define MAX_D_MODEL 1024
#define MAX_D_FF 512
#define MAX_NUM_EXPERTS 512
#define MAX_TOP_K 4
```

## 9. 正式评测场景

评测包含四种形状：

| 场景 | $N$ | $D$ | $H$ | $E$ | $K$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 | 1 | 256 | 128 | 16 | 4 |
| S2 | 1 | 1024 | 512 | 16 | 4 |
| S3 | 128 | 256 | 128 | 16 | 4 |
| S4 | 1024 | 512 | 128 | 512 | 2 |

这四个场景体现了不同优化需求：

- S1、S2 只有一个 token，不能依赖 token 之间的并行。
- S3 有较大的 batch，可以考虑 token 级或专家级数据复用。
- S4 的 token 和专家数量都很大，但每个 token 只选择两个路由专家，需要关注路由分布、负载均衡和权重访问。

实现必须从运行时参数读取维度，不能只针对其中一个场景写死。

## 10. 正确性标准

INT8 矩阵乘本身是精确整数运算，但向量化可能改变 FP32 加法顺序、SiLU 结果或舍入边界。因此评测不是要求逐元素完全一致，而是检查：

- 每个 token 的相对 L2 误差小于 `2e-2`。
- 全局相对 RMSE 小于 `2e-3`。

隐藏层的一个 FP32 微小误差可能使：

```text
round(h / s_h)
```

中的某个 INT8 值相差 1，所以少量舍入差异是允许的。但错误的专家选择、错误使用 bias、错误的矩阵布局或遗漏 scale 通常会明显超过容差。

## 11. 文档建议考虑的优化方向

本实验要求的是优化后的完整前向实现，而不仅是把参考实现复制到学生文件。`index.md` 给出的主要方向包括：

- 使用 AVX-512 或 VNNI 加速 INT8 点积。
- 在 Sapphire Rapids 上使用 AMX 执行矩阵乘法。
- 在 `preprocess` 中预打包权重，改善向量指令和 AMX 的访问布局。
- 将 token 按专家聚集，避免同一个专家权重被逐 token 重复从较慢的存储层加载。
- 提高同一权重块在缓存或寄存器中的复用率。
- 根据 batch 大小选择 token、expert 或投影行/Tile 作为并行粒度。
- 使用性能分析工具确认热点、缓存瓶颈、线程负载不均和同步开销。

这些只是优化方向，不改变前面定义的数学结果和接口语义。

## 12. 阅读代码的推荐顺序

建议按照下面的顺序理解代码：

1. 阅读 `include/moe.h`，认识维度、权重布局和接口。
2. 阅读 `src/moe_ref.cpp`，理解单个 token 的完整计算顺序。
3. 阅读 `main.cpp`，理解数据如何分配、何时计时、何时验证。
4. 阅读 `src/data.cpp`，理解随机数据和正确性容差。
5. 最后阅读并实现 `student/moe_opt.cpp`。

其中 `src/moe_ref.cpp` 是理解正确计算语义最直接的依据；`index.md` 则主要解释 MoE 原理、量化方式和可采用的体系结构优化技术。
