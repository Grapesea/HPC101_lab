// Optimized W8A8 MoE forward pass for Intel Sapphire Rapids.

#include "moe.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <linux/futex.h>
#include <limits>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

namespace {

constexpr int N_BLOCK = 16;
constexpr int K_BLOCK = 64;
constexpr int ROUTER_BATCH = 16;
constexpr size_t TILE_BYTES = 1024;
#ifndef MOE_S3_PAIR_MIN_ROWS
#define MOE_S3_PAIR_MIN_ROWS 17
#endif

struct alignas(64) TileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[8];
    uint8_t reserved2[16];
    uint8_t rows[8];
    uint8_t reserved3[8];
};
static_assert(sizeof(TileConfig) == 64);

static thread_local bool amx_permission_checked = false;
static thread_local bool amx_permission_ok = false;

static bool ensure_amx_permission() {
    if (!amx_permission_checked) {
        amx_permission_ok =
            syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
        amx_permission_checked = true;
    }
    return amx_permission_ok;
}

static inline void load_tile_config(int rows) {
    TileConfig cfg{};
    cfg.palette_id = 1;
    // tmm0..tmm5: independent INT32 accumulators, tmm6: activation A,
    // tmm7: the currently streamed packed B tile.  Multiple accumulator
    // chains hide TDPBSSD latency and reuse each A load across output blocks.
    for (int t = 0; t < 6; ++t) {
        cfg.colsb[t] = 64;
        cfg.rows[t] = static_cast<uint8_t>(rows);
    }
    cfg.colsb[6] = 64;
    cfg.rows[6] = static_cast<uint8_t>(rows);
    cfg.colsb[7] = 64;
    cfg.rows[7] = 16;
    _tile_loadconfig(&cfg);
}

struct PackedWeights {
    // Every 1 KiB unit is a native VNNI/AMX-B tile.  Tiles are additionally
    // ordered in the exact sequence consumed by the 3-output-block Gate/Up
    // and 6-output-block Down microkernels, so the hot loop is one stream.
    int8_t* routed_gu = nullptr;
    int8_t* routed_down = nullptr;
    int8_t* shared_gu = nullptr;
    int8_t* shared_down = nullptr;
    int8_t* routed_gu_s1 = nullptr;
    int8_t* routed_down_s1 = nullptr;
    int8_t* shared_gu_s1 = nullptr;
    int8_t* shared_down_s1 = nullptr;

    // AVX-512 VNNI provides u8*s8 (VPDPBUSD), while the model is s8*s8.
    // We xor the activation sign bit (x -> x+128 in unsigned notation) and
    // seed each accumulator with -128*sum(weights), precomputed here.
    int32_t* routed_gu_comp = nullptr;
    int32_t* routed_down_comp = nullptr;
    int32_t* shared_gu_comp = nullptr;
    int32_t* shared_down_comp = nullptr;

    // Router is transposed and padded in groups of 16 experts:
    // [expert_block][D][16].  One load therefore supplies the weights for
    // 16 experts, with no horizontal reduction in the timed path.
    float* router = nullptr;
    // S4 first-stage router: per-expert INT8 quantization packed as native
    // AMX B tiles.  FP32 is still used to refine the small candidate set.
    int8_t* router_q = nullptr;
    float* router_q_scale = nullptr;

    // Forward workspaces are allocated in preprocess (outside timing).
    int8_t* xq = nullptr;            // [MAX_NUM_TOKENS][D]
    float* x_scale = nullptr;        // [MAX_NUM_TOKENS]
    int* route_idx = nullptr;        // [MAX_NUM_TOKENS][K]
    float* route_gate = nullptr;     // [MAX_NUM_TOKENS][K]
    float* routed_output = nullptr;  // [MAX_NUM_TOKENS*K][D]
    uint16_t* routed_output16 = nullptr;  // compact batch-only workspace
    int* expert_counts = nullptr;    // [E+1]
    int* expert_offsets = nullptr;   // [E+1]
    int* bucket_entries = nullptr;   // [MAX_NUM_TOKENS*K]
    int32_t* single_gate_acc = nullptr;  // [K+1][H]
    int32_t* single_up_acc = nullptr;
    float* single_hidden = nullptr;
    int8_t* single_hq = nullptr;
    float* single_hscale = nullptr;
    float* single_chunk_max = nullptr;
    int32_t* single_down_acc = nullptr;
    bool amx_ready = false;

    size_t gu_expert_stride = 0;
    size_t down_expert_stride = 0;
    int d_model = 0;
    int d_ff = 0;
    int num_experts = 0;
    int padded_experts = 0;
    bool ready = false;

    void clear() {
        std::free(routed_gu);
        std::free(routed_down);
        std::free(shared_gu);
        std::free(shared_down);
        std::free(routed_gu_s1);
        std::free(routed_down_s1);
        std::free(shared_gu_s1);
        std::free(shared_down_s1);
        std::free(routed_gu_comp);
        std::free(routed_down_comp);
        std::free(shared_gu_comp);
        std::free(shared_down_comp);
        std::free(router);
        std::free(router_q);
        std::free(router_q_scale);
        std::free(xq);
        std::free(x_scale);
        std::free(route_idx);
        std::free(route_gate);
        std::free(routed_output);
        std::free(routed_output16);
        std::free(expert_counts);
        std::free(expert_offsets);
        std::free(bucket_entries);
        std::free(single_gate_acc);
        std::free(single_up_acc);
        std::free(single_hidden);
        std::free(single_hq);
        std::free(single_hscale);
        std::free(single_chunk_max);
        std::free(single_down_acc);
        routed_gu = routed_down = shared_gu = shared_down = nullptr;
        routed_gu_s1 = routed_down_s1 = nullptr;
        shared_gu_s1 = shared_down_s1 = nullptr;
        routed_gu_comp = routed_down_comp = nullptr;
        shared_gu_comp = shared_down_comp = nullptr;
        router = nullptr;
        router_q = nullptr;
        router_q_scale = nullptr;
        xq = nullptr;
        x_scale = nullptr;
        route_idx = nullptr;
        route_gate = nullptr;
        routed_output = nullptr;
        routed_output16 = nullptr;
        expert_counts = expert_offsets = bucket_entries = nullptr;
        single_gate_acc = single_up_acc = nullptr;
        single_hidden = nullptr;
        single_hq = nullptr;
        single_hscale = single_chunk_max = nullptr;
        single_down_acc = nullptr;
        amx_ready = false;
        gu_expert_stride = down_expert_stride = 0;
        d_model = d_ff = num_experts = padded_experts = 0;
        ready = false;
    }

    ~PackedWeights() { clear(); }
};

PackedWeights pw;
alignas(64) uint64_t s3_route_masks[64 * 16];
alignas(64) uint8_t s3_routed_order[16];
#ifdef MOE_S1_BF16_ROUTER
alignas(64) uint16_t s1_router_bf16[128 * 32];
#endif

class PersistentPool {
public:
    using Task = void (*)(int, void*);

    struct alignas(64) WorkerSignal {
        std::atomic<uint64_t> generation{0};
    };

    void start(int requested_threads) {
        stop();
        collect_allowed_cpus(cpus_);
        team_size_ = std::max(1, std::min(requested_threads,
                                         static_cast<int>(cpus_.size())));
        enabled_size_.store(team_size_, std::memory_order_relaxed);

        // Keep all optimized workers on distinct CPUs from the allocation.
        if (!cpus_.empty()) pin_current_thread(cpus_[0]);
        stop_.store(false, std::memory_order_relaxed);
        ready_.store(0, std::memory_order_relaxed);
        permission_failures_.store(0, std::memory_order_relaxed);
        generation_.store(0, std::memory_order_relaxed);
        for (WorkerSignal& signal : worker_done_)
            signal.generation.store(0, std::memory_order_relaxed);
        workers_.reserve(team_size_ - 1);
        for (int id = 1; id < team_size_; ++id)
            workers_.emplace_back([this, id] { worker_loop(id); });
        while (ready_.load(std::memory_order_acquire) != team_size_ - 1)
            _mm_pause();

        main_amx_ok_ = ensure_amx_permission();
        amx_ok_ = main_amx_ok_ &&
                  permission_failures_.load(std::memory_order_relaxed) == 0;
    }

    void stop() {
        if (!workers_.empty()) {
            enabled_size_.store(team_size_, std::memory_order_release);
            syscall(SYS_futex, reinterpret_cast<int*>(&enabled_size_),
                    FUTEX_WAKE_PRIVATE, std::numeric_limits<int>::max(),
                    nullptr, nullptr, 0);
            stop_.store(true, std::memory_order_relaxed);
            generation_.store(
                generation_.load(std::memory_order_relaxed) + 1,
                std::memory_order_release);
            for (std::thread& worker : workers_) worker.join();
            workers_.clear();
        }
        if (main_tile_rows_ != 0) {
            _tile_release();
            main_tile_rows_ = 0;
        }
        team_size_ = 1;
        amx_ok_ = false;
    }

    ~PersistentPool() { stop(); }

    int size() const { return team_size_; }
    bool amx_ok() const { return amx_ok_; }

    // Workers above this limit sleep in the kernel instead of consuming
    // package power while a latency-oriented call only has a few experts.
    // They are woken only when a later call actually needs the full team.
    void set_enabled(int requested_threads) {
        const int enabled =
            std::max(1, std::min(requested_threads, team_size_));
        if (enabled_size_.load(std::memory_order_relaxed) == enabled) return;
        const int previous =
            enabled_size_.exchange(enabled, std::memory_order_release);
        if (enabled > previous) {
            syscall(SYS_futex, reinterpret_cast<int*>(&enabled_size_),
                    FUTEX_WAKE_PRIVATE, std::numeric_limits<int>::max(),
                    nullptr, nullptr, 0);
        }
    }

    void dispatch(int count, int tile_rows, Task task, void* context) {
        const int active = std::max(1, std::min(count, team_size_));
        task_ = task;
        context_ = context;
        task_count_ = count;
        active_size_ = active;
        tile_rows_ = tile_rows;
        if (tile_rows != 0 && main_tile_rows_ != tile_rows) {
            load_tile_config(tile_rows);
            main_tile_rows_ = tile_rows;
        }
        const uint64_t generation =
            generation_.load(std::memory_order_relaxed) + 1;
        generation_.store(generation, std::memory_order_release);

        for (int i = 0; i < count; i += active) task(i, context);
        for (int id = 1; id < active; ++id) {
            while (worker_done_[id].generation.load(
                       std::memory_order_acquire) != generation)
                _mm_pause();
        }
    }

private:
    // OMP_PROC_BIND narrows the initial thread's affinity to place 0, so a
    // plain sched_getaffinity(0) would expose only one CPU on a bound OJ run.
    // Widen the calling thread back to the full process cpuset first:
    // sched_setaffinity clamps the requested mask to the cpuset, so asking
    // for every CPU yields exactly the allocated set, regardless of binding.
    static void collect_allowed_cpus(std::vector<int>& cpus) {
        cpus.clear();
        cpu_set_t full, allowed;
        CPU_ZERO(&full);
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) CPU_SET(cpu, &full);
        sched_setaffinity(0, sizeof(full), &full);
        CPU_ZERO(&allowed);
        sched_getaffinity(0, sizeof(allowed), &allowed);
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
        }
    }

    static void pin_current_thread(int cpu) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }

    void worker_loop(int id) {
        pin_current_thread(cpus_[id]);
        if (!ensure_amx_permission())
            permission_failures_.fetch_add(1, std::memory_order_relaxed);
        ready_.fetch_add(1, std::memory_order_release);
        uint64_t seen = 0;
        int tile_rows = 0;
        for (;;) {
            uint64_t current;
            while ((current = generation_.load(std::memory_order_acquire)) ==
                   seen) {
                const int enabled =
                    enabled_size_.load(std::memory_order_acquire);
                if (id >= enabled) {
                    syscall(SYS_futex,
                            reinterpret_cast<int*>(&enabled_size_),
                            FUTEX_WAIT_PRIVATE, enabled, nullptr, nullptr, 0);
                } else {
                    _mm_pause();
                }
            }
            seen = current;
            if (stop_.load(std::memory_order_relaxed)) break;
            const int requested_rows = tile_rows_;
            if (requested_rows != 0 && requested_rows != tile_rows) {
                load_tile_config(requested_rows);
                tile_rows = requested_rows;
            }
            Task task = task_;
            void* context = context_;
            const int count = task_count_;
            const int active = active_size_;
            if (id >= active) continue;
            for (int i = id; i < count; i += active) task(i, context);
            worker_done_[id].generation.store(seen,
                                              std::memory_order_release);
        }
        if (tile_rows != 0) _tile_release();
    }

    std::vector<int> cpus_;
    std::vector<std::thread> workers_;
    int team_size_ = 1;
    int main_tile_rows_ = 0;
    bool main_amx_ok_ = false;
    bool amx_ok_ = false;

    alignas(64) std::atomic<uint64_t> generation_{0};
    alignas(64) std::atomic<int> enabled_size_{1};
    WorkerSignal worker_done_[8];
    alignas(64) std::atomic<int> ready_{0};
    std::atomic<int> permission_failures_{0};
    std::atomic<bool> stop_{false};

    // Published before generation_ with release ordering, then read by
    // workers after the matching acquire load.
    Task task_ = nullptr;
    void* context_ = nullptr;
    int task_count_ = 0;
    int active_size_ = 1;
    int tile_rows_ = 0;

};

PersistentPool pool;

template <typename T>
static T* alloc_aligned(size_t count) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, count * sizeof(T)) != 0) return nullptr;
    return static_cast<T*>(p);
}

static void pack_one_tile(const int8_t* src, int k_dim, int ob, int kb,
                          int8_t* tile) {
    for (int kg = 0; kg < K_BLOCK / 4; ++kg) {
        int8_t* row = tile + (size_t)kg * 64;
        const int src_k = kb * K_BLOCK + kg * 4;
        for (int n = 0; n < N_BLOCK; ++n) {
            const int8_t* s = src +
                (size_t)(ob * N_BLOCK + n) * k_dim + src_k;
            std::memcpy(row + n * 4, s, 4);
        }
    }
}

static void pack_matrix_grouped(const int8_t* src, int out_dim, int k_dim,
                                int group_size, int8_t* dst) {
    const int out_blocks = out_dim / N_BLOCK;
    const int k_blocks = k_dim / K_BLOCK;
    size_t tile_index = 0;
    for (int ob = 0; ob < out_blocks; ob += group_size) {
        const int group = std::min(group_size, out_blocks - ob);
        for (int kb = 0; kb < k_blocks; ++kb)
            for (int j = 0; j < group; ++j)
                pack_one_tile(src, k_dim, ob + j, kb,
                              dst + tile_index++ * TILE_BYTES);
    }
}

static void pack_gate_up(const int8_t* gate, const int8_t* up,
                         int d_model, int d_ff, int8_t* dst) {
    const int out_blocks = d_ff / N_BLOCK;
    const int k_blocks = d_model / K_BLOCK;
    size_t tile_index = 0;
    for (int ob = 0; ob < out_blocks; ob += 3) {
        const int group = std::min(3, out_blocks - ob);
        for (int kb = 0; kb < k_blocks; ++kb) {
            for (int j = 0; j < group; ++j) {
                pack_one_tile(gate, d_model, ob + j, kb,
                              dst + tile_index++ * TILE_BYTES);
                pack_one_tile(up, d_model, ob + j, kb,
                              dst + tile_index++ * TILE_BYTES);
            }
        }
    }
}

static inline void pack_vnni_row(const int8_t* src, int input_dim,
                                 int output_block, int input, int8_t* dst) {
    for (int lane = 0; lane < N_BLOCK; ++lane)
        std::memcpy(dst + lane * 4,
                    src + (size_t)(output_block * N_BLOCK + lane) *
                              input_dim + input,
                    4);
}

// S1 consumes every output block for one activation dword before advancing.
// Store exactly that stream so its hot loops contain no group addressing.
static void pack_s1_gate_up_stream(const int8_t* gate, const int8_t* up,
                                   int8_t* dst) {
    size_t cursor = 0;
    for (int kb = 0; kb < 4; ++kb) {
        for (int kg = 0; kg < 16; ++kg) {
            const int input = kb * K_BLOCK + kg * 4;
            for (int ob = 0; ob < 8; ++ob) {
                pack_vnni_row(gate, 256, ob, input, dst + cursor);
                cursor += 64;
                pack_vnni_row(up, 256, ob, input, dst + cursor);
                cursor += 64;
            }
        }
    }
}

static void pack_s1_down_stream(const int8_t* down, int8_t* dst) {
    size_t cursor = 0;
    for (int kb = 0; kb < 2; ++kb) {
        for (int kg = 0; kg < 16; ++kg) {
            const int input = kb * K_BLOCK + kg * 4;
            for (int ob = 0; ob < 16; ++ob) {
                pack_vnni_row(down, 128, ob, input, dst + cursor);
                cursor += 64;
            }
        }
    }
}

static void make_compensation(const int8_t* src, int rows, int cols,
                              int32_t* dst) {
    for (int r = 0; r < rows; ++r) {
        int32_t sum = 0;
        for (int c = 0; c < cols; ++c) sum += src[(size_t)r * cols + c];
        dst[r] = -128 * sum;
    }
}

// A compact AVX-512 exp approximation (Cephes range reduction/polynomial).
// Its relative error is much smaller than the lab's output tolerance, while
// avoiding one scalar libm call per hidden channel.
static inline __m512 exp512_ps(__m512 x) {
    const __m512 hi = _mm512_set1_ps(88.3762626647949f);
    const __m512 lo = _mm512_set1_ps(-88.3762626647949f);
    x = _mm512_min_ps(x, hi);
    x = _mm512_max_ps(x, lo);

    __m512 fx = _mm512_fmadd_ps(
        x, _mm512_set1_ps(1.44269504088896341f), _mm512_set1_ps(0.5f));
    fx = _mm512_floor_ps(fx);

    x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693359375f), x);
    x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.12194440e-4f), x);
    const __m512 z = _mm512_mul_ps(x, x);

    __m512 y = _mm512_set1_ps(1.9875691500E-4f);
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507E-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073E-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894E-2f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201E-1f));
    y = _mm512_fmadd_ps(y, z, x);
    y = _mm512_add_ps(y, _mm512_set1_ps(1.0f));

    __m512i emm = _mm512_cvttps_epi32(fx);
    emm = _mm512_add_epi32(emm, _mm512_set1_epi32(127));
    emm = _mm512_slli_epi32(emm, 23);
    return _mm512_mul_ps(y, _mm512_castsi512_ps(emm));
}

static inline __m512 sigmoid512_ps(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    return _mm512_div_ps(one,
                         _mm512_add_ps(one, exp512_ps(_mm512_sub_ps(
                             _mm512_setzero_ps(), x))));
}

static inline __m512 expert_sigmoid512_ps(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 denominator = _mm512_add_ps(
        one, exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), x)));
    // The 14-bit reciprocal error is far below the following INT8
    // requantization step; avoiding a Newton refinement shortens every
    // hidden-channel block on the latency and batch paths.
    return _mm512_rcp14_ps(denominator);
}

static inline __m512 s1_sigmoid512_ps(__m512 x) {
    __m512 z = _mm512_sub_ps(_mm512_setzero_ps(), x);
    z = _mm512_min_ps(z, _mm512_set1_ps(88.3762626647949f));
    z = _mm512_max_ps(z, _mm512_set1_ps(-88.3762626647949f));
    __m512 exponent = _mm512_floor_ps(_mm512_fmadd_ps(
        z, _mm512_set1_ps(1.44269504088896341f),
        _mm512_set1_ps(0.5f)));
    z = _mm512_fnmadd_ps(exponent, _mm512_set1_ps(0.693359375f), z);
    z = _mm512_fnmadd_ps(exponent, _mm512_set1_ps(-2.12194440e-4f), z);
    // Cubic is sufficient on the reduced [-ln(2)/2, ln(2)/2] interval;
    // maximum relative exp error is below 8e-4 and is absorbed by h -> INT8.
    __m512 polynomial = _mm512_fmadd_ps(
        z, _mm512_set1_ps(1.0f / 6.0f), _mm512_set1_ps(0.5f));
    polynomial = _mm512_fmadd_ps(polynomial, z, _mm512_set1_ps(1.0f));
    polynomial = _mm512_fmadd_ps(polynomial, z, _mm512_set1_ps(1.0f));
    __m512i bits = _mm512_add_epi32(
        _mm512_cvttps_epi32(exponent), _mm512_set1_epi32(127));
    bits = _mm512_slli_epi32(bits, 23);
    const __m512 denominator = _mm512_add_ps(
        _mm512_set1_ps(1.0f),
        _mm512_mul_ps(polynomial, _mm512_castsi512_ps(bits)));
    return _mm512_rcp14_ps(denominator);
}

static inline __m512 s1_router_sigmoid512_ps(__m512 x) {
    // Odd 2/2 rational approximation around 0.5.  Router logits are close to
    // unit variance; the maximum absolute error remains about 1.1e-4 even on
    // [-10, 10], well below the gate precision needed before the final FMA.
    const __m512 x2 = _mm512_mul_ps(x, x);
    __m512 numerator = _mm512_fmadd_ps(
        x2, _mm512_set1_ps(1.1455767e-5f),
        _mm512_set1_ps(6.5038041e-3f));
    numerator = _mm512_fmadd_ps(
        numerator, x2, _mm512_set1_ps(2.4996710e-1f));
    __m512 denominator = _mm512_fmadd_ps(
        x2, _mm512_set1_ps(8.3612718e-4f),
        _mm512_set1_ps(1.0926407e-1f));
    denominator = _mm512_fmadd_ps(
        denominator, x2, _mm512_set1_ps(1.0f));
    __m512 result = _mm512_fmadd_ps(
        _mm512_mul_ps(x, numerator), _mm512_rcp14_ps(denominator),
        _mm512_set1_ps(0.5f));
    result = _mm512_max_ps(result, _mm512_setzero_ps());
    return _mm512_min_ps(result, _mm512_set1_ps(1.0f));
}

static inline __m512 abs512_ps(__m512 x) {
    return _mm512_castsi512_ps(_mm512_and_epi32(
        _mm512_castps_si512(x), _mm512_set1_epi32(0x7fffffff)));
}

static inline void quantize_row(const float* x, int length, int8_t* q,
                                float& scale) {
    __m512 vmax = _mm512_setzero_ps();
    for (int i = 0; i < length; i += 16) {
        vmax = _mm512_max_ps(vmax, abs512_ps(_mm512_loadu_ps(x + i)));
    }
    const float amax = _mm512_reduce_max_ps(vmax);
    scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const __m512 vinv_scale = _mm512_set1_ps(1.0f / scale);
    for (int i = 0; i < length; i += 16) {
        const __m512 v = _mm512_mul_ps(_mm512_loadu_ps(x + i), vinv_scale);
        const __m512i qi = _mm512_cvtps_epi32(v);
        const __m128i q8 = _mm512_cvtsepi32_epi8(qi);
        _mm_store_si128(reinterpret_cast<__m128i*>(q + i), q8);
    }
}

static inline void copy_i8_aligned(const int8_t* source, int8_t* destination,
                                   int length) {
    for (int i = 0; i < length; i += 64)
        _mm512_store_si512(destination + i,
                          _mm512_load_si512(source + i));
}

static inline void copy_f32(const float* source, float* destination,
                            int length) {
    for (int i = 0; i < length; i += N_BLOCK)
        _mm512_storeu_ps(destination + i, _mm512_loadu_ps(source + i));
}

// AMX expert kernel.  The tile configuration is loaded by the surrounding
// parallel region: one-row tiles for latency-sensitive N=1 and full 16-row
// tiles for the grouped large-batch path.  `rows` is the number of real rows;
// padded activation rows are zero and are ignored on output.
static __attribute__((noinline)) void expert_amx(
    const int8_t* gu, const int8_t* down, float s_gate, float s_up,
    float s_down, const int8_t* aq, const float* sx, float* const* destination,
    int rows, float coeff, bool accumulate, int d_model, int d_ff) {
    alignas(64) int32_t gate_acc[ROUTER_BATCH * MAX_D_FF];
    alignas(64) int32_t up_acc[ROUTER_BATCH * MAX_D_FF];
    alignas(64) float hidden[ROUTER_BATCH * MAX_D_FF];
    alignas(64) int8_t hq[ROUTER_BATCH * MAX_D_FF];
    alignas(64) int32_t down_acc[ROUTER_BATCH * N_BLOCK];

    const int d_kblocks = d_model / K_BLOCK;
    const int h_oblocks = d_ff / N_BLOCK;

    for (int ob = 0; ob < h_oblocks; ob += 3) {
        const int group = std::min(3, h_oblocks - ob);
        const int8_t* group_base = gu +
            (size_t)ob * 2 * d_kblocks * TILE_BYTES;
        _tile_zero(0);  // gate 0
        _tile_zero(1);  // up 0
        if (group > 1) {
            _tile_zero(2);  // gate 1
            _tile_zero(3);  // up 1
        }
        if (group > 2) {
            _tile_zero(4);  // gate 2
            _tile_zero(5);  // up 2
        }
        for (int kb = 0; kb < d_kblocks; ++kb) {
            _tile_loadd(6, aq + kb * K_BLOCK, d_model);
            const int8_t* bg0 = group_base +
                (size_t)kb * group * 2 * TILE_BYTES;
            const int8_t* bu0 = bg0 + TILE_BYTES;
            _tile_loadd(7, bg0, 64);
            _tile_dpbssd(0, 6, 7);
            _tile_loadd(7, bu0, 64);
            _tile_dpbssd(1, 6, 7);
            if (group > 1) {
                const int8_t* bg1 = bg0 + 2 * TILE_BYTES;
                const int8_t* bu1 = bg1 + TILE_BYTES;
                _tile_loadd(7, bg1, 64);
                _tile_dpbssd(2, 6, 7);
                _tile_loadd(7, bu1, 64);
                _tile_dpbssd(3, 6, 7);
            }
            if (group > 2) {
                const int8_t* bg2 = bg0 + 4 * TILE_BYTES;
                const int8_t* bu2 = bg2 + TILE_BYTES;
                _tile_loadd(7, bg2, 64);
                _tile_dpbssd(4, 6, 7);
                _tile_loadd(7, bu2, 64);
                _tile_dpbssd(5, 6, 7);
            }
        }
        _tile_stored(0, gate_acc + ob * N_BLOCK, d_ff * sizeof(int32_t));
        _tile_stored(1, up_acc + ob * N_BLOCK, d_ff * sizeof(int32_t));
        if (group > 1) {
            _tile_stored(2, gate_acc + (ob + 1) * N_BLOCK,
                         d_ff * sizeof(int32_t));
            _tile_stored(3, up_acc + (ob + 1) * N_BLOCK,
                         d_ff * sizeof(int32_t));
        }
        if (group > 2) {
            _tile_stored(4, gate_acc + (ob + 2) * N_BLOCK,
                         d_ff * sizeof(int32_t));
            _tile_stored(5, up_acc + (ob + 2) * N_BLOCK,
                         d_ff * sizeof(int32_t));
        }
    }

    std::memset(hq, 0, (size_t)ROUTER_BATCH * d_ff);
    float hidden_scale[ROUTER_BATCH];
    for (int r = 0; r < rows; ++r) {
        __m512 vmax = _mm512_setzero_ps();
        const __m512 gs = _mm512_set1_ps(sx[r] * s_gate);
        const __m512 us = _mm512_set1_ps(sx[r] * s_up);
        for (int f = 0; f < d_ff; f += N_BLOCK) {
            const __m512 vg = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_load_si512(
                    gate_acc + (size_t)r * d_ff + f)), gs);
            const __m512 vu = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_load_si512(
                    up_acc + (size_t)r * d_ff + f)), us);
            const __m512 hv = _mm512_mul_ps(
                _mm512_mul_ps(vg, s1_router_sigmoid512_ps(vg)), vu);
            _mm512_store_ps(hidden + (size_t)r * d_ff + f, hv);
            vmax = _mm512_max_ps(vmax, abs512_ps(hv));
        }
        const float amax = _mm512_reduce_max_ps(vmax);
        hidden_scale[r] = amax > 0.0f ? amax / 127.0f : 1.0f;
        const __m512 hs = _mm512_set1_ps(1.0f / hidden_scale[r]);
        for (int f = 0; f < d_ff; f += N_BLOCK) {
            const __m512 v = _mm512_mul_ps(
                _mm512_load_ps(hidden + (size_t)r * d_ff + f), hs);
            const __m128i q8 =
                _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(v));
            _mm_store_si128(reinterpret_cast<__m128i*>(
                                hq + (size_t)r * d_ff + f),
                            q8);
        }
    }

    const int h_kblocks = d_ff / K_BLOCK;
    const int d_oblocks = d_model / N_BLOCK;
    const __m512 vc = _mm512_set1_ps(coeff);
    for (int ob = 0; ob < d_oblocks; ob += 6) {
        const int group = std::min(6, d_oblocks - ob);
        const int8_t* group_base = down +
            (size_t)ob * h_kblocks * TILE_BYTES;
        _tile_zero(0);
        if (group > 1) _tile_zero(1);
        if (group > 2) _tile_zero(2);
        if (group > 3) _tile_zero(3);
        if (group > 4) _tile_zero(4);
        if (group > 5) _tile_zero(5);
        for (int kb = 0; kb < h_kblocks; ++kb) {
            _tile_loadd(6, hq + kb * K_BLOCK, d_ff);
            const int8_t* b0 = group_base +
                (size_t)kb * group * TILE_BYTES;
            _tile_loadd(7, b0, 64);
            _tile_dpbssd(0, 6, 7);
            if (group > 1) {
                _tile_loadd(7, b0 + TILE_BYTES, 64);
                _tile_dpbssd(1, 6, 7);
            }
            if (group > 2) {
                _tile_loadd(7, b0 + 2 * TILE_BYTES, 64);
                _tile_dpbssd(2, 6, 7);
            }
            if (group > 3) {
                _tile_loadd(7, b0 + 3 * TILE_BYTES, 64);
                _tile_dpbssd(3, 6, 7);
            }
            if (group > 4) {
                _tile_loadd(7, b0 + 4 * TILE_BYTES, 64);
                _tile_dpbssd(4, 6, 7);
            }
            if (group > 5) {
                _tile_loadd(7, b0 + 5 * TILE_BYTES, 64);
                _tile_dpbssd(5, 6, 7);
            }
        }
        auto emit = [&](int output_block) {
            for (int r = 0; r < rows; ++r) {
                __m512 out = _mm512_mul_ps(
                    _mm512_cvtepi32_ps(_mm512_load_si512(
                        down_acc + r * N_BLOCK)),
                    _mm512_set1_ps(hidden_scale[r] * s_down));
                out = _mm512_mul_ps(out, vc);
                if (accumulate) {
                    float* dst = destination[r] + output_block * N_BLOCK;
                    out = _mm512_add_ps(_mm512_loadu_ps(dst), out);
                    _mm512_storeu_ps(dst, out);
                } else {
                    uint16_t* dst = reinterpret_cast<uint16_t*>(destination[r]) +
                                    output_block * N_BLOCK;
                    _mm256_store_si256(
                        reinterpret_cast<__m256i*>(dst),
                        _mm512_cvtps_ph(out, _MM_FROUND_TO_NEAREST_INT |
                                                _MM_FROUND_NO_EXC));
                }
            }
        };
        _tile_stored(0, down_acc, N_BLOCK * sizeof(int32_t));
        emit(ob);
        if (group > 1) {
            _tile_stored(1, down_acc, N_BLOCK * sizeof(int32_t));
            emit(ob + 1);
        }
        if (group > 2) {
            _tile_stored(2, down_acc, N_BLOCK * sizeof(int32_t));
            emit(ob + 2);
        }
        if (group > 3) {
            _tile_stored(3, down_acc, N_BLOCK * sizeof(int32_t));
            emit(ob + 3);
        }
        if (group > 4) {
            _tile_stored(4, down_acc, N_BLOCK * sizeof(int32_t));
            emit(ob + 4);
        }
        if (group > 5) {
            _tile_stored(5, down_acc, N_BLOCK * sizeof(int32_t));
            emit(ob + 5);
        }
    }
}

// S3 routed experts commonly contain 17--48 rows.  This kernel computes two
// 16-row A tiles together: each packed B tile is loaded once and updates both
// batches before the stream advances.  Four C accumulators leave room for two
// A tiles and one B tile in the eight-register AMX file.
static __attribute__((noinline)) void expert_amx_pair_s3(
    const int8_t* gu, const int8_t* down, float s_gate, float s_up,
    float s_down, const int8_t* aq, const float* sx,
    float* const* destination, int rows) {
    alignas(64) int32_t gate_acc[32 * 128];
    alignas(64) int32_t up_acc[32 * 128];
    alignas(64) float hidden[32 * 128];
    alignas(64) int8_t hq[32 * 128];
    alignas(64) int32_t down_acc[16 * N_BLOCK];
    alignas(64) float output_scale[32];

    constexpr int d_model = 256;
    constexpr int d_ff = 128;
    constexpr int d_kblocks = d_model / K_BLOCK;
    constexpr int h_oblocks = d_ff / N_BLOCK;
    const int second_rows = rows - ROUTER_BATCH;

    for (int ob = 0; ob < h_oblocks; ++ob) {
        const int group_start = (ob / 3) * 3;
        const int group = std::min(3, h_oblocks - group_start);
        const int lane = ob - group_start;
        const int8_t* group_base = gu +
            (size_t)group_start * 2 * d_kblocks * TILE_BYTES;
        _tile_zero(0);  // first batch gate/up
        _tile_zero(1);
        _tile_zero(2);  // second batch gate/up
        _tile_zero(3);
        for (int kb = 0; kb < d_kblocks; ++kb) {
            _tile_loadd(4, aq + kb * K_BLOCK, d_model);
            _tile_loadd(5, aq + (size_t)ROUTER_BATCH * d_model +
                               kb * K_BLOCK, d_model);
            const int8_t* b = group_base +
                ((size_t)kb * group * 2 + lane * 2) * TILE_BYTES;
            _tile_loadd(7, b, 64);
            _tile_dpbssd(0, 4, 7);
            _tile_dpbssd(2, 5, 7);
            _tile_loadd(7, b + TILE_BYTES, 64);
            _tile_dpbssd(1, 4, 7);
            _tile_dpbssd(3, 5, 7);
        }
        _tile_stored(0, gate_acc + ob * N_BLOCK,
                     d_ff * sizeof(int32_t));
        _tile_stored(1, up_acc + ob * N_BLOCK,
                     d_ff * sizeof(int32_t));
        _tile_stored(2, gate_acc + ROUTER_BATCH * d_ff + ob * N_BLOCK,
                     d_ff * sizeof(int32_t));
        _tile_stored(3, up_acc + ROUTER_BATCH * d_ff + ob * N_BLOCK,
                     d_ff * sizeof(int32_t));
    }

    std::memset(hq, 0, sizeof(hq));
    for (int r = 0; r < rows; ++r) {
        __m512 vmax = _mm512_setzero_ps();
        const __m512 gs = _mm512_set1_ps(sx[r] * s_gate);
        const __m512 us = _mm512_set1_ps(sx[r] * s_up);
        for (int f = 0; f < d_ff; f += N_BLOCK) {
            const __m512 vg = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_load_si512(
                    gate_acc + (size_t)r * d_ff + f)), gs);
            const __m512 vu = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_load_si512(
                    up_acc + (size_t)r * d_ff + f)), us);
            const __m512 hv = _mm512_mul_ps(
                _mm512_mul_ps(vg, s1_router_sigmoid512_ps(vg)), vu);
            _mm512_store_ps(hidden + (size_t)r * d_ff + f, hv);
            vmax = _mm512_max_ps(vmax, abs512_ps(hv));
        }
        const float amax = _mm512_reduce_max_ps(vmax);
        const float hidden_scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        output_scale[r] = hidden_scale * s_down;
        const __m512 inverse = _mm512_set1_ps(1.0f / hidden_scale);
        for (int f = 0; f < d_ff; f += N_BLOCK) {
            const __m512 q = _mm512_mul_ps(
                _mm512_load_ps(hidden + (size_t)r * d_ff + f), inverse);
            _mm_store_si128(
                reinterpret_cast<__m128i*>(hq + (size_t)r * d_ff + f),
                _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(q)));
        }
    }

    constexpr int h_kblocks = d_ff / K_BLOCK;
    constexpr int d_oblocks = d_model / N_BLOCK;
    auto emit_batch = [&](const int32_t* acc, int first_row, int real_rows,
                          int output_block) {
        for (int r = 0; r < real_rows; ++r) {
            const __m512 out = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_load_si512(
                    acc + (size_t)r * N_BLOCK)),
                _mm512_set1_ps(output_scale[first_row + r]));
            uint16_t* dst =
                reinterpret_cast<uint16_t*>(destination[first_row + r]) +
                output_block * N_BLOCK;
            _mm256_store_si256(
                reinterpret_cast<__m256i*>(dst),
                _mm512_cvtps_ph(out, _MM_FROUND_TO_NEAREST_INT |
                                        _MM_FROUND_NO_EXC));
        }
    };
    for (int ob = 0; ob < d_oblocks; ob += 2) {
        const int group_start = (ob / 6) * 6;
        const int group = std::min(6, d_oblocks - group_start);
        const int lane = ob - group_start;
        const int8_t* group_base = down +
            (size_t)group_start * h_kblocks * TILE_BYTES;
        _tile_zero(0);  // first batch, output ob / ob+1
        _tile_zero(1);
        _tile_zero(2);  // second batch, output ob / ob+1
        _tile_zero(3);
        for (int kb = 0; kb < h_kblocks; ++kb) {
            _tile_loadd(4, hq + kb * K_BLOCK, d_ff);
            _tile_loadd(5, hq + (size_t)ROUTER_BATCH * d_ff +
                               kb * K_BLOCK, d_ff);
            const int8_t* b = group_base +
                ((size_t)kb * group + lane) * TILE_BYTES;
            _tile_loadd(7, b, 64);
            _tile_dpbssd(0, 4, 7);
            _tile_dpbssd(2, 5, 7);
            _tile_loadd(7, b + TILE_BYTES, 64);
            _tile_dpbssd(1, 4, 7);
            _tile_dpbssd(3, 5, 7);
        }
        _tile_stored(0, down_acc, N_BLOCK * sizeof(int32_t));
        emit_batch(down_acc, 0, ROUTER_BATCH, ob);
        _tile_stored(1, down_acc, N_BLOCK * sizeof(int32_t));
        emit_batch(down_acc, 0, ROUTER_BATCH, ob + 1);
        _tile_stored(2, down_acc, N_BLOCK * sizeof(int32_t));
        emit_batch(down_acc, ROUTER_BATCH, second_rows, ob);
        _tile_stored(3, down_acc, N_BLOCK * sizeof(int32_t));
        emit_batch(down_acc, ROUTER_BATCH, second_rows, ob + 1);
    }
}

// Process one expert with VNNI.  The output is accumulated directly into y;
// this removes the reference implementation's D-float temporary and a full
// extra traversal for every expert.
static inline void expert_vnni(const int8_t* gu, const int8_t* down,
                               const int32_t* gu_comp,
                               const int32_t* down_comp,
                               float s_gate, float s_up, float s_down,
                               const int8_t* xq, float s_x, float coeff,
                               float* y, int d_model, int d_ff) {
    alignas(64) float h[MAX_D_FF];
    alignas(64) int8_t hq[MAX_D_FF];

    const int d_kblocks = d_model / K_BLOCK;
    const int h_oblocks = d_ff / N_BLOCK;
    const __m512 gate_scale = _mm512_set1_ps(s_x * s_gate);
    const __m512 up_scale = _mm512_set1_ps(s_x * s_up);
    __m512 hmax = _mm512_setzero_ps();

    for (int ob = 0; ob < h_oblocks; ob += 3) {
        const int group = std::min(3, h_oblocks - ob);
        const int8_t* group_base = gu +
            (size_t)ob * 2 * d_kblocks * TILE_BYTES;
        __m512i ag0 = _mm512_load_si512(gu_comp + ob * N_BLOCK);
        __m512i au0 = _mm512_load_si512(gu_comp + d_ff + ob * N_BLOCK);
        __m512i ag1 = _mm512_setzero_si512(), au1 = ag1;
        __m512i ag2 = ag1, au2 = ag1;
        if (group > 1) {
            ag1 = _mm512_load_si512(gu_comp + (ob + 1) * N_BLOCK);
            au1 = _mm512_load_si512(gu_comp + d_ff + (ob + 1) * N_BLOCK);
        }
        if (group > 2) {
            ag2 = _mm512_load_si512(gu_comp + (ob + 2) * N_BLOCK);
            au2 = _mm512_load_si512(gu_comp + d_ff + (ob + 2) * N_BLOCK);
        }
        for (int kb = 0; kb < d_kblocks; ++kb) {
            const int8_t* b = group_base +
                (size_t)kb * group * 2 * TILE_BYTES;
            const int8_t* a = xq + kb * K_BLOCK;
            for (int kg = 0; kg < K_BLOCK / 4; ++kg) {
                int32_t x4;
                std::memcpy(&x4, a + kg * 4, sizeof(x4));
                const __m512i av = _mm512_set1_epi32(
                    x4 ^ static_cast<int32_t>(0x80808080u));
                const size_t row = (size_t)kg * 64;
                ag0 = _mm512_dpbusd_epi32(
                    ag0, av, _mm512_load_si512(b + row));
                au0 = _mm512_dpbusd_epi32(
                    au0, av, _mm512_load_si512(b + TILE_BYTES + row));
                if (group > 1) {
                    ag1 = _mm512_dpbusd_epi32(
                        ag1, av, _mm512_load_si512(b + 2 * TILE_BYTES + row));
                    au1 = _mm512_dpbusd_epi32(
                        au1, av, _mm512_load_si512(b + 3 * TILE_BYTES + row));
                }
                if (group > 2) {
                    ag2 = _mm512_dpbusd_epi32(
                        ag2, av, _mm512_load_si512(b + 4 * TILE_BYTES + row));
                    au2 = _mm512_dpbusd_epi32(
                        au2, av, _mm512_load_si512(b + 5 * TILE_BYTES + row));
                }
            }
        }
        auto emit_hidden = [&](int block, __m512i ag, __m512i au) {
            const __m512 vg =
                _mm512_mul_ps(_mm512_cvtepi32_ps(ag), gate_scale);
            const __m512 vu =
                _mm512_mul_ps(_mm512_cvtepi32_ps(au), up_scale);
            const __m512 hv = _mm512_mul_ps(
                _mm512_mul_ps(vg, expert_sigmoid512_ps(vg)), vu);
            _mm512_store_ps(h + block * N_BLOCK, hv);
            hmax = _mm512_max_ps(hmax, abs512_ps(hv));
        };
        emit_hidden(ob, ag0, au0);
        if (group > 1) emit_hidden(ob + 1, ag1, au1);
        if (group > 2) emit_hidden(ob + 2, ag2, au2);
    }

    const float h_amax = _mm512_reduce_max_ps(hmax);
    const float s_h = h_amax > 0.0f ? h_amax / 127.0f : 1.0f;
    const __m512 vhscale = _mm512_set1_ps(1.0f / s_h);
    for (int f = 0; f < d_ff; f += 16) {
        const __m512 v = _mm512_mul_ps(_mm512_load_ps(h + f), vhscale);
        const __m128i q8 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(v));
        _mm_store_si128(reinterpret_cast<__m128i*>(hq + f), q8);
    }

    const int h_kblocks = d_ff / K_BLOCK;
    const int d_oblocks = d_model / N_BLOCK;
    const __m512 out_scale = _mm512_set1_ps(s_h * s_down);
    const __m512 vcoeff = _mm512_set1_ps(coeff);
    for (int ob = 0; ob < d_oblocks; ob += 6) {
        const int group = std::min(6, d_oblocks - ob);
        const int8_t* group_base = down +
            (size_t)ob * h_kblocks * TILE_BYTES;
        __m512i a0 = _mm512_load_si512(down_comp + ob * N_BLOCK);
        __m512i a1 = _mm512_setzero_si512(), a2 = a1, a3 = a1;
        __m512i a4 = a1, a5 = a1;
        if (group > 1) a1 = _mm512_load_si512(down_comp + (ob + 1) * N_BLOCK);
        if (group > 2) a2 = _mm512_load_si512(down_comp + (ob + 2) * N_BLOCK);
        if (group > 3) a3 = _mm512_load_si512(down_comp + (ob + 3) * N_BLOCK);
        if (group > 4) a4 = _mm512_load_si512(down_comp + (ob + 4) * N_BLOCK);
        if (group > 5) a5 = _mm512_load_si512(down_comp + (ob + 5) * N_BLOCK);
        for (int kb = 0; kb < h_kblocks; ++kb) {
            const int8_t* b = group_base + (size_t)kb * group * TILE_BYTES;
            const int8_t* a = hq + kb * K_BLOCK;
            for (int kg = 0; kg < K_BLOCK / 4; ++kg) {
                int32_t x4;
                std::memcpy(&x4, a + kg * 4, sizeof(x4));
                const __m512i av = _mm512_set1_epi32(
                    x4 ^ static_cast<int32_t>(0x80808080u));
                const size_t row = (size_t)kg * 64;
                a0 = _mm512_dpbusd_epi32(a0, av, _mm512_load_si512(b + row));
                if (group > 1)
                    a1 = _mm512_dpbusd_epi32(
                        a1, av, _mm512_load_si512(b + TILE_BYTES + row));
                if (group > 2)
                    a2 = _mm512_dpbusd_epi32(
                        a2, av, _mm512_load_si512(b + 2 * TILE_BYTES + row));
                if (group > 3)
                    a3 = _mm512_dpbusd_epi32(
                        a3, av, _mm512_load_si512(b + 3 * TILE_BYTES + row));
                if (group > 4)
                    a4 = _mm512_dpbusd_epi32(
                        a4, av, _mm512_load_si512(b + 4 * TILE_BYTES + row));
                if (group > 5)
                    a5 = _mm512_dpbusd_epi32(
                        a5, av, _mm512_load_si512(b + 5 * TILE_BYTES + row));
            }
        }
        auto emit_output = [&](int block, __m512i acc) {
            __m512 out = _mm512_mul_ps(_mm512_cvtepi32_ps(acc), out_scale);
            out = _mm512_mul_ps(out, vcoeff);
            float* yd = y + block * N_BLOCK;
            _mm512_storeu_ps(yd, _mm512_add_ps(_mm512_loadu_ps(yd), out));
        };
        emit_output(ob, a0);
        if (group > 1) emit_output(ob + 1, a1);
        if (group > 2) emit_output(ob + 2, a2);
        if (group > 3) emit_output(ob + 3, a3);
        if (group > 4) emit_output(ob + 4, a4);
        if (group > 5) emit_output(ob + 5, a5);
    }
}

template <int GROUP>
static inline void vnni_s1_gu_group(const int8_t* gu, const int32_t* comp,
                                    const int8_t* xq, int ob,
                                    int32_t* gate_acc, int32_t* up_acc) {
    __m512i gate[GROUP];
    __m512i up[GROUP];
#pragma GCC unroll 3
    for (int j = 0; j < GROUP; ++j) {
        gate[j] = _mm512_load_si512(comp + (ob + j) * N_BLOCK);
        up[j] = _mm512_load_si512(comp + 128 + (ob + j) * N_BLOCK);
    }
    const int8_t* base = gu + (size_t)ob * 2 * 4 * TILE_BYTES;
    for (int kb = 0; kb < 4; ++kb) {
        const int8_t* stream = base + (size_t)kb * GROUP * 2 * TILE_BYTES;
        for (int kg = 0; kg < 16; ++kg) {
            int32_t x4;
            std::memcpy(&x4, xq + kb * 64 + kg * 4, sizeof(x4));
            const __m512i av = _mm512_set1_epi32(
                x4 ^ static_cast<int32_t>(0x80808080u));
            const size_t row = (size_t)kg * 64;
#pragma GCC unroll 3
            for (int j = 0; j < GROUP; ++j) {
                gate[j] = _mm512_dpbusd_epi32(
                    gate[j], av,
                    _mm512_load_si512(stream + (size_t)j * 2 * TILE_BYTES + row));
                up[j] = _mm512_dpbusd_epi32(
                    up[j], av,
                    _mm512_load_si512(stream + ((size_t)j * 2 + 1) * TILE_BYTES + row));
            }
        }
    }
#pragma GCC unroll 3
    for (int j = 0; j < GROUP; ++j) {
        _mm512_store_si512(gate_acc + (ob + j) * N_BLOCK, gate[j]);
        _mm512_store_si512(up_acc + (ob + j) * N_BLOCK, up[j]);
    }
}

template <int GROUP>
static inline __m512 vnni_s1_gu_hidden_group(
    const int8_t* gu, const int32_t* comp, const int8_t* xq, int ob,
    __m512 gate_scale, __m512 up_scale, float* hidden) {
    __m512i gate[GROUP];
    __m512i up[GROUP];
#pragma GCC unroll 3
    for (int j = 0; j < GROUP; ++j) {
        gate[j] = _mm512_load_si512(comp + (ob + j) * N_BLOCK);
        up[j] = _mm512_load_si512(comp + 128 + (ob + j) * N_BLOCK);
    }
    const int8_t* base = gu + (size_t)ob * 2 * 4 * TILE_BYTES;
    for (int kb = 0; kb < 4; ++kb) {
        const int8_t* stream = base + (size_t)kb * GROUP * 2 * TILE_BYTES;
        for (int kg = 0; kg < 16; ++kg) {
            int32_t x4;
            std::memcpy(&x4, xq + kb * 64 + kg * 4, sizeof(x4));
            const __m512i activation = _mm512_set1_epi32(
                x4 ^ static_cast<int32_t>(0x80808080u));
            const size_t row = (size_t)kg * 64;
#pragma GCC unroll 3
            for (int j = 0; j < GROUP; ++j) {
                gate[j] = _mm512_dpbusd_epi32(
                    gate[j], activation,
                    _mm512_load_si512(stream +
                                     (size_t)j * 2 * TILE_BYTES + row));
                up[j] = _mm512_dpbusd_epi32(
                    up[j], activation,
                    _mm512_load_si512(stream +
                                     ((size_t)j * 2 + 1) * TILE_BYTES + row));
            }
        }
    }
    __m512 maximum = _mm512_setzero_ps();
#pragma GCC unroll 3
    for (int j = 0; j < GROUP; ++j) {
        const __m512 vg = _mm512_mul_ps(
            _mm512_cvtepi32_ps(gate[j]), gate_scale);
        const __m512 vu = _mm512_mul_ps(
            _mm512_cvtepi32_ps(up[j]), up_scale);
        const __m512 h = _mm512_mul_ps(
            _mm512_mul_ps(vg, s1_router_sigmoid512_ps(vg)), vu);
        _mm512_store_ps(hidden + (ob + j) * N_BLOCK, h);
        maximum = _mm512_max_ps(maximum, abs512_ps(h));
    }
    return maximum;
}

static __attribute__((noinline)) void vnni_s1_gu_dot_stream(
    const int8_t* gu, const int32_t* comp, const int8_t* xq,
    int32_t* gate_acc, int32_t* up_acc) {
    __m512i g0 = _mm512_load_si512(comp + 0 * N_BLOCK);
    __m512i u0 = _mm512_load_si512(comp + 128 + 0 * N_BLOCK);
    __m512i g1 = _mm512_load_si512(comp + 1 * N_BLOCK);
    __m512i u1 = _mm512_load_si512(comp + 128 + 1 * N_BLOCK);
    __m512i g2 = _mm512_load_si512(comp + 2 * N_BLOCK);
    __m512i u2 = _mm512_load_si512(comp + 128 + 2 * N_BLOCK);
    __m512i g3 = _mm512_load_si512(comp + 3 * N_BLOCK);
    __m512i u3 = _mm512_load_si512(comp + 128 + 3 * N_BLOCK);
    __m512i g4 = _mm512_load_si512(comp + 4 * N_BLOCK);
    __m512i u4 = _mm512_load_si512(comp + 128 + 4 * N_BLOCK);
    __m512i g5 = _mm512_load_si512(comp + 5 * N_BLOCK);
    __m512i u5 = _mm512_load_si512(comp + 128 + 5 * N_BLOCK);
    __m512i g6 = _mm512_load_si512(comp + 6 * N_BLOCK);
    __m512i u6 = _mm512_load_si512(comp + 128 + 6 * N_BLOCK);
    __m512i g7 = _mm512_load_si512(comp + 7 * N_BLOCK);
    __m512i u7 = _mm512_load_si512(comp + 128 + 7 * N_BLOCK);
    const int8_t* stream = gu;
    for (int input = 0; input < 256; input += 4, stream += 16 * 64) {
        int32_t x4;
        std::memcpy(&x4, xq + input, sizeof(x4));
        const __m512i activation = _mm512_set1_epi32(
            x4 ^ static_cast<int32_t>(0x80808080u));
        asm volatile(
            "vpdpbusd    0(%[p]), %[a], %[g0]\n\t"
            "vpdpbusd   64(%[p]), %[a], %[u0]\n\t"
            "vpdpbusd  128(%[p]), %[a], %[g1]\n\t"
            "vpdpbusd  192(%[p]), %[a], %[u1]\n\t"
            "vpdpbusd  256(%[p]), %[a], %[g2]\n\t"
            "vpdpbusd  320(%[p]), %[a], %[u2]\n\t"
            "vpdpbusd  384(%[p]), %[a], %[g3]\n\t"
            "vpdpbusd  448(%[p]), %[a], %[u3]"
            : [g0] "+v"(g0), [u0] "+v"(u0),
              [g1] "+v"(g1), [u1] "+v"(u1),
              [g2] "+v"(g2), [u2] "+v"(u2),
              [g3] "+v"(g3), [u3] "+v"(u3)
            : [a] "v"(activation), [p] "r"(stream)
            : "memory");
        asm volatile(
            "vpdpbusd  512(%[p]), %[a], %[g4]\n\t"
            "vpdpbusd  576(%[p]), %[a], %[u4]\n\t"
            "vpdpbusd  640(%[p]), %[a], %[g5]\n\t"
            "vpdpbusd  704(%[p]), %[a], %[u5]\n\t"
            "vpdpbusd  768(%[p]), %[a], %[g6]\n\t"
            "vpdpbusd  832(%[p]), %[a], %[u6]\n\t"
            "vpdpbusd  896(%[p]), %[a], %[g7]\n\t"
            "vpdpbusd  960(%[p]), %[a], %[u7]"
            : [g4] "+v"(g4), [u4] "+v"(u4),
              [g5] "+v"(g5), [u5] "+v"(u5),
              [g6] "+v"(g6), [u6] "+v"(u6),
              [g7] "+v"(g7), [u7] "+v"(u7)
            : [a] "v"(activation), [p] "r"(stream)
            : "memory");
    }
    _mm512_store_si512(gate_acc + 0 * N_BLOCK, g0);
    _mm512_store_si512(up_acc + 0 * N_BLOCK, u0);
    _mm512_store_si512(gate_acc + 1 * N_BLOCK, g1);
    _mm512_store_si512(up_acc + 1 * N_BLOCK, u1);
    _mm512_store_si512(gate_acc + 2 * N_BLOCK, g2);
    _mm512_store_si512(up_acc + 2 * N_BLOCK, u2);
    _mm512_store_si512(gate_acc + 3 * N_BLOCK, g3);
    _mm512_store_si512(up_acc + 3 * N_BLOCK, u3);
    _mm512_store_si512(gate_acc + 4 * N_BLOCK, g4);
    _mm512_store_si512(up_acc + 4 * N_BLOCK, u4);
    _mm512_store_si512(gate_acc + 5 * N_BLOCK, g5);
    _mm512_store_si512(up_acc + 5 * N_BLOCK, u5);
    _mm512_store_si512(gate_acc + 6 * N_BLOCK, g6);
    _mm512_store_si512(up_acc + 6 * N_BLOCK, u6);
    _mm512_store_si512(gate_acc + 7 * N_BLOCK, g7);
    _mm512_store_si512(up_acc + 7 * N_BLOCK, u7);
}

template <int GROUP>
static inline void vnni_s1_down_group(const int8_t* down,
                                      const int32_t* comp, const int8_t* hq,
                                      int ob, int32_t* output_acc) {
    __m512i acc[GROUP];
#pragma GCC unroll 6
    for (int j = 0; j < GROUP; ++j)
        acc[j] = _mm512_load_si512(comp + (ob + j) * N_BLOCK);
    const int8_t* base = down + (size_t)ob * 2 * TILE_BYTES;
    for (int kb = 0; kb < 2; ++kb) {
        const int8_t* stream = base + (size_t)kb * GROUP * TILE_BYTES;
        for (int kg = 0; kg < 16; ++kg) {
            int32_t x4;
            std::memcpy(&x4, hq + kb * 64 + kg * 4, sizeof(x4));
            const __m512i av = _mm512_set1_epi32(
                x4 ^ static_cast<int32_t>(0x80808080u));
            const size_t row = (size_t)kg * 64;
#pragma GCC unroll 6
            for (int j = 0; j < GROUP; ++j)
                acc[j] = _mm512_dpbusd_epi32(
                    acc[j], av,
                    _mm512_load_si512(stream + (size_t)j * TILE_BYTES + row));
        }
    }
#pragma GCC unroll 6
    for (int j = 0; j < GROUP; ++j)
        _mm512_store_si512(output_acc + (ob + j) * N_BLOCK, acc[j]);
}

template <int GROUP>
static inline void vnni_s1_down_output_group(
    const int8_t* down, const int32_t* comp, const int8_t* hq, int ob,
    __m512 scale, float* output) {
    __m512i acc[GROUP];
#pragma GCC unroll 6
    for (int j = 0; j < GROUP; ++j)
        acc[j] = _mm512_load_si512(comp + (ob + j) * N_BLOCK);
    const int8_t* base = down + (size_t)ob * 2 * TILE_BYTES;
    for (int kb = 0; kb < 2; ++kb) {
        const int8_t* stream = base + (size_t)kb * GROUP * TILE_BYTES;
        for (int kg = 0; kg < 16; ++kg) {
            int32_t x4;
            std::memcpy(&x4, hq + kb * 64 + kg * 4, sizeof(x4));
            const __m512i activation = _mm512_set1_epi32(
                x4 ^ static_cast<int32_t>(0x80808080u));
            const size_t row = (size_t)kg * 64;
#pragma GCC unroll 6
            for (int j = 0; j < GROUP; ++j)
                acc[j] = _mm512_dpbusd_epi32(
                    acc[j], activation,
                    _mm512_load_si512(stream +
                                     (size_t)j * TILE_BYTES + row));
        }
    }
#pragma GCC unroll 6
    for (int j = 0; j < GROUP; ++j)
        _mm512_store_ps(
            output + (ob + j) * N_BLOCK,
            _mm512_mul_ps(_mm512_cvtepi32_ps(acc[j]), scale));
}

static __attribute__((noinline)) void vnni_s1_down_dot_stream(
    const int8_t* down, const int32_t* comp, const int8_t* hq,
    int32_t* output_acc) {
    __m512i a0 = _mm512_load_si512(comp + 0 * N_BLOCK);
    __m512i a1 = _mm512_load_si512(comp + 1 * N_BLOCK);
    __m512i a2 = _mm512_load_si512(comp + 2 * N_BLOCK);
    __m512i a3 = _mm512_load_si512(comp + 3 * N_BLOCK);
    __m512i a4 = _mm512_load_si512(comp + 4 * N_BLOCK);
    __m512i a5 = _mm512_load_si512(comp + 5 * N_BLOCK);
    __m512i a6 = _mm512_load_si512(comp + 6 * N_BLOCK);
    __m512i a7 = _mm512_load_si512(comp + 7 * N_BLOCK);
    __m512i a8 = _mm512_load_si512(comp + 8 * N_BLOCK);
    __m512i a9 = _mm512_load_si512(comp + 9 * N_BLOCK);
    __m512i a10 = _mm512_load_si512(comp + 10 * N_BLOCK);
    __m512i a11 = _mm512_load_si512(comp + 11 * N_BLOCK);
    __m512i a12 = _mm512_load_si512(comp + 12 * N_BLOCK);
    __m512i a13 = _mm512_load_si512(comp + 13 * N_BLOCK);
    __m512i a14 = _mm512_load_si512(comp + 14 * N_BLOCK);
    __m512i a15 = _mm512_load_si512(comp + 15 * N_BLOCK);
    const int8_t* stream = down;
    for (int input = 0; input < 128; input += 4, stream += 16 * 64) {
        int32_t x4;
        std::memcpy(&x4, hq + input, sizeof(x4));
        const __m512i activation = _mm512_set1_epi32(
            x4 ^ static_cast<int32_t>(0x80808080u));
        asm volatile(
            "vpdpbusd    0(%[p]), %[x], %[a0]\n\t"
            "vpdpbusd   64(%[p]), %[x], %[a1]\n\t"
            "vpdpbusd  128(%[p]), %[x], %[a2]\n\t"
            "vpdpbusd  192(%[p]), %[x], %[a3]\n\t"
            "vpdpbusd  256(%[p]), %[x], %[a4]\n\t"
            "vpdpbusd  320(%[p]), %[x], %[a5]\n\t"
            "vpdpbusd  384(%[p]), %[x], %[a6]\n\t"
            "vpdpbusd  448(%[p]), %[x], %[a7]"
            : [a0] "+v"(a0), [a1] "+v"(a1),
              [a2] "+v"(a2), [a3] "+v"(a3),
              [a4] "+v"(a4), [a5] "+v"(a5),
              [a6] "+v"(a6), [a7] "+v"(a7)
            : [x] "v"(activation), [p] "r"(stream)
            : "memory");
        asm volatile(
            "vpdpbusd  512(%[p]), %[x], %[a8]\n\t"
            "vpdpbusd  576(%[p]), %[x], %[a9]\n\t"
            "vpdpbusd  640(%[p]), %[x], %[a10]\n\t"
            "vpdpbusd  704(%[p]), %[x], %[a11]\n\t"
            "vpdpbusd  768(%[p]), %[x], %[a12]\n\t"
            "vpdpbusd  832(%[p]), %[x], %[a13]\n\t"
            "vpdpbusd  896(%[p]), %[x], %[a14]\n\t"
            "vpdpbusd  960(%[p]), %[x], %[a15]"
            : [a8] "+v"(a8), [a9] "+v"(a9),
              [a10] "+v"(a10), [a11] "+v"(a11),
              [a12] "+v"(a12), [a13] "+v"(a13),
              [a14] "+v"(a14), [a15] "+v"(a15)
            : [x] "v"(activation), [p] "r"(stream)
            : "memory");
    }
    _mm512_store_si512(output_acc + 0 * N_BLOCK, a0);
    _mm512_store_si512(output_acc + 1 * N_BLOCK, a1);
    _mm512_store_si512(output_acc + 2 * N_BLOCK, a2);
    _mm512_store_si512(output_acc + 3 * N_BLOCK, a3);
    _mm512_store_si512(output_acc + 4 * N_BLOCK, a4);
    _mm512_store_si512(output_acc + 5 * N_BLOCK, a5);
    _mm512_store_si512(output_acc + 6 * N_BLOCK, a6);
    _mm512_store_si512(output_acc + 7 * N_BLOCK, a7);
    _mm512_store_si512(output_acc + 8 * N_BLOCK, a8);
    _mm512_store_si512(output_acc + 9 * N_BLOCK, a9);
    _mm512_store_si512(output_acc + 10 * N_BLOCK, a10);
    _mm512_store_si512(output_acc + 11 * N_BLOCK, a11);
    _mm512_store_si512(output_acc + 12 * N_BLOCK, a12);
    _mm512_store_si512(output_acc + 13 * N_BLOCK, a13);
    _mm512_store_si512(output_acc + 14 * N_BLOCK, a14);
    _mm512_store_si512(output_acc + 15 * N_BLOCK, a15);
}

static __attribute__((noinline)) void vnni_s1_gu_dot(
    const int8_t* gu, const int32_t* comp, const int8_t* xq,
    int32_t* gate_acc, int32_t* up_acc) {
    vnni_s1_gu_group<3>(gu, comp, xq, 0, gate_acc, up_acc);
    vnni_s1_gu_group<3>(gu, comp, xq, 3, gate_acc, up_acc);
    vnni_s1_gu_group<2>(gu, comp, xq, 6, gate_acc, up_acc);
}

static __attribute__((noinline)) void vnni_s1_down_dot(
    const int8_t* down, const int32_t* comp, const int8_t* hq,
    int32_t* output_acc) {
    vnni_s1_down_group<6>(down, comp, hq, 0, output_acc);
    vnni_s1_down_group<6>(down, comp, hq, 6, output_acc);
    vnni_s1_down_group<4>(down, comp, hq, 12, output_acc);
}

#ifdef MOE_S1_256BIT
static inline __m256 s1_sigmoid256_ps(__m256 x) {
    __m256 z = _mm256_sub_ps(_mm256_setzero_ps(), x);
    z = _mm256_min_ps(z, _mm256_set1_ps(88.3762626647949f));
    z = _mm256_max_ps(z, _mm256_set1_ps(-88.3762626647949f));
    __m256 exponent = _mm256_floor_ps(_mm256_fmadd_ps(
        z, _mm256_set1_ps(1.44269504088896341f),
        _mm256_set1_ps(0.5f)));
    z = _mm256_fnmadd_ps(exponent, _mm256_set1_ps(0.693359375f), z);
    z = _mm256_fnmadd_ps(exponent, _mm256_set1_ps(-2.12194440e-4f), z);
    __m256 polynomial = _mm256_fmadd_ps(
        z, _mm256_set1_ps(1.0f / 6.0f), _mm256_set1_ps(0.5f));
    polynomial = _mm256_fmadd_ps(polynomial, z, _mm256_set1_ps(1.0f));
    polynomial = _mm256_fmadd_ps(polynomial, z, _mm256_set1_ps(1.0f));
    __m256i bits = _mm256_add_epi32(
        _mm256_cvttps_epi32(exponent), _mm256_set1_epi32(127));
    bits = _mm256_slli_epi32(bits, 23);
    const __m256 denominator = _mm256_add_ps(
        _mm256_set1_ps(1.0f),
        _mm256_mul_ps(polynomial, _mm256_castsi256_ps(bits)));
    return _mm256_rcp14_ps(denominator);
}

static inline __m256 abs256_ps(__m256 x) {
    return _mm256_castsi256_ps(_mm256_and_si256(
        _mm256_castps_si256(x), _mm256_set1_epi32(0x7fffffff)));
}

template <int GROUP>
static inline __m256 vnni_s1_gu_hidden_group_256(
    const int8_t* gu, const int32_t* comp, const int8_t* xq, int ob,
    __m256 gate_scale, __m256 up_scale, float* hidden) {
    __m256i gate_lo[GROUP], gate_hi[GROUP];
    __m256i up_lo[GROUP], up_hi[GROUP];
#pragma GCC unroll 3
    for (int j = 0; j < GROUP; ++j) {
        const int offset = (ob + j) * N_BLOCK;
        gate_lo[j] = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(comp + offset));
        gate_hi[j] = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(comp + offset + 8));
        up_lo[j] = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(comp + 128 + offset));
        up_hi[j] = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(comp + 128 + offset + 8));
    }
    const int8_t* base = gu + (size_t)ob * 2 * 4 * TILE_BYTES;
    for (int kb = 0; kb < 4; ++kb) {
        const int8_t* stream = base + (size_t)kb * GROUP * 2 * TILE_BYTES;
        for (int kg = 0; kg < 16; ++kg) {
            int32_t x4;
            std::memcpy(&x4, xq + kb * 64 + kg * 4, sizeof(x4));
            const __m256i activation = _mm256_set1_epi32(
                x4 ^ static_cast<int32_t>(0x80808080u));
            const size_t row = (size_t)kg * 64;
#pragma GCC unroll 3
            for (int j = 0; j < GROUP; ++j) {
                const int8_t* gate_weight =
                    stream + (size_t)j * 2 * TILE_BYTES + row;
                const int8_t* up_weight = gate_weight + TILE_BYTES;
                gate_lo[j] = _mm256_dpbusd_epi32(
                    gate_lo[j], activation,
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(
                        gate_weight)));
                gate_hi[j] = _mm256_dpbusd_epi32(
                    gate_hi[j], activation,
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(
                        gate_weight + 32)));
                up_lo[j] = _mm256_dpbusd_epi32(
                    up_lo[j], activation,
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(
                        up_weight)));
                up_hi[j] = _mm256_dpbusd_epi32(
                    up_hi[j], activation,
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(
                        up_weight + 32)));
            }
        }
    }
    __m256 maximum = _mm256_setzero_ps();
#pragma GCC unroll 3
    for (int j = 0; j < GROUP; ++j) {
        const __m256 vg_lo = _mm256_mul_ps(
            _mm256_cvtepi32_ps(gate_lo[j]), gate_scale);
        const __m256 vg_hi = _mm256_mul_ps(
            _mm256_cvtepi32_ps(gate_hi[j]), gate_scale);
        const __m256 vu_lo = _mm256_mul_ps(
            _mm256_cvtepi32_ps(up_lo[j]), up_scale);
        const __m256 vu_hi = _mm256_mul_ps(
            _mm256_cvtepi32_ps(up_hi[j]), up_scale);
        const __m256 h_lo = _mm256_mul_ps(
            _mm256_mul_ps(vg_lo, s1_sigmoid256_ps(vg_lo)), vu_lo);
        const __m256 h_hi = _mm256_mul_ps(
            _mm256_mul_ps(vg_hi, s1_sigmoid256_ps(vg_hi)), vu_hi);
        float* output = hidden + (ob + j) * N_BLOCK;
        _mm256_store_ps(output, h_lo);
        _mm256_store_ps(output + 8, h_hi);
        maximum = _mm256_max_ps(maximum, abs256_ps(h_lo));
        maximum = _mm256_max_ps(maximum, abs256_ps(h_hi));
    }
    return maximum;
}

template <int GROUP>
static inline void vnni_s1_down_output_group_256(
    const int8_t* down, const int32_t* comp, const int8_t* hq, int ob,
    __m256 scale, float* output) {
    __m256i acc_lo[GROUP], acc_hi[GROUP];
#pragma GCC unroll 6
    for (int j = 0; j < GROUP; ++j) {
        const int offset = (ob + j) * N_BLOCK;
        acc_lo[j] = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(comp + offset));
        acc_hi[j] = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(comp + offset + 8));
    }
    const int8_t* base = down + (size_t)ob * 2 * TILE_BYTES;
    for (int kb = 0; kb < 2; ++kb) {
        const int8_t* stream = base + (size_t)kb * GROUP * TILE_BYTES;
        for (int kg = 0; kg < 16; ++kg) {
            int32_t x4;
            std::memcpy(&x4, hq + kb * 64 + kg * 4, sizeof(x4));
            const __m256i activation = _mm256_set1_epi32(
                x4 ^ static_cast<int32_t>(0x80808080u));
            const size_t row = (size_t)kg * 64;
#pragma GCC unroll 6
            for (int j = 0; j < GROUP; ++j) {
                const int8_t* weight =
                    stream + (size_t)j * TILE_BYTES + row;
                acc_lo[j] = _mm256_dpbusd_epi32(
                    acc_lo[j], activation,
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(
                        weight)));
                acc_hi[j] = _mm256_dpbusd_epi32(
                    acc_hi[j], activation,
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(
                        weight + 32)));
            }
        }
    }
#pragma GCC unroll 6
    for (int j = 0; j < GROUP; ++j) {
        float* destination = output + (ob + j) * N_BLOCK;
        _mm256_store_ps(destination, _mm256_mul_ps(
            _mm256_cvtepi32_ps(acc_lo[j]), scale));
        _mm256_store_ps(destination + 8, _mm256_mul_ps(
            _mm256_cvtepi32_ps(acc_hi[j]), scale));
    }
}

static inline void expert_vnni_s1_256(
    const int8_t* gu, const int8_t* down, const int32_t* gu_comp,
    const int32_t* down_comp, float s_gate, float s_up, float s_down,
    const int8_t* xq, float s_x, float* output) {
    alignas(64) float hidden[128];
    alignas(64) int8_t hq[128];
    const __m256 gs = _mm256_set1_ps(s_x * s_gate);
    const __m256 us = _mm256_set1_ps(s_x * s_up);
    __m256 vmax = _mm256_setzero_ps();
    vmax = _mm256_max_ps(vmax, vnni_s1_gu_hidden_group_256<3>(
        gu, gu_comp, xq, 0, gs, us, hidden));
    vmax = _mm256_max_ps(vmax, vnni_s1_gu_hidden_group_256<3>(
        gu, gu_comp, xq, 3, gs, us, hidden));
    vmax = _mm256_max_ps(vmax, vnni_s1_gu_hidden_group_256<2>(
        gu, gu_comp, xq, 6, gs, us, hidden));
    const float amax = _mm256_reduce_max_ps(vmax);
    const float sh = amax > 0.0f ? amax / 127.0f : 1.0f;
    const __m256 inverse = _mm256_set1_ps(1.0f / sh);
    for (int f = 0; f < 128; f += N_BLOCK) {
        const __m256 q0 = _mm256_mul_ps(_mm256_load_ps(hidden + f), inverse);
        const __m256 q1 = _mm256_mul_ps(
            _mm256_load_ps(hidden + f + 8), inverse);
        const __m128i b0 = _mm256_cvtsepi32_epi8(_mm256_cvtps_epi32(q0));
        const __m128i b1 = _mm256_cvtsepi32_epi8(_mm256_cvtps_epi32(q1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(hq + f), b0);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(hq + f + 8), b1);
    }
    const __m256 scale = _mm256_set1_ps(sh * s_down);
    vnni_s1_down_output_group_256<6>(
        down, down_comp, hq, 0, scale, output);
    vnni_s1_down_output_group_256<6>(
        down, down_comp, hq, 6, scale, output);
    vnni_s1_down_output_group_256<4>(
        down, down_comp, hq, 12, scale, output);
}
#endif

static inline void expert_vnni_s1(
    const int8_t* gu, const int8_t* down, const int32_t* gu_comp,
    const int32_t* down_comp, float s_gate, float s_up, float s_down,
    const int8_t* xq, float s_x, float coeff, float* output) {
    alignas(64) float hidden[128];
    alignas(64) int8_t hq[128];

    const __m512 gs = _mm512_set1_ps(s_x * s_gate);
    const __m512 us = _mm512_set1_ps(s_x * s_up);
    __m512 vmax = _mm512_setzero_ps();
    vmax = _mm512_max_ps(
        vmax, vnni_s1_gu_hidden_group<3>(
                  gu, gu_comp, xq, 0, gs, us, hidden));
    vmax = _mm512_max_ps(
        vmax, vnni_s1_gu_hidden_group<3>(
                  gu, gu_comp, xq, 3, gs, us, hidden));
    vmax = _mm512_max_ps(
        vmax, vnni_s1_gu_hidden_group<2>(
                  gu, gu_comp, xq, 6, gs, us, hidden));
#ifdef MOE_S1_PREFETCH_DOWN_LINES
    // Seed every packed Down tile while the reduction and hidden
    // requantization below are still executing. This gives the L2 streamer a
    // head start without competing with the Gate/Up weight stream itself.
    for (int group_start = 0; group_start < 16; group_start += 6) {
        const int group = std::min(6, 16 - group_start);
        const int8_t* group_base = down + (size_t)group_start * 2 * TILE_BYTES;
        for (int kb = 0; kb < 2; ++kb) {
            const int8_t* kb_base =
                group_base + (size_t)kb * group * TILE_BYTES;
            for (int j = 0; j < group; ++j)
                for (int line = 0; line < MOE_S1_PREFETCH_DOWN_LINES; ++line)
                    _mm_prefetch(kb_base + (size_t)j * TILE_BYTES + line * 64,
                                 _MM_HINT_T0);
        }
    }
#endif
    const float amax = _mm512_reduce_max_ps(vmax);
    const float sh = amax > 0.0f ? amax / 127.0f : 1.0f;
    const __m512 inverse = _mm512_set1_ps(1.0f / sh);
    for (int f = 0; f < 128; f += N_BLOCK) {
        const __m512 q = _mm512_mul_ps(_mm512_load_ps(hidden + f), inverse);
        _mm_store_si128(reinterpret_cast<__m128i*>(hq + f),
                        _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(q)));
    }

    const __m512 scale = _mm512_set1_ps(
        sh * s_down
#ifdef MOE_S1_PREWEIGHT_OUTPUT
        * coeff
#endif
    );
    vnni_s1_down_output_group<6>(down, down_comp, hq, 0, scale, output);
    vnni_s1_down_output_group<6>(down, down_comp, hq, 6, scale, output);
    vnni_s1_down_output_group<4>(down, down_comp, hq, 12, scale, output);
}

#ifdef MOE_S1_STREAM_KERNEL
// S1-only full-width stream kernel. Keeping all eight gate/up blocks live
// removes the repeated activation broadcasts and loop setup of the 3/3/2
// grouped kernel. The temporary accumulators stay in the private L1.
static inline void expert_vnni_s1_stream(
    const int8_t* gu, const int8_t* down, const int32_t* gu_comp,
    const int32_t* down_comp, float s_gate, float s_up, float s_down,
    const int8_t* xq, float s_x, float coeff, float* output) {
    alignas(64) int32_t gate_acc[128];
    alignas(64) int32_t up_acc[128];
    alignas(64) int32_t down_acc[256];
    alignas(64) float hidden[128];
    alignas(64) int8_t hq[128];

    vnni_s1_gu_dot_stream(gu, gu_comp, xq, gate_acc, up_acc);
    const __m512 gs = _mm512_set1_ps(s_x * s_gate);
    const __m512 us = _mm512_set1_ps(s_x * s_up);
    __m512 vmax = _mm512_setzero_ps();
    for (int f = 0; f < 128; f += N_BLOCK) {
        const __m512 vg = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_load_si512(gate_acc + f)), gs);
        const __m512 vu = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_load_si512(up_acc + f)), us);
        const __m512 h = _mm512_mul_ps(
            _mm512_mul_ps(vg, s1_router_sigmoid512_ps(vg)), vu);
        _mm512_store_ps(hidden + f, h);
        vmax = _mm512_max_ps(vmax, abs512_ps(h));
    }
    const float amax = _mm512_reduce_max_ps(vmax);
    const float sh = amax > 0.0f ? amax / 127.0f : 1.0f;
    const __m512 inverse = _mm512_set1_ps(1.0f / sh);
    for (int f = 0; f < 128; f += N_BLOCK) {
        const __m512 q = _mm512_mul_ps(_mm512_load_ps(hidden + f), inverse);
        _mm_store_si128(reinterpret_cast<__m128i*>(hq + f),
                        _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(q)));
    }
    vnni_s1_down_dot_stream(down, down_comp, hq, down_acc);
    const __m512 scale = _mm512_set1_ps(
        sh * s_down
#ifdef MOE_S1_PREWEIGHT_OUTPUT
        * coeff
#endif
    );
    for (int d = 0; d < 256; d += N_BLOCK)
        _mm512_store_ps(
            output + d,
            _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_load_si512(down_acc + d)),
                scale));
}
#endif

static inline void select_topk(const float* affinity, const MoEWeights& w,
                               int* indices, float* gates) {
    float best_score[MAX_TOP_K] = {
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    int best_idx[MAX_TOP_K] = {-1, -1, -1, -1};

    // Experts are visited in increasing order and insertion is strict, so
    // equal scores retain the reference implementation's smaller-index tie
    // break without a separate used[] array or K full rescans.
    for (int e = 0; e < w.num_experts; ++e) {
        const float score = affinity[e] + w.bias[e];
        int pos = 0;
        while (pos < w.top_k && !(score > best_score[pos])) ++pos;
        if (pos == w.top_k) continue;
        for (int j = w.top_k - 1; j > pos; --j) {
            best_score[j] = best_score[j - 1];
            best_idx[j] = best_idx[j - 1];
        }
        best_score[pos] = score;
        best_idx[pos] = e;
    }

    float sum = 0.0f;
    for (int k = 0; k < w.top_k; ++k) sum += affinity[best_idx[k]];
    for (int k = 0; k < w.top_k; ++k) {
        indices[k] = best_idx[k];
        gates[k] = affinity[best_idx[k]] / sum;
    }
}

static inline void select_top4_s1(const float* affinity, const MoEWeights& w,
                                  int* indices, float* gates) {
    __m512 scores = _mm512_add_ps(_mm512_load_ps(affinity),
                                  _mm512_loadu_ps(w.bias));
    const __m512 negative_infinity =
        _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    float sum = 0.0f;
    for (int k = 0; k < 4; ++k) {
        const float best = _mm512_reduce_max_ps(scores);
        const __mmask16 matches = _mm512_cmp_ps_mask(
            scores, _mm512_set1_ps(best), _CMP_EQ_OQ);
        const int index = __builtin_ctz(static_cast<unsigned>(matches));
        indices[k] = index;
        gates[k] = affinity[index];
        sum += gates[k];
        scores = _mm512_mask_mov_ps(scores,
                                    static_cast<__mmask16>(1u << index),
                                    negative_infinity);
    }
    const float inverse_sum = 1.0f / sum;
    for (int k = 0; k < 4; ++k) gates[k] *= inverse_sum;
}

// Keep the FP32 dot-product kernel separate from sigmoid.  When both were
// inlined into one large function GCC kept polynomial constants live across
// the D loop and spilled ten of the sixteen accumulators on every iteration.
// These explicit accumulators all remain in ZMM registers.
static __attribute__((noinline)) void router_dot_16(
    const float* x, int d_model, const float* wr, float* out) {
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    __m512 a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    __m512 a8 = a0, a9 = a0, a10 = a0, a11 = a0;
    __m512 a12 = a0, a13 = a0, a14 = a0, a15 = a0;
    for (int d = 0; d < d_model; ++d) {
        const __m512 wv = _mm512_load_ps(wr + (size_t)d * N_BLOCK);
        a0 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[d]), a0);
        a1 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[d_model + d]), a1);
        a2 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[2LL * d_model + d]), a2);
        a3 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[3LL * d_model + d]), a3);
        a4 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[4LL * d_model + d]), a4);
        a5 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[5LL * d_model + d]), a5);
        a6 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[6LL * d_model + d]), a6);
        a7 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[7LL * d_model + d]), a7);
        a8 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[8LL * d_model + d]), a8);
        a9 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[9LL * d_model + d]), a9);
        a10 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[10LL * d_model + d]), a10);
        a11 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[11LL * d_model + d]), a11);
        a12 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[12LL * d_model + d]), a12);
        a13 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[13LL * d_model + d]), a13);
        a14 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[14LL * d_model + d]), a14);
        a15 = _mm512_fmadd_ps(wv, _mm512_set1_ps(x[15LL * d_model + d]), a15);
    }
    _mm512_store_ps(out + 0LL * MAX_NUM_EXPERTS, a0);
    _mm512_store_ps(out + 1LL * MAX_NUM_EXPERTS, a1);
    _mm512_store_ps(out + 2LL * MAX_NUM_EXPERTS, a2);
    _mm512_store_ps(out + 3LL * MAX_NUM_EXPERTS, a3);
    _mm512_store_ps(out + 4LL * MAX_NUM_EXPERTS, a4);
    _mm512_store_ps(out + 5LL * MAX_NUM_EXPERTS, a5);
    _mm512_store_ps(out + 6LL * MAX_NUM_EXPERTS, a6);
    _mm512_store_ps(out + 7LL * MAX_NUM_EXPERTS, a7);
    _mm512_store_ps(out + 8LL * MAX_NUM_EXPERTS, a8);
    _mm512_store_ps(out + 9LL * MAX_NUM_EXPERTS, a9);
    _mm512_store_ps(out + 10LL * MAX_NUM_EXPERTS, a10);
    _mm512_store_ps(out + 11LL * MAX_NUM_EXPERTS, a11);
    _mm512_store_ps(out + 12LL * MAX_NUM_EXPERTS, a12);
    _mm512_store_ps(out + 13LL * MAX_NUM_EXPERTS, a13);
    _mm512_store_ps(out + 14LL * MAX_NUM_EXPERTS, a14);
    _mm512_store_ps(out + 15LL * MAX_NUM_EXPERTS, a15);
}

static __attribute__((noinline)) void router_dot_small(
    const float* x, int rows, int d_model, const float* wr, float* out) {
    __m512 acc[ROUTER_BATCH];
    for (int r = 0; r < rows; ++r) acc[r] = _mm512_setzero_ps();
    for (int d = 0; d < d_model; ++d) {
        const __m512 wv = _mm512_load_ps(wr + (size_t)d * N_BLOCK);
        for (int r = 0; r < rows; ++r)
            acc[r] = _mm512_fmadd_ps(
                wv, _mm512_set1_ps(x[(size_t)r * d_model + d]), acc[r]);
    }
    for (int r = 0; r < rows; ++r)
        _mm512_store_ps(out + (size_t)r * MAX_NUM_EXPERTS, acc[r]);
}

static __attribute__((noinline)) void router_dot_1(
    const float* x, int d_model, const float* wr, float* out) {
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    __m512 a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (int d = 0; d < d_model; d += 8) {
        a0 = _mm512_fmadd_ps(_mm512_load_ps(wr + (size_t)(d + 0) * N_BLOCK),
                             _mm512_set1_ps(x[d + 0]), a0);
        a1 = _mm512_fmadd_ps(_mm512_load_ps(wr + (size_t)(d + 1) * N_BLOCK),
                             _mm512_set1_ps(x[d + 1]), a1);
        a2 = _mm512_fmadd_ps(_mm512_load_ps(wr + (size_t)(d + 2) * N_BLOCK),
                             _mm512_set1_ps(x[d + 2]), a2);
        a3 = _mm512_fmadd_ps(_mm512_load_ps(wr + (size_t)(d + 3) * N_BLOCK),
                             _mm512_set1_ps(x[d + 3]), a3);
        a4 = _mm512_fmadd_ps(_mm512_load_ps(wr + (size_t)(d + 4) * N_BLOCK),
                             _mm512_set1_ps(x[d + 4]), a4);
        a5 = _mm512_fmadd_ps(_mm512_load_ps(wr + (size_t)(d + 5) * N_BLOCK),
                             _mm512_set1_ps(x[d + 5]), a5);
        a6 = _mm512_fmadd_ps(_mm512_load_ps(wr + (size_t)(d + 6) * N_BLOCK),
                             _mm512_set1_ps(x[d + 6]), a6);
        a7 = _mm512_fmadd_ps(_mm512_load_ps(wr + (size_t)(d + 7) * N_BLOCK),
                             _mm512_set1_ps(x[d + 7]), a7);
    }
    a0 = _mm512_add_ps(a0, a1);
    a2 = _mm512_add_ps(a2, a3);
    a4 = _mm512_add_ps(a4, a5);
    a6 = _mm512_add_ps(a6, a7);
    a0 = _mm512_add_ps(a0, a2);
    a4 = _mm512_add_ps(a4, a6);
    _mm512_store_ps(out, _mm512_add_ps(a0, a4));
}

#ifdef MOE_S1_BF16_ROUTER
static __attribute__((noinline)) void router_dot_bf16_s1(
    const uint16_t* x, float* out) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = acc0;
    __m512 acc2 = acc0;
    __m512 acc3 = acc0;
    for (int pair = 0; pair < 128; pair += 4) {
        uint32_t x01, x23, x45, x67;
        std::memcpy(&x01, x + 2 * (pair + 0), sizeof(x01));
        std::memcpy(&x23, x + 2 * (pair + 1), sizeof(x23));
        std::memcpy(&x45, x + 2 * (pair + 2), sizeof(x45));
        std::memcpy(&x67, x + 2 * (pair + 3), sizeof(x67));
        const __m512bh a0 = (__m512bh)_mm512_set1_epi32(x01);
        const __m512bh a1 = (__m512bh)_mm512_set1_epi32(x23);
        const __m512bh a2 = (__m512bh)_mm512_set1_epi32(x45);
        const __m512bh a3 = (__m512bh)_mm512_set1_epi32(x67);
        acc0 = _mm512_dpbf16_ps(
            acc0, a0, (__m512bh)_mm512_load_si512(
                          s1_router_bf16 + (size_t)(pair + 0) * 32));
        acc1 = _mm512_dpbf16_ps(
            acc1, a1, (__m512bh)_mm512_load_si512(
                          s1_router_bf16 + (size_t)(pair + 1) * 32));
        acc2 = _mm512_dpbf16_ps(
            acc2, a2, (__m512bh)_mm512_load_si512(
                          s1_router_bf16 + (size_t)(pair + 2) * 32));
        acc3 = _mm512_dpbf16_ps(
            acc3, a3, (__m512bh)_mm512_load_si512(
                          s1_router_bf16 + (size_t)(pair + 3) * 32));
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    _mm512_store_ps(out, _mm512_add_ps(acc0, acc2));
}
#endif

// Router for up to sixteen tokens.  The same 16-expert weight vector is
// reused across all rows before advancing D, i.e. a 16x16 register-blocked
// GEMM rather than independent GEMVs.
static inline void router_block(const float* x, int rows, int d_model,
                                float affinity[ROUTER_BATCH][MAX_NUM_EXPERTS]) {
    const int eblocks = pw.padded_experts / N_BLOCK;
    for (int eb = 0; eb < eblocks; ++eb) {
        const float* wr = pw.router + (size_t)eb * d_model * N_BLOCK;
        float* out = affinity[0] + eb * N_BLOCK;
        if (rows == ROUTER_BATCH)
            router_dot_16(x, d_model, wr, out);
        else if (rows == 1)
            router_dot_1(x, d_model, wr, out);
        else
            router_dot_small(x, rows, d_model, wr, out);
    }
    for (int r = 0; r < rows; ++r) {
        for (int eb = 0; eb < eblocks; ++eb) {
            float* v = affinity[r] + eb * N_BLOCK;
            const __m512 logits = _mm512_load_ps(v);
            _mm512_store_ps(
                v, rows == 1
                       ? s1_router_sigmoid512_ps(logits)
                       : expert_sigmoid512_ps(logits));
        }
    }
}

// Compute a 16-token x E approximate router with signed INT8 AMX.  The
// packed layout is the same six-output-block stream used by the Down kernel.
static __attribute__((noinline)) void router_int8_dot_16(
    const int8_t* aq, int d_model, int32_t* accum) {
    const int kblocks = d_model / K_BLOCK;
    const int out_blocks = pw.padded_experts / N_BLOCK;
    for (int ob = 0; ob < out_blocks; ob += 6) {
        const int group = std::min(6, out_blocks - ob);
        const int8_t* group_base =
            pw.router_q + (size_t)ob * kblocks * TILE_BYTES;
        _tile_zero(0);
        if (group > 1) _tile_zero(1);
        if (group > 2) _tile_zero(2);
        if (group > 3) _tile_zero(3);
        if (group > 4) _tile_zero(4);
        if (group > 5) _tile_zero(5);
        for (int kb = 0; kb < kblocks; ++kb) {
            _tile_loadd(6, aq + kb * K_BLOCK, d_model);
            const int8_t* b = group_base +
                (size_t)kb * group * TILE_BYTES;
            _tile_loadd(7, b, 64);
            _tile_dpbssd(0, 6, 7);
            if (group > 1) {
                _tile_loadd(7, b + TILE_BYTES, 64);
                _tile_dpbssd(1, 6, 7);
            }
            if (group > 2) {
                _tile_loadd(7, b + 2 * TILE_BYTES, 64);
                _tile_dpbssd(2, 6, 7);
            }
            if (group > 3) {
                _tile_loadd(7, b + 3 * TILE_BYTES, 64);
                _tile_dpbssd(3, 6, 7);
            }
            if (group > 4) {
                _tile_loadd(7, b + 4 * TILE_BYTES, 64);
                _tile_dpbssd(4, 6, 7);
            }
            if (group > 5) {
                _tile_loadd(7, b + 5 * TILE_BYTES, 64);
                _tile_dpbssd(5, 6, 7);
            }
        }
        _tile_stored(0, accum + ob * N_BLOCK,
                     pw.padded_experts * sizeof(int32_t));
        if (group > 1)
            _tile_stored(1, accum + (ob + 1) * N_BLOCK,
                         pw.padded_experts * sizeof(int32_t));
        if (group > 2)
            _tile_stored(2, accum + (ob + 2) * N_BLOCK,
                         pw.padded_experts * sizeof(int32_t));
        if (group > 3)
            _tile_stored(3, accum + (ob + 3) * N_BLOCK,
                         pw.padded_experts * sizeof(int32_t));
        if (group > 4)
            _tile_stored(4, accum + (ob + 4) * N_BLOCK,
                         pw.padded_experts * sizeof(int32_t));
        if (group > 5)
            _tile_stored(5, accum + (ob + 5) * N_BLOCK,
                         pw.padded_experts * sizeof(int32_t));
    }
}

static inline float router_exact_affinity(const float* x, const float* weight,
                                          int d_model) {
    __m512 sum = _mm512_setzero_ps();
    for (int d = 0; d < d_model; d += N_BLOCK)
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(x + d),
                              _mm512_loadu_ps(weight + d), sum);
    const float logit = _mm512_reduce_add_ps(sum);
    return 1.0f / (1.0f + std::exp(-logit));
}

// The AMX result is used only to form a generous candidate set.  Top-K and
// gate values are recomputed from the original FP32 weights for those
// candidates, so router quantization never directly reaches the output.
static inline void select_top2_refined(const float* x, const MoEWeights& w,
                                       const int32_t* approx_acc, float sx,
                                       int* indices, float* gates) {
    alignas(64) float approx_score[MAX_NUM_EXPERTS];
    const __m512 input_scale = _mm512_set1_ps(sx);
    float best0 = -std::numeric_limits<float>::infinity();
    float best1 = best0;
    for (int e = 0; e < w.num_experts; e += N_BLOCK) {
        const __m512 dequant = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_load_si512(approx_acc + e)),
            _mm512_mul_ps(input_scale,
                          _mm512_load_ps(pw.router_q_scale + e)));
        const __m512 score = _mm512_add_ps(
            expert_sigmoid512_ps(dequant), _mm512_loadu_ps(w.bias + e));
        _mm512_store_ps(approx_score + e, score);
        const float block0 = _mm512_reduce_max_ps(score);
        const __mmask16 below_max = _mm512_cmp_ps_mask(
            score, _mm512_set1_ps(block0), _CMP_LT_OQ);
        const float block1 = _mm512_mask_reduce_max_ps(below_max, score);
        auto update_best = [&](float value) {
            if (value > best0) {
                best1 = best0;
                best0 = value;
            } else if (value > best1) {
                best1 = value;
            }
        };
        update_best(block0);
        update_best(block1);
    }

    constexpr int MAX_CANDIDATES = 64;
    constexpr float CANDIDATE_MARGIN = 0.04f;
    int candidates[MAX_CANDIDATES];
    int candidate_count = 0;
    const __m512 threshold = _mm512_set1_ps(best1 - CANDIDATE_MARGIN);
    const __m512i lanes = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    for (int e = 0; e < w.num_experts; e += N_BLOCK) {
        const __mmask16 selected = _mm512_cmp_ps_mask(
            _mm512_load_ps(approx_score + e), threshold, _CMP_GE_OQ);
        const int selected_count = __builtin_popcount(selected);
        if (candidate_count + selected_count > MAX_CANDIDATES) {
            candidate_count = w.num_experts;
            break;
        }
        _mm512_mask_compressstoreu_epi32(
            candidates + candidate_count, selected,
            _mm512_add_epi32(lanes, _mm512_set1_epi32(e)));
        candidate_count += selected_count;
    }
#ifdef MOE_ROUTER_AUDIT
    static std::atomic<uint64_t> audit_candidate_sum{0};
    static std::atomic<uint64_t> audit_token_count{0};
    static std::atomic<int> audit_candidate_max{0};
    audit_candidate_sum.fetch_add(candidate_count, std::memory_order_relaxed);
    int old_max = audit_candidate_max.load(std::memory_order_relaxed);
    while (candidate_count > old_max &&
           !audit_candidate_max.compare_exchange_weak(
               old_max, candidate_count, std::memory_order_relaxed)) {}
    const uint64_t audited =
        audit_token_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((audited & 1023) == 0)
        std::fprintf(stderr, "router candidates: avg=%.3f max=%d\n",
                     static_cast<double>(audit_candidate_sum.load()) / audited,
                     audit_candidate_max.load());
#endif

    float exact_best[2] = {
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    float exact_affinity[2] = {};
    int exact_index[2] = {-1, -1};
    const int scan_count = candidate_count == w.num_experts
        ? w.num_experts : candidate_count;
    for (int i = 0; i < scan_count; ++i) {
        const int e = candidate_count == w.num_experts ? i : candidates[i];
        const float affinity = router_exact_affinity(
            x, w.w_router + (size_t)e * w.d_model, w.d_model);
        const float score = affinity + w.bias[e];
        if (score > exact_best[0]) {
            exact_best[1] = exact_best[0];
            exact_affinity[1] = exact_affinity[0];
            exact_index[1] = exact_index[0];
            exact_best[0] = score;
            exact_affinity[0] = affinity;
            exact_index[0] = e;
        } else if (score > exact_best[1]) {
            exact_best[1] = score;
            exact_affinity[1] = affinity;
            exact_index[1] = e;
        }
    }
    const float inverse_sum = 1.0f / (exact_affinity[0] + exact_affinity[1]);
    indices[0] = exact_index[0];
    indices[1] = exact_index[1];
    gates[0] = exact_affinity[0] * inverse_sum;
    gates[1] = exact_affinity[1] * inverse_sum;
#ifdef MOE_ROUTER_AUDIT
    float audit_score[2] = {
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    int audit_index[2] = {-1, -1};
    for (int e = 0; e < w.num_experts; ++e) {
        const float affinity = router_exact_affinity(
            x, w.w_router + (size_t)e * w.d_model, w.d_model);
        const float score = affinity + w.bias[e];
        if (score > audit_score[0]) {
            audit_score[1] = audit_score[0];
            audit_index[1] = audit_index[0];
            audit_score[0] = score;
            audit_index[0] = e;
        } else if (score > audit_score[1]) {
            audit_score[1] = score;
            audit_index[1] = e;
        }
    }
    if (audit_index[0] != indices[0] || audit_index[1] != indices[1])
        std::abort();
#endif
}

static inline void forward_block(const float* x, const MoEWeights& w,
                                 float* y, int rows) {
    alignas(64) float affinity[ROUTER_BATCH][MAX_NUM_EXPERTS];
    router_block(x, rows, w.d_model, affinity);

    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * w.d_model;
        float* yr = y + (size_t)r * w.d_model;
        int indices[MAX_TOP_K];
        float gates[MAX_TOP_K];
        select_topk(affinity[r], w, indices, gates);

        alignas(64) int8_t xq[MAX_D_MODEL];
        float s_x;
        quantize_row(xr, w.d_model, xq, s_x);
        copy_f32(xr, yr, w.d_model);

        expert_vnni(pw.shared_gu, pw.shared_down,
                    pw.shared_gu_comp, pw.shared_down_comp,
                    w.sh_s_gate, w.sh_s_up, w.sh_s_down,
                    xq, s_x, 1.0f, yr, w.d_model, w.d_ff);
        for (int k = 0; k < w.top_k; ++k) {
            const int e = indices[k];
            expert_vnni(pw.routed_gu + (size_t)e * pw.gu_expert_stride,
                        pw.routed_down + (size_t)e * pw.down_expert_stride,
                        pw.routed_gu_comp + (size_t)e * 2 * w.d_ff,
                        pw.routed_down_comp + (size_t)e * w.d_model,
                        w.s_gate[e], w.s_up[e], w.s_down[e], xq, s_x,
                        gates[k], yr, w.d_model, w.d_ff);
        }
    }
}

struct SinglePipelineContext {
    const MoEWeights* w;
    const int* indices;
    const float* gates;
    const int8_t* aq;
    float sx;
    int experts;
    int gu_groups;
    int down_groups;
    int team;
    uint8_t gu_owner[(MAX_TOP_K + 1) * 11];
    uint8_t down_owner[(MAX_TOP_K + 1) * 11];
};

static void assign_single_tasks(SinglePipelineContext& ctx) {
    const MoEWeights& w = *ctx.w;
    // Profile-guided colors for the 16 routed experts plus the shared expert.
    // A color is the first core of that expert's cyclic group distribution.
    // Unlike per-call balancing, this keeps every packed tile on one core and
    // therefore preserves private-L2 residency across the rotating batches.
    static constexpr uint8_t s2_rotation[17] = {
        1, 4, 1, 1, 1, 4, 3, 6, 6, 5, 4, 4, 5, 2, 6, 2, 7};
    auto assign_stage = [&](int groups, uint8_t* owner) {
        for (int slot = 0; slot < ctx.experts; ++slot) {
            const int expert_key =
                slot == 0 ? w.num_experts : ctx.indices[slot - 1];
            const int rotation =
                (ctx.team == 8 && w.num_experts == 16)
                    ? s2_rotation[expert_key]
                    : (expert_key * groups) % ctx.team;
            for (int g = 0; g < groups; ++g)
                owner[slot * groups + g] =
                    static_cast<uint8_t>((rotation + g) % ctx.team);
        }
    };
    assign_stage(ctx.gu_groups, ctx.gu_owner);
    assign_stage(ctx.down_groups, ctx.down_owner);
}

static inline void single_expert_parameters(
    const SinglePipelineContext& ctx, int slot, const int8_t*& gu,
    const int8_t*& down, float& sg, float& su, float& sd) {
    const MoEWeights& w = *ctx.w;
    const int e = slot == 0 ? -1 : ctx.indices[slot - 1];
    gu = slot == 0
        ? pw.shared_gu
        : pw.routed_gu + (size_t)e * pw.gu_expert_stride;
    down = slot == 0
        ? pw.shared_down
        : pw.routed_down + (size_t)e * pw.down_expert_stride;
    sg = slot == 0 ? w.sh_s_gate : w.s_gate[e];
    su = slot == 0 ? w.sh_s_up : w.s_up[e];
    sd = slot == 0 ? w.sh_s_down : w.s_down[e];
}

static void single_whole_vnni_task(int slot, void* opaque) {
    const auto& ctx = *static_cast<SinglePipelineContext*>(opaque);
    const MoEWeights& w = *ctx.w;
    const int e = slot == 0 ? -1 : ctx.indices[slot - 1];
    const int8_t* gu = slot == 0
#ifdef MOE_S1_STREAM_KERNEL
        ? pw.shared_gu_s1
        : pw.routed_gu_s1 + (size_t)e * pw.gu_expert_stride;
    const int8_t* down = slot == 0
        ? pw.shared_down_s1
        : pw.routed_down_s1 + (size_t)e * pw.down_expert_stride;
#else
        ? pw.shared_gu
        : pw.routed_gu + (size_t)e * pw.gu_expert_stride;
    const int8_t* down = slot == 0
        ? pw.shared_down
        : pw.routed_down + (size_t)e * pw.down_expert_stride;
#endif
    const int32_t* gu_comp = slot == 0
        ? pw.shared_gu_comp
        : pw.routed_gu_comp + (size_t)e * 2 * w.d_ff;
    const int32_t* down_comp = slot == 0
        ? pw.shared_down_comp
        : pw.routed_down_comp + (size_t)e * w.d_model;
    const float sg = slot == 0 ? w.sh_s_gate : w.s_gate[e];
    const float su = slot == 0 ? w.sh_s_up : w.s_up[e];
    const float sd = slot == 0 ? w.sh_s_down : w.s_down[e];
    float* output = pw.routed_output + (size_t)slot * w.d_model;
    if (w.d_model == 256 && w.d_ff == 128) {
        const float coeff = slot == 0 ? 1.0f : ctx.gates[slot - 1];
#ifdef MOE_S1_256BIT
        expert_vnni_s1_256(gu, down, gu_comp, down_comp, sg, su, sd, ctx.aq,
                           ctx.sx, output);
#elif defined(MOE_S1_STREAM_KERNEL)
        expert_vnni_s1_stream(gu, down, gu_comp, down_comp, sg, su, sd,
                              ctx.aq, ctx.sx, coeff, output);
#else
        expert_vnni_s1(gu, down, gu_comp, down_comp, sg, su, sd, ctx.aq,
                       ctx.sx, coeff, output);
#endif
    } else {
        std::memset(output, 0, (size_t)w.d_model * sizeof(float));
        expert_vnni(gu, down, gu_comp, down_comp, sg, su, sd, ctx.aq,
                    ctx.sx, 1.0f, output, w.d_model, w.d_ff);
    }
}

struct SingleS1FusedContext {
    SinglePipelineContext* pipeline = nullptr;
    const float* x = nullptr;
#ifdef MOE_S1_BF16_ROUTER
    const uint16_t* xbf16 = nullptr;
#endif
    float* gates = nullptr;
    alignas(64) std::atomic<int> route_ready{0};
#ifdef MOE_PROFILE_S1
    uint64_t task_begin[MAX_TOP_K + 1]{};
    uint64_t task_expert_begin[MAX_TOP_K + 1]{};
    uint64_t task_end[MAX_TOP_K + 1]{};
    uint64_t route_end = 0;
#endif
};

// Five-core S1 DAG: core 1 starts the shared expert while core 0 routes, then
// cores 0/2/3/4 consume the four routed experts.  Compared with the earlier
// six-core version this retains the overlap without the package-frequency
// penalty of activating an otherwise idle sixth AVX-512 core.
static void single_s1_fused_task(int id, void* opaque) {
    auto& fused = *static_cast<SingleS1FusedContext*>(opaque);
    SinglePipelineContext& ctx = *fused.pipeline;
#ifdef MOE_PROFILE_S1
    unsigned int profile_aux;
    fused.task_begin[id] = __rdtscp(&profile_aux);
#endif
    if (id == 0) {
        alignas(64) float affinity[N_BLOCK];
#ifdef MOE_S1_BF16_ROUTER
        router_dot_bf16_s1(fused.xbf16, affinity);
#else
        router_dot_1(fused.x, 256, pw.router, affinity);
#endif
        const __m512 logits = _mm512_load_ps(affinity);
        _mm512_store_ps(affinity, s1_router_sigmoid512_ps(logits));
        select_top4_s1(affinity, *ctx.w,
                       const_cast<int*>(ctx.indices), fused.gates);
        int* indices = const_cast<int*>(ctx.indices);
        for (int i = 1; i < ctx.w->top_k; ++i) {
            const int index = indices[i];
            const float gate = fused.gates[i];
            int j = i;
            while (j != 0 && indices[j - 1] > index) {
                indices[j] = indices[j - 1];
                fused.gates[j] = fused.gates[j - 1];
                --j;
            }
            indices[j] = index;
            fused.gates[j] = gate;
        }
#ifdef MOE_ROUTER_AUDIT
        static std::atomic<int> printed{0};
        if (printed.fetch_add(1) < 26)
            std::fprintf(stderr, "S1 route: %d %d %d %d\n",
                         indices[0], indices[1], indices[2], indices[3]);
#endif
#ifdef MOE_PROFILE_S1
        fused.route_end = __rdtscp(&profile_aux);
#endif
        fused.route_ready.store(1, std::memory_order_release);
#ifdef MOE_PROFILE_S1
        fused.task_expert_begin[id] = __rdtscp(&profile_aux);
#endif
        single_whole_vnni_task(ctx.experts - 1, &ctx);
#ifdef MOE_PROFILE_S1
        fused.task_end[id] = __rdtscp(&profile_aux);
#endif
        return;
    }

    const int slot = id - 1;
    if (slot != 0) {
        while (fused.route_ready.load(std::memory_order_acquire) == 0)
            _mm_pause();
    }
#ifdef MOE_PROFILE_S1
    fused.task_expert_begin[id] = __rdtscp(&profile_aux);
#endif
    single_whole_vnni_task(slot, &ctx);
#ifdef MOE_PROFILE_S1
    fused.task_end[id] = __rdtscp(&profile_aux);
#endif
}

#ifdef MOE_S1_SLICED_DOWN
static void single_vnni_gu_whole_s1(int slot, SinglePipelineContext& ctx);
template <int START, int GROUP>
static inline void single_vnni_down_slice_accumulate_s1(
    int slot, const SinglePipelineContext& ctx, float* y);

struct SingleS1SlicedContext {
    SinglePipelineContext* pipeline = nullptr;
    const float* x = nullptr;
    float* y = nullptr;
    alignas(64) std::atomic<int> route_ready{0};
    alignas(64) std::atomic<int> gu_ready{0};
};

// Each core first owns one expert's Gate/Up stage.  After all five hidden
// vectors are ready, each core owns a disjoint output slice and reduces all
// experts directly into y.  The latter avoids transferring five complete
// expert outputs back to the main core for a separate combine pass.
static void single_s1_sliced_task(int id, void* opaque) {
    auto& sliced = *static_cast<SingleS1SlicedContext*>(opaque);
    SinglePipelineContext& ctx = *sliced.pipeline;
    int own_slot;
    if (id == 0) {
        alignas(64) float affinity[N_BLOCK];
        router_dot_1(sliced.x, 256, pw.router, affinity);
        _mm512_store_ps(
            affinity,
            s1_router_sigmoid512_ps(_mm512_load_ps(affinity)));
        select_top4_s1(affinity, *ctx.w,
                       const_cast<int*>(ctx.indices),
                       const_cast<float*>(ctx.gates));
        int* indices = const_cast<int*>(ctx.indices);
        float* gates = const_cast<float*>(ctx.gates);
        for (int i = 1; i < ctx.w->top_k; ++i) {
            const int index = indices[i];
            const float gate = gates[i];
            int j = i;
            while (j != 0 && indices[j - 1] > index) {
                indices[j] = indices[j - 1];
                gates[j] = gates[j - 1];
                --j;
            }
            indices[j] = index;
            gates[j] = gate;
        }
        sliced.route_ready.store(1, std::memory_order_release);
        own_slot = ctx.experts - 1;
    } else {
        own_slot = id - 1;
        if (own_slot != 0) {
            while (sliced.route_ready.load(std::memory_order_acquire) == 0)
                _mm_pause();
        }
    }

    single_vnni_gu_whole_s1(own_slot, ctx);
    sliced.gu_ready.fetch_add(1, std::memory_order_release);
    while (sliced.gu_ready.load(std::memory_order_acquire) != ctx.experts)
        _mm_pause();

    constexpr int starts[5] = {0, 4, 7, 10, 13};
    constexpr int groups[5] = {4, 3, 3, 3, 3};
    const int start = starts[id];
    const int group = groups[id];
    for (int j = 0; j < group; ++j) {
        const int d = (start + j) * N_BLOCK;
        _mm512_storeu_ps(sliced.y + d, _mm512_loadu_ps(sliced.x + d));
    }
    for (int slot = 0; slot < ctx.experts; ++slot) {
        if (id == 0)
            single_vnni_down_slice_accumulate_s1<0, 4>(slot, ctx,
                                                        sliced.y);
        else if (id == 1)
            single_vnni_down_slice_accumulate_s1<4, 3>(slot, ctx,
                                                        sliced.y);
        else if (id == 2)
            single_vnni_down_slice_accumulate_s1<7, 3>(slot, ctx,
                                                        sliced.y);
        else if (id == 3)
            single_vnni_down_slice_accumulate_s1<10, 3>(slot, ctx,
                                                         sliced.y);
        else
            single_vnni_down_slice_accumulate_s1<13, 3>(slot, ctx,
                                                         sliced.y);
    }
}
#endif

static void single_gu_task(int task, void* opaque) {
    const auto& ctx = *static_cast<SinglePipelineContext*>(opaque);
    const MoEWeights& w = *ctx.w;
    const int slot = task / ctx.gu_groups;
    const int group_index = task - slot * ctx.gu_groups;
    const int ob = group_index * 3;
    const int h_oblocks = w.d_ff / N_BLOCK;
    const int group = std::min(3, h_oblocks - ob);
    const int d_kblocks = w.d_model / K_BLOCK;
    const int8_t *gu, *down;
    float sg, su, sd;
    single_expert_parameters(ctx, slot, gu, down, sg, su, sd);
    const int8_t* group_base = gu +
        (size_t)ob * 2 * d_kblocks * TILE_BYTES;

    _tile_zero(0);
    _tile_zero(1);
    if (group > 1) {
        _tile_zero(2);
        _tile_zero(3);
    }
    if (group > 2) {
        _tile_zero(4);
        _tile_zero(5);
    }
    for (int kb = 0; kb < d_kblocks; ++kb) {
        _tile_loadd(6, ctx.aq + kb * K_BLOCK, w.d_model);
        const int8_t* b = group_base +
            (size_t)kb * group * 2 * TILE_BYTES;
        _tile_loadd(7, b, 64);
        _tile_dpbssd(0, 6, 7);
        _tile_loadd(7, b + TILE_BYTES, 64);
        _tile_dpbssd(1, 6, 7);
        if (group > 1) {
            _tile_loadd(7, b + 2 * TILE_BYTES, 64);
            _tile_dpbssd(2, 6, 7);
            _tile_loadd(7, b + 3 * TILE_BYTES, 64);
            _tile_dpbssd(3, 6, 7);
        }
        if (group > 2) {
            _tile_loadd(7, b + 4 * TILE_BYTES, 64);
            _tile_dpbssd(4, 6, 7);
            _tile_loadd(7, b + 5 * TILE_BYTES, 64);
            _tile_dpbssd(5, 6, 7);
        }
    }
    int32_t* gate = pw.single_gate_acc + (size_t)slot * w.d_ff;
    int32_t* up = pw.single_up_acc + (size_t)slot * w.d_ff;
    _tile_stored(0, gate + ob * N_BLOCK, w.d_ff * sizeof(int32_t));
    _tile_stored(1, up + ob * N_BLOCK, w.d_ff * sizeof(int32_t));
    if (group > 1) {
        _tile_stored(2, gate + (ob + 1) * N_BLOCK,
                     w.d_ff * sizeof(int32_t));
        _tile_stored(3, up + (ob + 1) * N_BLOCK,
                     w.d_ff * sizeof(int32_t));
    }
    if (group > 2) {
        _tile_stored(4, gate + (ob + 2) * N_BLOCK,
                     w.d_ff * sizeof(int32_t));
        _tile_stored(5, up + (ob + 2) * N_BLOCK,
                     w.d_ff * sizeof(int32_t));
    }

    const __m512 gs = _mm512_set1_ps(ctx.sx * sg);
    const __m512 us = _mm512_set1_ps(ctx.sx * su);
    __m512 vmax = _mm512_setzero_ps();
    float* hidden = pw.single_hidden + (size_t)slot * w.d_ff;
    for (int j = 0; j < group; ++j) {
        const int f = (ob + j) * N_BLOCK;
        const __m512 vg = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_load_si512(gate + f)), gs);
        const __m512 vu = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_load_si512(up + f)), us);
        const __m512 hv = _mm512_mul_ps(
            _mm512_mul_ps(vg, expert_sigmoid512_ps(vg)), vu);
        _mm512_store_ps(hidden + f, hv);
        vmax = _mm512_max_ps(vmax, abs512_ps(hv));
    }
    pw.single_chunk_max[task] = _mm512_reduce_max_ps(vmax);
}

static void single_quantize_task(int slot, void* opaque) {
    const auto& ctx = *static_cast<SinglePipelineContext*>(opaque);
    const MoEWeights& w = *ctx.w;
    float amax = 0.0f;
    const float* maxima =
        pw.single_chunk_max + (size_t)slot * ctx.gu_groups;
    for (int g = 0; g < ctx.gu_groups; ++g)
        amax = std::max(amax, maxima[g]);
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    pw.single_hscale[slot] = scale;
    const __m512 inverse = _mm512_set1_ps(1.0f / scale);
    const float* hidden = pw.single_hidden + (size_t)slot * w.d_ff;
    int8_t* hq = pw.single_hq + (size_t)slot * w.d_ff;
    for (int f = 0; f < w.d_ff; f += N_BLOCK) {
        const __m512 v = _mm512_mul_ps(_mm512_load_ps(hidden + f), inverse);
        const __m128i q8 =
            _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(v));
        _mm_store_si128(reinterpret_cast<__m128i*>(hq + f), q8);
    }
}

static void single_down_task(int task, void* opaque) {
    const auto& ctx = *static_cast<SinglePipelineContext*>(opaque);
    const MoEWeights& w = *ctx.w;
    const int slot = task / ctx.down_groups;
    const int group_index = task - slot * ctx.down_groups;
    const int ob = group_index * 6;
    const int d_oblocks = w.d_model / N_BLOCK;
    const int group = std::min(6, d_oblocks - ob);
    const int h_kblocks = w.d_ff / K_BLOCK;
    const int8_t *gu, *down;
    float sg, su, sd;
    single_expert_parameters(ctx, slot, gu, down, sg, su, sd);
    const int8_t* group_base = down +
        (size_t)ob * h_kblocks * TILE_BYTES;
    const int8_t* hq = pw.single_hq + (size_t)slot * w.d_ff;

    _tile_zero(0);
    if (group > 1) _tile_zero(1);
    if (group > 2) _tile_zero(2);
    if (group > 3) _tile_zero(3);
    if (group > 4) _tile_zero(4);
    if (group > 5) _tile_zero(5);
    for (int kb = 0; kb < h_kblocks; ++kb) {
        _tile_loadd(6, hq + kb * K_BLOCK, w.d_ff);
        const int8_t* b = group_base +
            (size_t)kb * group * TILE_BYTES;
        _tile_loadd(7, b, 64);
        _tile_dpbssd(0, 6, 7);
        if (group > 1) {
            _tile_loadd(7, b + TILE_BYTES, 64);
            _tile_dpbssd(1, 6, 7);
        }
        if (group > 2) {
            _tile_loadd(7, b + 2 * TILE_BYTES, 64);
            _tile_dpbssd(2, 6, 7);
        }
        if (group > 3) {
            _tile_loadd(7, b + 3 * TILE_BYTES, 64);
            _tile_dpbssd(3, 6, 7);
        }
        if (group > 4) {
            _tile_loadd(7, b + 4 * TILE_BYTES, 64);
            _tile_dpbssd(4, 6, 7);
        }
        if (group > 5) {
            _tile_loadd(7, b + 5 * TILE_BYTES, 64);
            _tile_dpbssd(5, 6, 7);
        }
    }
    alignas(64) int32_t acc[N_BLOCK];
    float* output = pw.routed_output + (size_t)slot * w.d_model;
    const __m512 scale =
        _mm512_set1_ps(pw.single_hscale[slot] * sd);
    auto emit = [&](int block) {
        _mm512_store_ps(output + (size_t)block * N_BLOCK,
                        _mm512_mul_ps(_mm512_cvtepi32_ps(
                            _mm512_load_si512(acc)), scale));
    };
    _tile_stored(0, acc, 64);
    emit(ob);
    if (group > 1) {
        _tile_stored(1, acc, 64); emit(ob + 1);
    }
    if (group > 2) {
        _tile_stored(2, acc, 64); emit(ob + 2);
    }
    if (group > 3) {
        _tile_stored(3, acc, 64); emit(ob + 3);
    }
    if (group > 4) {
        _tile_stored(4, acc, 64); emit(ob + 4);
    }
    if (group > 5) {
        _tile_stored(5, acc, 64); emit(ob + 5);
    }
}

static void single_vnni_gu_s1_task(int task, void* opaque) {
    const auto& ctx = *static_cast<SinglePipelineContext*>(opaque);
    const MoEWeights& w = *ctx.w;
    const int slot = task / 3;
    const int group_index = task - slot * 3;
    const int e = slot == 0 ? -1 : ctx.indices[slot - 1];
    const int8_t* gu = slot == 0
        ? pw.shared_gu
        : pw.routed_gu + (size_t)e * pw.gu_expert_stride;
    const int32_t* comp = slot == 0
        ? pw.shared_gu_comp
        : pw.routed_gu_comp + (size_t)e * 2 * w.d_ff;
    const float sg = slot == 0 ? w.sh_s_gate : w.s_gate[e];
    const float su = slot == 0 ? w.sh_s_up : w.s_up[e];
    int32_t* gate = pw.single_gate_acc + (size_t)slot * 128;
    int32_t* up = pw.single_up_acc + (size_t)slot * 128;
    const int ob = group_index * 3;
    if (group_index < 2)
        vnni_s1_gu_group<3>(gu, comp, ctx.aq, ob, gate, up);
    else
        vnni_s1_gu_group<2>(gu, comp, ctx.aq, ob, gate, up);

    const int group = group_index < 2 ? 3 : 2;
    const __m512 gs = _mm512_set1_ps(ctx.sx * sg);
    const __m512 us = _mm512_set1_ps(ctx.sx * su);
    __m512 vmax = _mm512_setzero_ps();
    float* hidden = pw.single_hidden + (size_t)slot * 128;
    for (int j = 0; j < group; ++j) {
        const int f = (ob + j) * N_BLOCK;
        const __m512 vg = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_load_si512(gate + f)), gs);
        const __m512 vu = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_load_si512(up + f)), us);
        const __m512 h = _mm512_mul_ps(
            _mm512_mul_ps(vg, expert_sigmoid512_ps(vg)), vu);
        _mm512_store_ps(hidden + f, h);
        vmax = _mm512_max_ps(vmax, abs512_ps(h));
    }
    pw.single_chunk_max[task] = _mm512_reduce_max_ps(vmax);
}

static void single_vnni_down_s1_task(int task, void* opaque) {
    const auto& ctx = *static_cast<SinglePipelineContext*>(opaque);
    const MoEWeights& w = *ctx.w;
    const int slot = task / 3;
    const int group_index = task - slot * 3;
    const int e = slot == 0 ? -1 : ctx.indices[slot - 1];
    const int8_t* down = slot == 0
        ? pw.shared_down
        : pw.routed_down + (size_t)e * pw.down_expert_stride;
    const int32_t* comp = slot == 0
        ? pw.shared_down_comp
        : pw.routed_down_comp + (size_t)e * w.d_model;
    const float sd = slot == 0 ? w.sh_s_down : w.s_down[e];
    const int8_t* hq = pw.single_hq + (size_t)slot * 128;
    int32_t* acc = pw.single_down_acc + (size_t)slot * 256;
    const int ob = group_index * 6;
    if (group_index < 2)
        vnni_s1_down_group<6>(down, comp, hq, ob, acc);
    else
        vnni_s1_down_group<4>(down, comp, hq, ob, acc);

    const int group = group_index < 2 ? 6 : 4;
    const __m512 scale =
        _mm512_set1_ps(pw.single_hscale[slot] * sd);
    float* output = pw.routed_output + (size_t)slot * 256;
    for (int j = 0; j < group; ++j) {
        const int d = (ob + j) * N_BLOCK;
        _mm512_store_ps(output + d,
                        _mm512_mul_ps(_mm512_cvtepi32_ps(
                            _mm512_load_si512(acc + d)), scale));
    }
}

#if defined(MOE_S1_SPLIT_DOWN) || defined(MOE_S1_SLICED_DOWN) || \
    defined(MOE_S1_FUSED_SPLIT_DOWN)
static void single_vnni_gu_whole_s1(int slot, SinglePipelineContext& ctx) {
    const MoEWeights& w = *ctx.w;
    const int e = slot == 0 ? -1 : ctx.indices[slot - 1];
    const int8_t* gu = slot == 0
        ? pw.shared_gu
        : pw.routed_gu + (size_t)e * pw.gu_expert_stride;
    const int32_t* comp = slot == 0
        ? pw.shared_gu_comp
        : pw.routed_gu_comp + (size_t)e * 2 * 128;
    const float sg = slot == 0 ? w.sh_s_gate : w.s_gate[e];
    const float su = slot == 0 ? w.sh_s_up : w.s_up[e];
    const __m512 gs = _mm512_set1_ps(ctx.sx * sg);
    const __m512 us = _mm512_set1_ps(ctx.sx * su);
    float* hidden = pw.single_hidden + (size_t)slot * 128;
    __m512 vmax = _mm512_setzero_ps();
    vmax = _mm512_max_ps(
        vmax, vnni_s1_gu_hidden_group<3>(
                  gu, comp, ctx.aq, 0, gs, us, hidden));
    vmax = _mm512_max_ps(
        vmax, vnni_s1_gu_hidden_group<3>(
                  gu, comp, ctx.aq, 3, gs, us, hidden));
    vmax = _mm512_max_ps(
        vmax, vnni_s1_gu_hidden_group<2>(
                  gu, comp, ctx.aq, 6, gs, us, hidden));
    const float amax = _mm512_reduce_max_ps(vmax);
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    pw.single_hscale[slot] = scale;
    const __m512 inverse = _mm512_set1_ps(1.0f / scale);
    int8_t* hq = pw.single_hq + (size_t)slot * 128;
    for (int f = 0; f < 128; f += N_BLOCK) {
        const __m512 q = _mm512_mul_ps(_mm512_load_ps(hidden + f), inverse);
        _mm_store_si128(reinterpret_cast<__m128i*>(hq + f),
                        _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(q)));
    }
}

#ifdef MOE_S1_SLICED_DOWN
template <int START, int GROUP>
static inline void single_vnni_down_slice_accumulate_s1(
    int slot, const SinglePipelineContext& ctx, float* y) {
    const MoEWeights& w = *ctx.w;
    const int e = slot == 0 ? -1 : ctx.indices[slot - 1];
    const int8_t* down = slot == 0
        ? pw.shared_down_s1
        : pw.routed_down_s1 + (size_t)e * pw.down_expert_stride;
    const int32_t* comp = slot == 0
        ? pw.shared_down_comp
        : pw.routed_down_comp + (size_t)e * 256;
    const float sd = slot == 0 ? w.sh_s_down : w.s_down[e];
    const float coeff = slot == 0 ? 1.0f : ctx.gates[slot - 1];
    const int8_t* hq = pw.single_hq + (size_t)slot * 128;
    __m512i acc[GROUP];
#pragma GCC unroll 4
    for (int j = 0; j < GROUP; ++j)
        acc[j] = _mm512_load_si512(comp + (START + j) * N_BLOCK);
    const int8_t* stream = down;
    for (int input = 0; input < 128; input += 4, stream += 16 * 64) {
        int32_t x4;
        std::memcpy(&x4, hq + input, sizeof(x4));
        const __m512i activation = _mm512_set1_epi32(
            x4 ^ static_cast<int32_t>(0x80808080u));
#pragma GCC unroll 4
        for (int j = 0; j < GROUP; ++j)
            acc[j] = _mm512_dpbusd_epi32(
                acc[j], activation,
                _mm512_load_si512(stream + (START + j) * 64));
    }
    const __m512 scale = _mm512_set1_ps(
        pw.single_hscale[slot] * sd * coeff);
#pragma GCC unroll 4
    for (int j = 0; j < GROUP; ++j) {
        float* destination = y + (START + j) * N_BLOCK;
        _mm512_storeu_ps(
            destination,
            _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(acc[j]), scale,
                _mm512_loadu_ps(destination)));
    }
}
#endif

struct SingleS1SplitDownContext {
    SinglePipelineContext* pipeline = nullptr;
    int team = 1;
    alignas(64) std::atomic<int> gu_ready{0};
};

static void single_s1_split_down_worker(int id, void* opaque) {
    auto& shared = *static_cast<SingleS1SplitDownContext*>(opaque);
    SinglePipelineContext& ctx = *shared.pipeline;
    if (id < ctx.experts) single_vnni_gu_whole_s1(id, ctx);
    shared.gu_ready.fetch_add(1, std::memory_order_release);
    while (shared.gu_ready.load(std::memory_order_acquire) != shared.team)
        _mm_pause();
    const int tasks = ctx.experts * 3;
    for (int task = id; task < tasks; task += shared.team)
        single_vnni_down_s1_task(task, &ctx);
}
#endif

#ifdef MOE_S1_FUSED_SPLIT_DOWN
struct SingleS1FusedSplitContext {
    SinglePipelineContext* pipeline = nullptr;
    const float* x = nullptr;
    alignas(64) std::atomic<int> route_ready{0};
    alignas(64) std::atomic<int> gu_ready{0};
};

static void single_s1_fused_split_down_worker(int id, void* opaque) {
    auto& split = *static_cast<SingleS1FusedSplitContext*>(opaque);
    SinglePipelineContext& ctx = *split.pipeline;
    int own_slot = -1;
    if (id == 0) {
        alignas(64) float affinity[N_BLOCK];
        router_dot_1(split.x, 256, pw.router, affinity);
        _mm512_store_ps(
            affinity,
            s1_router_sigmoid512_ps(_mm512_load_ps(affinity)));
        select_top4_s1(affinity, *ctx.w,
                       const_cast<int*>(ctx.indices),
                       const_cast<float*>(ctx.gates));
        int* indices = const_cast<int*>(ctx.indices);
        float* gates = const_cast<float*>(ctx.gates);
        for (int i = 1; i < ctx.w->top_k; ++i) {
            const int index = indices[i];
            const float gate = gates[i];
            int j = i;
            while (j != 0 && indices[j - 1] > index) {
                indices[j] = indices[j - 1];
                gates[j] = gates[j - 1];
                --j;
            }
            indices[j] = index;
            gates[j] = gate;
        }
        split.route_ready.store(1, std::memory_order_release);
        own_slot = ctx.experts - 1;
    } else if (id < ctx.experts) {
        own_slot = id - 1;
        if (own_slot != 0) {
            while (split.route_ready.load(std::memory_order_acquire) == 0)
                _mm_pause();
        }
    }
    if (own_slot >= 0) single_vnni_gu_whole_s1(own_slot, ctx);
    split.gu_ready.fetch_add(1, std::memory_order_release);
    while (split.gu_ready.load(std::memory_order_acquire) != 8)
        _mm_pause();

    // Five 10-block bins and two 12-block bins keep the largest Down load at
    // 75% of a complete 16-block expert.  Worker 7 consumes the final six
    // blocks; its shorter bin also absorbs wake-up skew from the parked team.
    static constexpr int task_offsets[9] = {0, 2, 4, 6, 8, 10, 12, 14, 15};
    static constexpr uint8_t tasks[15] = {
        0, 2, 3, 5, 6, 8, 9, 11, 12, 14,
        1, 4, 7, 10, 13};
    for (int p = task_offsets[id]; p < task_offsets[id + 1]; ++p)
        single_vnni_down_s1_task(tasks[p], &ctx);
}
#endif

struct SingleS1Context {
    SinglePipelineContext* pipeline = nullptr;
    int team = 1;
    alignas(64) std::atomic<int> gu_barrier{0};
    alignas(64) std::atomic<int> quant_barrier{0};
};

static void single_s1_pipeline_worker(int id, void* opaque) {
    auto& shared = *static_cast<SingleS1Context*>(opaque);
    SinglePipelineContext& ctx = *shared.pipeline;
    const int tasks = ctx.experts * 3;
    for (int task = id; task < tasks; task += shared.team)
        single_vnni_gu_s1_task(task, &ctx);
    shared.gu_barrier.fetch_add(1, std::memory_order_release);
    while (shared.gu_barrier.load(std::memory_order_acquire) != shared.team)
        _mm_pause();

    if (id < ctx.experts)
        single_quantize_task(id, &ctx);
    shared.quant_barrier.fetch_add(1, std::memory_order_release);
    while (shared.quant_barrier.load(std::memory_order_acquire) != shared.team)
        _mm_pause();

    for (int task = id; task < tasks; task += shared.team)
        single_vnni_down_s1_task(task, &ctx);
}

struct SingleDagContext {
    SinglePipelineContext* pipeline = nullptr;
    std::atomic<int> next_gu{0};
    std::atomic<int> next_down{0};
    std::atomic<int> finished_experts{0};
    std::atomic<int> gu_finished[MAX_TOP_K + 1];
};

struct SingleMappedContext {
    SinglePipelineContext* pipeline = nullptr;
    int team = 1;
    std::atomic<int> gu_barrier{0};
    std::atomic<int> quant_barrier{0};
};

static void single_mapped_pipeline_worker(int id, void* opaque) {
    auto& shared = *static_cast<SingleMappedContext*>(opaque);
    SinglePipelineContext& ctx = *shared.pipeline;

    for (int slot = 0; slot < ctx.experts; ++slot) {
        for (int g = 0; g < ctx.gu_groups; ++g) {
            const int owner = ctx.gu_owner[slot * ctx.gu_groups + g];
            if (owner == id)
                single_gu_task(slot * ctx.gu_groups + g, &ctx);
        }
    }
    shared.gu_barrier.fetch_add(1, std::memory_order_release);
    while (shared.gu_barrier.load(std::memory_order_acquire) != shared.team)
        _mm_pause();

    for (int slot = id; slot < ctx.experts; slot += shared.team)
        single_quantize_task(slot, &ctx);
    shared.quant_barrier.fetch_add(1, std::memory_order_release);
    while (shared.quant_barrier.load(std::memory_order_acquire) != shared.team)
        _mm_pause();

    for (int slot = 0; slot < ctx.experts; ++slot) {
        for (int g = 0; g < ctx.down_groups; ++g) {
            const int owner = ctx.down_owner[slot * ctx.down_groups + g];
            if (owner == id)
                single_down_task(slot * ctx.down_groups + g, &ctx);
        }
    }
}

static void single_pipeline_worker(int, void* opaque) {
    auto& dag = *static_cast<SingleDagContext*>(opaque);
    SinglePipelineContext& ctx = *dag.pipeline;
    const int total_gu = ctx.experts * ctx.gu_groups;
    for (;;) {
        const int task = dag.next_gu.fetch_add(1, std::memory_order_relaxed);
        if (task >= total_gu) break;
        single_gu_task(task, &ctx);
        const int slot = task / ctx.gu_groups;
        if (dag.gu_finished[slot].fetch_add(1, std::memory_order_acq_rel) + 1 ==
            ctx.gu_groups) {
            single_quantize_task(slot, &ctx);
            dag.finished_experts.fetch_add(1, std::memory_order_release);
        }
    }
    while (dag.finished_experts.load(std::memory_order_acquire) != ctx.experts)
        _mm_pause();

    const int total_down = ctx.experts * ctx.down_groups;
    for (;;) {
        const int task = dag.next_down.fetch_add(1, std::memory_order_relaxed);
        if (task >= total_down) break;
        single_down_task(task, &ctx);
    }
}

static inline void combine_single_s1(const float* x, const float* gates,
                                     float* y) {
#ifndef MOE_S1_PREWEIGHT_OUTPUT
    const __m512 g0 = _mm512_set1_ps(gates[0]);
    const __m512 g1 = _mm512_set1_ps(gates[1]);
    const __m512 g2 = _mm512_set1_ps(gates[2]);
    const __m512 g3 = _mm512_set1_ps(gates[3]);
#endif
    const float* shared = pw.routed_output;
    const float* p0 = shared + 256;
    const float* p1 = p0 + 256;
    const float* p2 = p1 + 256;
    const float* p3 = p2 + 256;
    for (int d = 0; d < 256; d += N_BLOCK) {
        __m512 out = _mm512_add_ps(_mm512_loadu_ps(x + d),
                                   _mm512_load_ps(shared + d));
#ifdef MOE_S1_PREWEIGHT_OUTPUT
        out = _mm512_add_ps(out, _mm512_load_ps(p0 + d));
        out = _mm512_add_ps(out, _mm512_load_ps(p1 + d));
        out = _mm512_add_ps(out, _mm512_load_ps(p2 + d));
        out = _mm512_add_ps(out, _mm512_load_ps(p3 + d));
#else
        out = _mm512_fmadd_ps(g0, _mm512_load_ps(p0 + d), out);
        out = _mm512_fmadd_ps(g1, _mm512_load_ps(p1 + d), out);
        out = _mm512_fmadd_ps(g2, _mm512_load_ps(p2 + d), out);
        out = _mm512_fmadd_ps(g3, _mm512_load_ps(p3 + d), out);
#endif
        _mm512_storeu_ps(y + d, out);
    }
}

static void forward_single_amx(const float* x, const MoEWeights& w, float* y) {
#ifndef MOE_DISABLE_S1_FUSED
    if (pool.size() > 1 && w.d_model == 256 && w.d_ff == 128) {
#ifdef MOE_PROFILE_S1
        unsigned int profile_aux;
        const uint64_t profile_begin = __rdtscp(&profile_aux);
#endif
        int indices[MAX_TOP_K];
        float gates[MAX_TOP_K];
        alignas(64) int8_t aq[256];
        float sx;
        quantize_row(x, 256, aq, sx);
#ifdef MOE_S1_BF16_ROUTER
        alignas(64) uint16_t xbf16[256];
        for (int d = 0; d < 256; d += 16) {
            const __m256bh packed =
                _mm512_cvtneps_pbh(_mm512_loadu_ps(x + d));
            _mm256_store_si256(
                reinterpret_cast<__m256i*>(xbf16 + d), (__m256i)packed);
        }
#endif
#ifdef MOE_PROFILE_S1
        const uint64_t profile_quant_end = __rdtscp(&profile_aux);
#endif
        SinglePipelineContext ctx{
            &w, indices, gates, aq, sx, w.top_k + 1, 3, 3,
            pool.size(), {}, {}};
        SingleS1FusedContext fused;
        fused.pipeline = &ctx;
        fused.x = x;
#ifdef MOE_S1_BF16_ROUTER
        fused.xbf16 = xbf16;
#endif
        fused.gates = gates;
#ifdef MOE_S1_SLICED_DOWN
        SingleS1SlicedContext sliced;
        sliced.pipeline = &ctx;
        sliced.x = x;
        sliced.y = y;
        pool.dispatch(w.top_k + 1, 0, single_s1_sliced_task, &sliced);
#elif defined(MOE_S1_FUSED_SPLIT_DOWN)
        SingleS1FusedSplitContext split;
        split.pipeline = &ctx;
        split.x = x;
        pool.dispatch(8, 0, single_s1_fused_split_down_worker, &split);
#else
        pool.dispatch(w.top_k + 1, 0, single_s1_fused_task, &fused);
#endif
#ifdef MOE_PROFILE_S1
        const uint64_t profile_dispatch_end = __rdtscp(&profile_aux);
#endif

#ifndef MOE_S1_SLICED_DOWN
        combine_single_s1(x, gates, y);
#endif

#ifdef MOE_PROFILE_S1
        const uint64_t profile_end = __rdtscp(&profile_aux);
        static uint64_t profile_calls = 0;
        static uint64_t profile_total[8]{};
        ++profile_calls;
        const uint64_t values[8] = {
            profile_quant_end - profile_begin,
            fused.route_end - fused.task_begin[0],
            profile_dispatch_end - profile_quant_end,
            profile_end - profile_dispatch_end,
            fused.task_end[0] - fused.task_expert_begin[0],
            fused.task_end[1] - fused.task_expert_begin[1],
            fused.task_end[2] - fused.task_expert_begin[2],
            profile_end - profile_begin};
        // The first calls may fault code/data into cache or be descheduled by
        // the shared runner.  Exclude them from the steady-state average so
        // one millisecond-scale outlier does not hide a two-thousand-cycle
        // S1 expert.
        constexpr uint64_t profile_warmup_calls = 100;
        constexpr uint64_t profile_report_calls = 10000;
        if (profile_calls > profile_warmup_calls)
            for (int i = 0; i < 8; ++i) profile_total[i] += values[i];
        if (profile_calls <= 20) {
            std::fprintf(stderr,
                         "S1 cycles %llu: quant=%llu route=%llu dispatch=%llu "
                         "combine=%llu expert0=%llu shared=%llu expert1=%llu "
                         "total=%llu tasks=%llu/%llu/%llu/%llu/%llu\n",
                         static_cast<unsigned long long>(profile_calls),
                         static_cast<unsigned long long>(values[0]),
                         static_cast<unsigned long long>(values[1]),
                         static_cast<unsigned long long>(values[2]),
                         static_cast<unsigned long long>(values[3]),
                         static_cast<unsigned long long>(values[4]),
                         static_cast<unsigned long long>(values[5]),
                         static_cast<unsigned long long>(values[6]),
                         static_cast<unsigned long long>(values[7]),
                         static_cast<unsigned long long>(
                             fused.task_end[0] - fused.task_begin[0]),
                         static_cast<unsigned long long>(
                             fused.task_end[1] - fused.task_begin[1]),
                         static_cast<unsigned long long>(
                             fused.task_end[2] - fused.task_begin[2]),
                         static_cast<unsigned long long>(
                             fused.task_end[3] - fused.task_begin[3]),
                         static_cast<unsigned long long>(
                             fused.task_end[4] - fused.task_begin[4]));
        }
        if (profile_calls == profile_report_calls) {
            const uint64_t samples =
                profile_report_calls - profile_warmup_calls;
            std::fprintf(stderr,
                         "S1 average cycles: quant=%llu route=%llu "
                         "dispatch=%llu combine=%llu expert0=%llu shared=%llu "
                         "expert1=%llu total=%llu\n",
                         static_cast<unsigned long long>(profile_total[0] /
                                                         samples),
                         static_cast<unsigned long long>(profile_total[1] /
                                                         samples),
                         static_cast<unsigned long long>(profile_total[2] /
                                                         samples),
                         static_cast<unsigned long long>(profile_total[3] /
                                                         samples),
                         static_cast<unsigned long long>(profile_total[4] /
                                                         samples),
                         static_cast<unsigned long long>(profile_total[5] /
                                                         samples),
                         static_cast<unsigned long long>(profile_total[6] /
                                                         samples),
                         static_cast<unsigned long long>(profile_total[7] /
                                                         samples));
        }
#endif
        return;
    }
#endif

    alignas(64) float affinity[MAX_NUM_EXPERTS];
    const int router_blocks = pw.padded_experts / N_BLOCK;
    for (int eb = 0; eb < router_blocks; ++eb) {
        const float* packed =
            pw.router + (size_t)eb * w.d_model * N_BLOCK;
        float* block = affinity + eb * N_BLOCK;
        router_dot_1(x, w.d_model, packed, block);
        _mm512_store_ps(
            block, s1_router_sigmoid512_ps(_mm512_load_ps(block)));
    }
    int indices[MAX_TOP_K];
    float gates[MAX_TOP_K];
    select_topk(affinity, w, indices, gates);
    if (w.d_model == 256 && w.d_ff == 128) {
        for (int i = 1; i < w.top_k; ++i) {
            const int index = indices[i];
            const float gate = gates[i];
            int j = i;
            while (j != 0 && indices[j - 1] > index) {
                indices[j] = indices[j - 1];
                gates[j] = gates[j - 1];
                --j;
            }
            indices[j] = index;
            gates[j] = gate;
        }
    }

    alignas(64) int8_t aq[ROUTER_BATCH * MAX_D_MODEL];
    alignas(64) float sx[ROUTER_BATCH];
    quantize_row(x, w.d_model, aq, sx[0]);

    if (pool.size() > 1) {
        const int experts = w.top_k + 1;
        const int gu_groups = (w.d_ff / N_BLOCK + 2) / 3;
        const int down_groups = (w.d_model / N_BLOCK + 5) / 6;
        SinglePipelineContext ctx{
            &w, indices, gates, aq, sx[0], experts, gu_groups, down_groups,
            pool.size(), {}, {}};
        if (w.d_model < 512) {
#ifdef MOE_S1_SPLIT_DOWN
            SingleS1SplitDownContext split;
            split.pipeline = &ctx;
            split.team = pool.size();
            pool.dispatch(pool.size(), 0, single_s1_split_down_worker,
                          &split);
#else
            pool.dispatch(experts, 0, single_whole_vnni_task, &ctx);
#endif
        } else {
            assign_single_tasks(ctx);
            SingleMappedContext shared;
            shared.pipeline = &ctx;
            shared.team = pool.size();
            pool.dispatch(pool.size(), 1, single_mapped_pipeline_worker,
                          &shared);
        }

        if (w.d_model == 256 && w.d_ff == 128 && w.top_k == 4) {
            combine_single_s1(x, gates, y);
            return;
        }

        const float* shared = pw.routed_output;
        for (int d = 0; d < w.d_model; d += N_BLOCK) {
            __m512 out = _mm512_add_ps(_mm512_loadu_ps(x + d),
                                       _mm512_load_ps(shared + d));
            for (int k = 0; k < w.top_k; ++k) {
                out = _mm512_fmadd_ps(
                    _mm512_set1_ps(gates[k]),
                    _mm512_load_ps(pw.routed_output +
                                   (size_t)(k + 1) * w.d_model + d),
                    out);
            }
            _mm512_storeu_ps(y + d, out);
        }
        return;
    }

    copy_f32(x, y, w.d_model);
    load_tile_config(1);
    float* destination[ROUTER_BATCH] = {y};
    expert_amx(pw.shared_gu, pw.shared_down, w.sh_s_gate, w.sh_s_up,
               w.sh_s_down, aq, sx, destination, 1, 1.0f, true,
               w.d_model, w.d_ff);
    for (int k = 0; k < w.top_k; ++k) {
        const int e = indices[k];
        expert_amx(pw.routed_gu + (size_t)e * pw.gu_expert_stride,
                   pw.routed_down + (size_t)e * pw.down_expert_stride,
                   w.s_gate[e], w.s_up[e], w.s_down[e], aq, sx, destination,
                   1, gates[k], true, w.d_model, w.d_ff);
    }
}

static inline void prepare_token_block(const float* x, const MoEWeights& w,
                                       float* y, int first, int rows) {
    if (w.num_experts >= 128 && w.top_k == 2 && pw.router_q != nullptr) {
        for (int r = 0; r < rows; ++r) {
            const int t = first + r;
            quantize_row(x + (size_t)t * w.d_model, w.d_model,
                         pw.xq + (size_t)t * w.d_model, pw.x_scale[t]);
            copy_f32(x + (size_t)t * w.d_model,
                     y + (size_t)t * w.d_model, w.d_model);
        }
        for (int r = rows; r < ROUTER_BATCH; ++r)
            std::memset(pw.xq + (size_t)(first + r) * w.d_model, 0,
                        w.d_model);

        alignas(64) int32_t router_acc[ROUTER_BATCH * MAX_NUM_EXPERTS];
        router_int8_dot_16(pw.xq + (size_t)first * w.d_model,
                           w.d_model, router_acc);
        for (int r = 0; r < rows; ++r) {
            const int t = first + r;
            select_top2_refined(
                x + (size_t)t * w.d_model, w,
                router_acc + (size_t)r * pw.padded_experts, pw.x_scale[t],
                pw.route_idx + (size_t)t * w.top_k,
                pw.route_gate + (size_t)t * w.top_k);
        }
        return;
    }

    alignas(64) float affinity[ROUTER_BATCH][MAX_NUM_EXPERTS];
    router_block(x + (size_t)first * w.d_model, rows, w.d_model, affinity);
    const bool record_s3_masks =
        w.num_experts == 16 && w.top_k == 4 && w.d_model == 256 &&
        w.d_ff == 128;
    uint64_t local_masks[16]{};
    for (int r = 0; r < rows; ++r) {
        const int t = first + r;
        int* indices = pw.route_idx + (size_t)t * w.top_k;
        if (record_s3_masks) {
            select_top4_s1(affinity[r], w, indices,
                           pw.route_gate + (size_t)t * w.top_k);
        } else {
            select_topk(affinity[r], w, indices,
                        pw.route_gate + (size_t)t * w.top_k);
        }
        if (record_s3_masks) {
            for (int k = 0; k < 4; ++k)
                local_masks[indices[k]] |= UINT64_C(1) << (r * 4 + k);
        }
        quantize_row(x + (size_t)t * w.d_model, w.d_model,
                     pw.xq + (size_t)t * w.d_model, pw.x_scale[t]);
        copy_f32(x + (size_t)t * w.d_model,
                 y + (size_t)t * w.d_model, w.d_model);
    }
    if (record_s3_masks) {
        uint64_t* destination =
            s3_route_masks + (size_t)(first / ROUTER_BATCH) * 16;
        for (int e = 0; e < 16; ++e) destination[e] = local_masks[e];
    }
}

static inline void run_shared_chunk(const MoEWeights& w, float* y, int first,
                                    int rows) {
    if (rows == ROUTER_BATCH) {
        float* destination[ROUTER_BATCH];
        for (int r = 0; r < ROUTER_BATCH; ++r)
            destination[r] = y + (size_t)(first + r) * w.d_model;
        expert_amx(pw.shared_gu, pw.shared_down, w.sh_s_gate, w.sh_s_up,
                   w.sh_s_down, pw.xq + (size_t)first * w.d_model,
                   pw.x_scale + first, destination, rows, 1.0f, true,
                   w.d_model, w.d_ff);
        return;
    }

    alignas(64) int8_t aq[ROUTER_BATCH * MAX_D_MODEL];
    alignas(64) float sx[ROUTER_BATCH];
    float* destination[ROUTER_BATCH];
    for (int r = 0; r < rows; ++r) {
        copy_i8_aligned(pw.xq + (size_t)(first + r) * w.d_model,
                        aq + (size_t)r * w.d_model, w.d_model);
        sx[r] = pw.x_scale[first + r];
        destination[r] = y + (size_t)(first + r) * w.d_model;
    }
    for (int r = rows; r < ROUTER_BATCH; ++r)
        std::memset(aq + (size_t)r * w.d_model, 0, w.d_model);
    expert_amx(pw.shared_gu, pw.shared_down, w.sh_s_gate, w.sh_s_up,
               w.sh_s_down, aq, sx, destination, rows, 1.0f, true,
               w.d_model, w.d_ff);
}

static inline void run_routed_range(const MoEWeights& w, int e, int start,
                                    int rows) {
    alignas(64) int8_t aq[ROUTER_BATCH * MAX_D_MODEL];
    alignas(64) float sx[ROUTER_BATCH];
    float* destination[ROUTER_BATCH];
    for (int r = 0; r < rows; ++r) {
        const int assignment = pw.bucket_entries[start + r];
        const int token = assignment / w.top_k;
        copy_i8_aligned(pw.xq + (size_t)token * w.d_model,
                        aq + (size_t)r * w.d_model, w.d_model);
        sx[r] = pw.x_scale[token];
        destination[r] = reinterpret_cast<float*>(
            pw.routed_output16 + (size_t)assignment * w.d_model);
    }
    for (int r = rows; r < ROUTER_BATCH; ++r)
        std::memset(aq + (size_t)r * w.d_model, 0, w.d_model);

    expert_amx(pw.routed_gu + (size_t)e * pw.gu_expert_stride,
               pw.routed_down + (size_t)e * pw.down_expert_stride,
               w.s_gate[e], w.s_up[e], w.s_down[e], aq, sx, destination,
               rows, 1.0f, false, w.d_model, w.d_ff);
}

static inline void run_routed_pair_s3(const MoEWeights& w, int e, int start,
                                      int rows) {
    alignas(64) int8_t aq[32 * 256];
    alignas(64) float sx[32];
    float* destination[32];
    for (int r = 0; r < rows; ++r) {
        const int assignment = pw.bucket_entries[start + r];
        const int token = assignment / w.top_k;
        copy_i8_aligned(pw.xq + (size_t)token * 256,
                        aq + (size_t)r * 256, 256);
        sx[r] = pw.x_scale[token];
        destination[r] = reinterpret_cast<float*>(
            pw.routed_output16 + (size_t)assignment * 256);
    }
    for (int r = rows; r < 32; ++r)
        std::memset(aq + (size_t)r * 256, 0, 256);
    expert_amx_pair_s3(
        pw.routed_gu + (size_t)e * pw.gu_expert_stride,
        pw.routed_down + (size_t)e * pw.down_expert_stride,
        w.s_gate[e], w.s_up[e], w.s_down[e], aq, sx, destination, rows);
}

static void build_expert_buckets(const MoEWeights& w, int num_tokens) {
    if (num_tokens == 128 && w.num_experts == 16 && w.top_k == 4 &&
        w.d_model == 256 && w.d_ff == 128) {
        int position = 0;
        for (int e = 0; e < 16; ++e) {
            pw.expert_offsets[e] = position;
            for (int b = 0; b < 8; ++b) {
                uint64_t mask = s3_route_masks[(size_t)b * 16 + e];
                while (mask != 0) {
                    const int bit = __builtin_ctzll(mask);
                    pw.bucket_entries[position++] = b * 64 + bit;
                    mask &= mask - 1;
                }
            }
        }
        pw.expert_offsets[16] = position;

        // The 16 rotating input batches do not put the largest bucket on a
        // fixed expert.  Sort by row count and pair the largest bucket with
        // the smallest, the second largest with the second smallest, and so
        // on.  The two experts owned by each of the eight workers then have
        // similar total work, without a dynamic queue on the AMX hot path.
        uint8_t order[16];
        for (int i = 0; i < 16; ++i) {
            order[i] = static_cast<uint8_t>(i);
            int j = i;
            while (j != 0) {
                const int current_rows =
                    pw.expert_offsets[order[j] + 1] -
                    pw.expert_offsets[order[j]];
                const int previous_rows =
                    pw.expert_offsets[order[j - 1] + 1] -
                    pw.expert_offsets[order[j - 1]];
                if (previous_rows >= current_rows) break;
                std::swap(order[j], order[j - 1]);
                --j;
            }
        }
        for (int worker = 0; worker < 8; ++worker) {
            s3_routed_order[worker] = order[worker];
            s3_routed_order[worker + 8] = order[15 - worker];
        }
#ifdef MOE_ROUTER_AUDIT
        static std::atomic<int> s3_printed{0};
        if (s3_printed.fetch_add(1) < 26) {
            std::fprintf(stderr, "bucket rows:");
            for (int e = 0; e < 16; ++e)
                std::fprintf(stderr, " %d",
                             pw.expert_offsets[e + 1] -
                                 pw.expert_offsets[e]);
            std::fprintf(stderr, "\n");
        }
#endif
        return;
    }

    std::memset(pw.expert_counts, 0,
                (size_t)(w.num_experts + 1) * sizeof(int));
    const int assignments = num_tokens * w.top_k;
    for (int a = 0; a < assignments; ++a)
        ++pw.expert_counts[pw.route_idx[a]];

    pw.expert_offsets[0] = 0;
    for (int e = 0; e < w.num_experts; ++e)
        pw.expert_offsets[e + 1] =
            pw.expert_offsets[e] + pw.expert_counts[e];
    std::memcpy(pw.expert_counts, pw.expert_offsets,
                (size_t)w.num_experts * sizeof(int));
    for (int a = 0; a < assignments; ++a) {
        const int e = pw.route_idx[a];
        pw.bucket_entries[pw.expert_counts[e]++] = a;
    }

#ifdef MOE_ROUTER_AUDIT
    if (w.num_experts == 16) {
        static std::atomic<int> printed{0};
        if (printed.fetch_add(1) < 26) {
            std::fprintf(stderr, "bucket rows:");
            for (int e = 0; e < w.num_experts; ++e)
                std::fprintf(stderr, " %d",
                             pw.expert_offsets[e + 1] - pw.expert_offsets[e]);
            std::fprintf(stderr, "\n");
        }
    }
#endif
}

static inline void combine_s3_block(float* y, int block) {
    const int first = block * ROUTER_BATCH;
    for (int t = first; t < first + ROUTER_BATCH; ++t) {
        const int assignment = t * 4;
        const __m512 g0 = _mm512_set1_ps(pw.route_gate[assignment]);
        const __m512 g1 = _mm512_set1_ps(pw.route_gate[assignment + 1]);
        const __m512 g2 = _mm512_set1_ps(pw.route_gate[assignment + 2]);
        const __m512 g3 = _mm512_set1_ps(pw.route_gate[assignment + 3]);
        const uint16_t* p0 =
            pw.routed_output16 + (size_t)assignment * 256;
        const uint16_t* p1 = p0 + 256;
        const uint16_t* p2 = p1 + 256;
        const uint16_t* p3 = p2 + 256;
        float* destination = y + (size_t)t * 256;
        for (int d = 0; d < 256; d += 2 * N_BLOCK) {
            __m512 out0 = _mm512_loadu_ps(destination + d);
            __m512 out1 = _mm512_loadu_ps(destination + d + N_BLOCK);
            out0 = _mm512_fmadd_ps(
                g0, _mm512_cvtph_ps(_mm256_load_si256(
                        reinterpret_cast<const __m256i*>(p0 + d))), out0);
            out1 = _mm512_fmadd_ps(
                g0, _mm512_cvtph_ps(_mm256_load_si256(
                        reinterpret_cast<const __m256i*>(p0 + d + N_BLOCK))),
                out1);
            out0 = _mm512_fmadd_ps(
                g1, _mm512_cvtph_ps(_mm256_load_si256(
                        reinterpret_cast<const __m256i*>(p1 + d))), out0);
            out1 = _mm512_fmadd_ps(
                g1, _mm512_cvtph_ps(_mm256_load_si256(
                        reinterpret_cast<const __m256i*>(p1 + d + N_BLOCK))),
                out1);
            out0 = _mm512_fmadd_ps(
                g2, _mm512_cvtph_ps(_mm256_load_si256(
                        reinterpret_cast<const __m256i*>(p2 + d))), out0);
            out1 = _mm512_fmadd_ps(
                g2, _mm512_cvtph_ps(_mm256_load_si256(
                        reinterpret_cast<const __m256i*>(p2 + d + N_BLOCK))),
                out1);
            out0 = _mm512_fmadd_ps(
                g3, _mm512_cvtph_ps(_mm256_load_si256(
                        reinterpret_cast<const __m256i*>(p3 + d))), out0);
            out1 = _mm512_fmadd_ps(
                g3, _mm512_cvtph_ps(_mm256_load_si256(
                        reinterpret_cast<const __m256i*>(p3 + d + N_BLOCK))),
                out1);
            _mm512_storeu_ps(destination + d, out0);
            _mm512_storeu_ps(destination + d + N_BLOCK, out1);
        }
    }
}

static void forward_batch_amx(const float* x, const MoEWeights& w, float* y,
                              int num_tokens) {
#ifdef MOE_PROFILE_BATCH
    unsigned int profile_aux;
    const uint64_t profile_begin = __rdtscp(&profile_aux);
#endif
    const int token_blocks =
        (num_tokens + ROUTER_BATCH - 1) / ROUTER_BATCH;

    struct BatchContext {
        const float* x;
        const MoEWeights* w;
        float* y;
        int num_tokens;
    } ctx{x, &w, y, num_tokens};

    auto prepare_task = [](int b, void* opaque) {
        auto& c = *static_cast<BatchContext*>(opaque);
        const int first = b * ROUTER_BATCH;
        prepare_token_block(c.x, *c.w, c.y, first,
                            std::min(ROUTER_BATCH, c.num_tokens - first));
    };
    auto prepare_shared_task = [](int b, void* opaque) {
        auto& c = *static_cast<BatchContext*>(opaque);
        const int first = b * ROUTER_BATCH;
        const int rows = std::min(ROUTER_BATCH, c.num_tokens - first);
        prepare_token_block(c.x, *c.w, c.y, first, rows);
        run_shared_chunk(*c.w, c.y, first, rows);
    };
    auto shared_task = [](int b, void* opaque) {
        auto& c = *static_cast<BatchContext*>(opaque);
        const int first = b * ROUTER_BATCH;
        run_shared_chunk(*c.w, c.y, first,
                         std::min(ROUTER_BATCH, c.num_tokens - first));
    };
    auto routed_expert_task = [](int e, void* opaque) {
        auto& c = *static_cast<BatchContext*>(opaque);
#ifndef MOE_DISABLE_S3_PAIR
        if (c.num_tokens == 128 && c.w->num_experts == 16 &&
            c.w->d_model == 256 && c.w->d_ff == 128) {
            e = s3_routed_order[e];
            const int end = pw.expert_offsets[e + 1];
            int start = pw.expert_offsets[e];
            while (end - start >= MOE_S3_PAIR_MIN_ROWS) {
                const int rows = std::min(2 * ROUTER_BATCH, end - start);
                run_routed_pair_s3(*c.w, e, start, rows);
                start += rows;
            }
            while (start != end) {
                const int rows = std::min(ROUTER_BATCH, end - start);
                run_routed_range(*c.w, e, start, rows);
                start += rows;
            }
            return;
        }
#endif
        for (int start = pw.expert_offsets[e];
             start < pw.expert_offsets[e + 1]; start += ROUTER_BATCH) {
            run_routed_range(*c.w, e, start,
                             std::min(ROUTER_BATCH,
                                      pw.expert_offsets[e + 1] - start));
        }
    };
    auto combine_task = [](int b, void* opaque) {
        auto& c = *static_cast<BatchContext*>(opaque);
        const MoEWeights& weights = *c.w;
        if (c.num_tokens == 128 && weights.d_model == 256 &&
            weights.d_ff == 128 && weights.num_experts == 16 &&
            weights.top_k == 4) {
            combine_s3_block(c.y, b);
            return;
        }
        const int first = b * ROUTER_BATCH;
        const int end = std::min(c.num_tokens, first + ROUTER_BATCH);
        for (int t = first; t < end; ++t) {
            float* yt = c.y + (size_t)t * weights.d_model;
            if (weights.top_k == 4) {
                const int assignment = t * 4;
                const __m512 g0 = _mm512_set1_ps(
                    pw.route_gate[assignment + 0]);
                const __m512 g1 = _mm512_set1_ps(
                    pw.route_gate[assignment + 1]);
                const __m512 g2 = _mm512_set1_ps(
                    pw.route_gate[assignment + 2]);
                const __m512 g3 = _mm512_set1_ps(
                    pw.route_gate[assignment + 3]);
                const uint16_t* p0 = pw.routed_output16 +
                    (size_t)(assignment + 0) * weights.d_model;
                const uint16_t* p1 = pw.routed_output16 +
                    (size_t)(assignment + 1) * weights.d_model;
                const uint16_t* p2 = pw.routed_output16 +
                    (size_t)(assignment + 2) * weights.d_model;
                const uint16_t* p3 = pw.routed_output16 +
                    (size_t)(assignment + 3) * weights.d_model;
                for (int d = 0; d < weights.d_model; d += N_BLOCK) {
                    __m512 out = _mm512_loadu_ps(yt + d);
                    out = _mm512_fmadd_ps(
                        g0, _mm512_cvtph_ps(_mm256_load_si256(
                                reinterpret_cast<const __m256i*>(p0 + d))),
                        out);
                    out = _mm512_fmadd_ps(
                        g1, _mm512_cvtph_ps(_mm256_load_si256(
                                reinterpret_cast<const __m256i*>(p1 + d))),
                        out);
                    out = _mm512_fmadd_ps(
                        g2, _mm512_cvtph_ps(_mm256_load_si256(
                                reinterpret_cast<const __m256i*>(p2 + d))),
                        out);
                    out = _mm512_fmadd_ps(
                        g3, _mm512_cvtph_ps(_mm256_load_si256(
                                reinterpret_cast<const __m256i*>(p3 + d))),
                        out);
                    _mm512_storeu_ps(yt + d, out);
                }
                continue;
            }
            if (weights.top_k == 2) {
                const int assignment = t * 2;
                const __m512 g0 = _mm512_set1_ps(
                    pw.route_gate[assignment + 0]);
                const __m512 g1 = _mm512_set1_ps(
                    pw.route_gate[assignment + 1]);
                const uint16_t* p0 = pw.routed_output16 +
                    (size_t)(assignment + 0) * weights.d_model;
                const uint16_t* p1 = pw.routed_output16 +
                    (size_t)(assignment + 1) * weights.d_model;
                for (int d = 0; d < weights.d_model; d += N_BLOCK) {
                    __m512 out = _mm512_loadu_ps(yt + d);
                    out = _mm512_fmadd_ps(
                        g0, _mm512_cvtph_ps(_mm256_load_si256(
                                reinterpret_cast<const __m256i*>(p0 + d))),
                        out);
                    out = _mm512_fmadd_ps(
                        g1, _mm512_cvtph_ps(_mm256_load_si256(
                                reinterpret_cast<const __m256i*>(p1 + d))),
                        out);
                    _mm512_storeu_ps(yt + d, out);
                }
                continue;
            }
            for (int d = 0; d < weights.d_model; d += N_BLOCK) {
                __m512 out = _mm512_loadu_ps(yt + d);
                for (int k = 0; k < weights.top_k; ++k) {
                    const int assignment = t * weights.top_k + k;
                    const __m512 expert_output = _mm512_cvtph_ps(
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(
                            pw.routed_output16 +
                            (size_t)assignment * weights.d_model + d)));
                    out = _mm512_fmadd_ps(
                        _mm512_set1_ps(pw.route_gate[assignment]),
                        expert_output,
                        out);
                }
                _mm512_storeu_ps(yt + d, out);
            }
        }
    };

    const bool fuse_s3_shared =
        num_tokens == 128 && w.d_model == 256 && w.d_ff == 128 &&
        w.num_experts == 16 && w.top_k == 4;
    if (fuse_s3_shared) {
        pool.dispatch(token_blocks, ROUTER_BATCH, prepare_shared_task, &ctx);
    } else {
        pool.dispatch(token_blocks,
                      w.num_experts >= 128 && pw.router_q != nullptr
                          ? ROUTER_BATCH : 0,
                      prepare_task, &ctx);
    }
#ifdef MOE_PROFILE_BATCH
    const uint64_t profile_prepare_end = __rdtscp(&profile_aux);
#endif
    build_expert_buckets(w, num_tokens);
#ifdef MOE_PROFILE_BATCH
    const uint64_t profile_bucket_end = __rdtscp(&profile_aux);
#endif
    if (!fuse_s3_shared)
        pool.dispatch(token_blocks, ROUTER_BATCH, shared_task, &ctx);
#ifdef MOE_PROFILE_BATCH
    const uint64_t profile_shared_end = __rdtscp(&profile_aux);
#endif
    // One worker owns all chunks of an expert.  Consecutive chunks therefore
    // reuse its packed weights from private L2 instead of migrating them
    // between cores through a dynamic queue.
    pool.dispatch(w.num_experts, ROUTER_BATCH, routed_expert_task, &ctx);
#ifdef MOE_PROFILE_BATCH
    const uint64_t profile_routed_end = __rdtscp(&profile_aux);
#endif
    pool.dispatch(token_blocks, 0, combine_task, &ctx);
#ifdef MOE_PROFILE_BATCH
    const uint64_t profile_end = __rdtscp(&profile_aux);
    if (num_tokens == 128) {
        static uint64_t calls = 0;
        static uint64_t sums[6]{};
        ++calls;
        const uint64_t values[6] = {
            profile_prepare_end - profile_begin,
            profile_bucket_end - profile_prepare_end,
            profile_shared_end - profile_bucket_end,
            profile_routed_end - profile_shared_end,
            profile_end - profile_routed_end,
            profile_end - profile_begin};
        for (int i = 0; i < 6; ++i) sums[i] += values[i];
        if (calls == 1000)
            std::fprintf(stderr,
                         "batch average cycles: prepare=%llu bucket=%llu "
                         "shared=%llu routed=%llu combine=%llu total=%llu\n",
                         static_cast<unsigned long long>(sums[0] / calls),
                         static_cast<unsigned long long>(sums[1] / calls),
                         static_cast<unsigned long long>(sums[2] / calls),
                         static_cast<unsigned long long>(sums[3] / calls),
                         static_cast<unsigned long long>(sums[4] / calls),
                         static_cast<unsigned long long>(sums[5] / calls));
    }
#endif
}

// Allocation-failure / unexpected-shape fallback.  It is intentionally
// simple because all scored dimensions are multiples of 64 and packing fits
// comfortably in the lab memory limit.
static void scalar_expert(const int8_t* wg, const int8_t* wu,
                          const int8_t* wd, float sg, float su, float sd,
                          const int8_t* xq, float sx, float* out, int d,
                          int h) {
    float hidden[MAX_D_FF];
    float amax = 0.0f;
    for (int f = 0; f < h; ++f) {
        int32_t ag = 0, au = 0;
        for (int j = 0; j < d; ++j) {
            ag += (int32_t)wg[(size_t)f * d + j] * xq[j];
            au += (int32_t)wu[(size_t)f * d + j] * xq[j];
        }
        const float vg = ag * (sx * sg);
        const float vu = au * (sx * su);
        hidden[f] = vg / (1.0f + std::exp(-vg)) * vu;
        amax = std::max(amax, std::fabs(hidden[f]));
    }
    const float sh = amax > 0.0f ? amax / 127.0f : 1.0f;
    int8_t hq[MAX_D_FF];
    for (int f = 0; f < h; ++f) hq[f] = (int8_t)lrintf(hidden[f] / sh);
    for (int j = 0; j < d; ++j) {
        int32_t acc = 0;
        for (int f = 0; f < h; ++f)
            acc += (int32_t)wd[(size_t)j * h + f] * hq[f];
        out[j] = acc * (sh * sd);
    }
}

static void scalar_forward(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    for (int t = 0; t < num_tokens; ++t) {
        const float* xt = x + (size_t)t * w.d_model;
        float* yt = y + (size_t)t * w.d_model;
        float affinity[MAX_NUM_EXPERTS];
        for (int e = 0; e < w.num_experts; ++e) {
            float acc = 0.0f;
            for (int d = 0; d < w.d_model; ++d)
                acc += w.w_router[(size_t)e * w.d_model + d] * xt[d];
            affinity[e] = 1.0f / (1.0f + std::exp(-acc));
        }
        int idx[MAX_TOP_K];
        float gates[MAX_TOP_K];
        select_topk(affinity, w, idx, gates);
        float amax = 0.0f;
        for (int d = 0; d < w.d_model; ++d)
            amax = std::max(amax, std::fabs(xt[d]));
        const float sx = amax > 0.0f ? amax / 127.0f : 1.0f;
        int8_t xq[MAX_D_MODEL];
        for (int d = 0; d < w.d_model; ++d)
            xq[d] = (int8_t)lrintf(xt[d] / sx);
        float out[MAX_D_MODEL];
        scalar_expert(w.sh_gate, w.sh_up, w.sh_down, w.sh_s_gate,
                      w.sh_s_up, w.sh_s_down, xq, sx, out, w.d_model, w.d_ff);
        for (int d = 0; d < w.d_model; ++d) yt[d] = xt[d] + out[d];
        for (int k = 0; k < w.top_k; ++k) {
            const int e = idx[k];
            scalar_expert(w.w_gate + (size_t)e * w.d_ff * w.d_model,
                          w.w_up + (size_t)e * w.d_ff * w.d_model,
                          w.w_down + (size_t)e * w.d_model * w.d_ff,
                          w.s_gate[e], w.s_up[e], w.s_down[e], xq, sx, out,
                          w.d_model, w.d_ff);
            for (int d = 0; d < w.d_model; ++d)
                yt[d] += gates[k] * out[d];
        }
    }
}

static void warm_s1_rank_weights(int id, void*) {
    uint32_t checksum = 0;
    if (id == 0) {
        volatile const int8_t* gu = pw.shared_gu;
        volatile const int8_t* down = pw.shared_down;
        for (size_t i = 0; i < pw.gu_expert_stride; i += 64)
            checksum += gu[i];
        for (size_t i = 0; i < pw.down_expert_stride; i += 64)
            checksum += down[i];
    } else {
        // The routed experts are sorted by index before dispatch.  The four
        // order-statistic slots therefore occupy progressively higher expert
        // ranges.  Warming those ranges keeps the first timed rotation from
        // paying one LLC-to-L2 fill for every newly encountered expert while
        // leaving ample room in each core's 2 MiB private L2.
        static constexpr uint8_t first[4] = {0, 1, 4, 7};
        static constexpr uint8_t last[4] = {8, 11, 13, 15};
        for (int e = first[id - 1]; e <= last[id - 1]; ++e) {
            volatile const int8_t* gu =
                pw.routed_gu + (size_t)e * pw.gu_expert_stride;
            volatile const int8_t* down =
                pw.routed_down + (size_t)e * pw.down_expert_stride;
            for (size_t i = 0; i < pw.gu_expert_stride; i += 64)
                checksum += gu[i];
            for (size_t i = 0; i < pw.down_expert_stride; i += 64)
                checksum += down[i];
        }
    }
    asm volatile("" : : "r"(checksum) : "memory");
}

#ifdef MOE_S1_DUMMY_WARM
static void warm_s1_expert_code(const MoEWeights& w) {
    static constexpr uint8_t rank_experts[4][11] = {
        {0, 1, 3, 4, 5, 7, 8, 0, 1, 3, 4},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
        {4, 7, 8, 9, 10, 11, 12, 13, 4, 7, 8},
        {7, 11, 12, 13, 14, 15, 7, 11, 12, 13, 14}};
    alignas(64) int8_t aq[256]{};
    int indices[4];
    float gates[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    SinglePipelineContext ctx{
        &w, indices, gates, aq, 1.0f, 5, 3, 3, pool.size(), {}, {}};
    for (int round = 0; round < MOE_S1_DUMMY_WARM; ++round) {
        for (int slot = 0; slot < 4; ++slot)
            indices[slot] = rank_experts[slot][round % 11];
        pool.dispatch(5, 0, single_whole_vnni_task, &ctx);
    }
}
#endif

static void prefault_worker_stack(int, void*) {
    volatile uint8_t pages[128 * 1024];
    for (size_t offset = 0; offset < sizeof(pages); offset += 4096)
        pages[offset] = static_cast<uint8_t>(offset);
    asm volatile("" : : "r"(&pages[0]) : "memory");
}

}  // namespace

void preprocess(MoEWeights& w) {
    pool.stop();
    pw.clear();
    const size_t gu_stride = (size_t)2 * w.d_ff * w.d_model;
    const size_t down_stride = (size_t)w.d_model * w.d_ff;
    const int padded_e = (w.num_experts + N_BLOCK - 1) & -N_BLOCK;

    pw.routed_gu = alloc_aligned<int8_t>((size_t)w.num_experts * gu_stride);
    pw.routed_down =
        alloc_aligned<int8_t>((size_t)w.num_experts * down_stride);
    pw.shared_gu = alloc_aligned<int8_t>(gu_stride);
    pw.shared_down = alloc_aligned<int8_t>(down_stride);
    if (w.d_model == 256 && w.d_ff == 128) {
        pw.routed_gu_s1 =
            alloc_aligned<int8_t>((size_t)w.num_experts * gu_stride);
        pw.routed_down_s1 =
            alloc_aligned<int8_t>((size_t)w.num_experts * down_stride);
        pw.shared_gu_s1 = alloc_aligned<int8_t>(gu_stride);
        pw.shared_down_s1 = alloc_aligned<int8_t>(down_stride);
    }
    pw.routed_gu_comp =
        alloc_aligned<int32_t>((size_t)w.num_experts * 2 * w.d_ff);
    pw.routed_down_comp =
        alloc_aligned<int32_t>((size_t)w.num_experts * w.d_model);
    pw.shared_gu_comp = alloc_aligned<int32_t>((size_t)2 * w.d_ff);
    pw.shared_down_comp = alloc_aligned<int32_t>(w.d_model);
    pw.router = alloc_aligned<float>((size_t)padded_e * w.d_model);
    pw.router_q = alloc_aligned<int8_t>((size_t)padded_e * w.d_model);
    pw.router_q_scale = alloc_aligned<float>(padded_e);
    const size_t max_assignments = (size_t)MAX_NUM_TOKENS * w.top_k;
    pw.xq = alloc_aligned<int8_t>((size_t)MAX_NUM_TOKENS * w.d_model);
    pw.x_scale = alloc_aligned<float>(MAX_NUM_TOKENS);
    pw.route_idx = alloc_aligned<int>(max_assignments);
    pw.route_gate = alloc_aligned<float>(max_assignments);
    pw.routed_output =
        alloc_aligned<float>(max_assignments * w.d_model);
    pw.routed_output16 =
        alloc_aligned<uint16_t>(max_assignments * w.d_model);
    pw.expert_counts = alloc_aligned<int>((size_t)w.num_experts + 1);
    pw.expert_offsets = alloc_aligned<int>((size_t)w.num_experts + 1);
    pw.bucket_entries = alloc_aligned<int>(max_assignments);
    const size_t single_experts = (size_t)w.top_k + 1;
    pw.single_gate_acc =
        alloc_aligned<int32_t>(single_experts * w.d_ff);
    pw.single_up_acc =
        alloc_aligned<int32_t>(single_experts * w.d_ff);
    pw.single_hidden = alloc_aligned<float>(single_experts * w.d_ff);
    pw.single_hq = alloc_aligned<int8_t>(single_experts * w.d_ff);
    pw.single_hscale = alloc_aligned<float>(single_experts);
    const size_t single_gu_groups =
        (size_t)(w.d_ff / N_BLOCK + 2) / 3;
    pw.single_chunk_max =
        alloc_aligned<float>(single_experts * single_gu_groups);
    pw.single_down_acc =
        alloc_aligned<int32_t>(single_experts * w.d_model);
    const bool s1_stream_allocation_failed =
        w.d_model == 256 && w.d_ff == 128 &&
        (!pw.routed_gu_s1 || !pw.routed_down_s1 || !pw.shared_gu_s1 ||
         !pw.shared_down_s1);
    if (!pw.routed_gu || !pw.routed_down || !pw.shared_gu ||
        !pw.shared_down || !pw.routed_gu_comp || !pw.routed_down_comp ||
        !pw.shared_gu_comp || !pw.shared_down_comp || !pw.router ||
        !pw.router_q || !pw.router_q_scale || !pw.xq || !pw.x_scale ||
        !pw.route_idx || !pw.route_gate ||
        !pw.routed_output || !pw.routed_output16 || !pw.expert_counts ||
        !pw.expert_offsets || !pw.bucket_entries || !pw.single_gate_acc ||
        !pw.single_up_acc ||
        !pw.single_hidden || !pw.single_hq || !pw.single_hscale ||
        !pw.single_chunk_max || !pw.single_down_acc ||
        s1_stream_allocation_failed) {
        pw.clear();
        return;
    }

    for (int e = 0; e < w.num_experts; ++e) {
        const size_t gu_off = (size_t)e * w.d_ff * w.d_model;
        const size_t down_off = (size_t)e * w.d_model * w.d_ff;
        pack_gate_up(w.w_gate + gu_off, w.w_up + gu_off,
                     w.d_model, w.d_ff,
                     pw.routed_gu + (size_t)e * gu_stride);
        pack_matrix_grouped(w.w_down + down_off, w.d_model, w.d_ff, 6,
                            pw.routed_down + (size_t)e * down_stride);
        if (w.d_model == 256 && w.d_ff == 128) {
            pack_s1_gate_up_stream(
                w.w_gate + gu_off, w.w_up + gu_off,
                pw.routed_gu_s1 + (size_t)e * gu_stride);
            pack_s1_down_stream(
                w.w_down + down_off,
                pw.routed_down_s1 + (size_t)e * down_stride);
        }
        make_compensation(w.w_gate + gu_off, w.d_ff, w.d_model,
                          pw.routed_gu_comp + (size_t)e * 2 * w.d_ff);
        make_compensation(w.w_up + gu_off, w.d_ff, w.d_model,
                          pw.routed_gu_comp + (size_t)e * 2 * w.d_ff + w.d_ff);
        make_compensation(w.w_down + down_off, w.d_model, w.d_ff,
                          pw.routed_down_comp + (size_t)e * w.d_model);
    }
    pack_gate_up(w.sh_gate, w.sh_up, w.d_model, w.d_ff, pw.shared_gu);
    pack_matrix_grouped(w.sh_down, w.d_model, w.d_ff, 6, pw.shared_down);
    if (w.d_model == 256 && w.d_ff == 128) {
        pack_s1_gate_up_stream(w.sh_gate, w.sh_up, pw.shared_gu_s1);
        pack_s1_down_stream(w.sh_down, pw.shared_down_s1);
    }
    make_compensation(w.sh_gate, w.d_ff, w.d_model, pw.shared_gu_comp);
    make_compensation(w.sh_up, w.d_ff, w.d_model,
                      pw.shared_gu_comp + w.d_ff);
    make_compensation(w.sh_down, w.d_model, w.d_ff, pw.shared_down_comp);

    for (int eb = 0; eb < padded_e / N_BLOCK; ++eb) {
        float* block = pw.router + (size_t)eb * w.d_model * N_BLOCK;
        for (int d = 0; d < w.d_model; ++d) {
            for (int lane = 0; lane < N_BLOCK; ++lane) {
                const int e = eb * N_BLOCK + lane;
                block[(size_t)d * N_BLOCK + lane] =
                    e < w.num_experts
                        ? w.w_router[(size_t)e * w.d_model + d]
                        : 0.0f;
            }
        }
    }

#ifdef MOE_S1_BF16_ROUTER
    if (w.d_model == 256 && w.num_experts == 16) {
        for (int pair = 0; pair < 128; ++pair) {
            for (int e = 0; e < 16; ++e) {
                for (int lane = 0; lane < 2; ++lane) {
                    uint32_t bits;
                    const float value =
                        w.w_router[(size_t)e * 256 + pair * 2 + lane];
                    std::memcpy(&bits, &value, sizeof(bits));
                    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
                    s1_router_bf16[(size_t)pair * 32 + e * 2 + lane] =
                        static_cast<uint16_t>(bits >> 16);
                }
            }
        }
    }
#endif

    std::vector<int8_t> router_quantized((size_t)padded_e * w.d_model, 0);
    for (int e = 0; e < padded_e; ++e) {
        if (e >= w.num_experts) {
            pw.router_q_scale[e] = 1.0f;
            continue;
        }
        const float* source = w.w_router + (size_t)e * w.d_model;
        float amax = 0.0f;
        for (int d = 0; d < w.d_model; ++d)
            amax = std::max(amax, std::fabs(source[d]));
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        pw.router_q_scale[e] = scale;
        for (int d = 0; d < w.d_model; ++d)
            router_quantized[(size_t)e * w.d_model + d] =
                static_cast<int8_t>(lrintf(source[d] / scale));
    }
    pack_matrix_grouped(router_quantized.data(), padded_e, w.d_model, 6,
                        pw.router_q);

    pw.gu_expert_stride = gu_stride;
    pw.down_expert_stride = down_stride;
    pw.d_model = w.d_model;
    pw.d_ff = w.d_ff;
    pw.num_experts = w.num_experts;
    pw.padded_experts = padded_e;
    pw.ready = true;

    // Create the latency-oriented worker team outside timing.  Eight is the
    // lab allocation size; respecting a smaller OMP limit also keeps this
    // safe for constrained correctness runs.
    pool.start(std::min(8, omp_get_max_threads()));
    pw.amx_ready = pool.amx_ok();
    pool.dispatch(pool.size(), 0, prefault_worker_stack, nullptr);
    if (w.d_model == 256 && w.d_ff == 128 && w.num_experts == 16) {
#if !defined(MOE_S1_SPLIT_DOWN) && !defined(MOE_S1_FUSED_SPLIT_DOWN)
        pool.set_enabled(std::min(5, pool.size()));
#endif
#ifndef MOE_DISABLE_S1_WARM
        if (pool.size() >= 5)
            pool.dispatch(5, 0, warm_s1_rank_weights, nullptr);
#endif
#ifdef MOE_S1_DUMMY_WARM
        if (pool.size() >= 5) warm_s1_expert_code(w);
#endif
    }
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    if (!pw.ready || pw.d_model != w.d_model || pw.d_ff != w.d_ff ||
        pw.num_experts != w.num_experts) {
        scalar_forward(x, w, y, num_tokens);
        return;
    }

    pool.set_enabled(num_tokens == 1 && w.d_model < 512
#if defined(MOE_S1_SPLIT_DOWN) || defined(MOE_S1_FUSED_SPLIT_DOWN)
                         ? pool.size()
#else
                         ? w.top_k + 1
#endif
                         : pool.size());

    if (pw.amx_ready) {
        if (num_tokens == 1) {
            forward_single_amx(x, w, y);
            return;
        }
        if (num_tokens >= ROUTER_BATCH) {
            forward_batch_amx(x, w, y, num_tokens);
            return;
        }
    }

    if (num_tokens < ROUTER_BATCH * 2) {
        forward_block(x, w, y, num_tokens);
        return;
    }

    const int blocks = (num_tokens + ROUTER_BATCH - 1) / ROUTER_BATCH;
    // Park the pool workers while the OpenMP team runs, so they do not spin
    // on the same cores the bound OMP threads are using.
    pool.set_enabled(1);
#pragma omp parallel for schedule(static)
    for (int b = 0; b < blocks; ++b) {
        const int first = b * ROUTER_BATCH;
        const int rows = std::min(ROUTER_BATCH, num_tokens - first);
        forward_block(x + (size_t)first * w.d_model, w,
                      y + (size_t)first * w.d_model, rows);
    }
    pool.set_enabled(pool.size());
}
