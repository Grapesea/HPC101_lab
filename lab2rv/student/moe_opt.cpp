// Optimized W8A8 MoE forward pass.
//
// x86_64/amd64: AVX-512 + AMX implementation tuned for Intel Sapphire
// Rapids (the lab's main task).
//
// Any other ISA (RISC-V in particular): a portable scalar backend that
// mirrors the reference math token-for-token and parallelizes over tokens
// with OpenMP.  See the #else branch at the bottom of this file.

#include "moe.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <omp.h>
#include <vector>

// MOE_FORCE_PORTABLE lets the portable branch be compiled and tested on an
// x86_64 host (used by the lab's local verification; normal builds never
// define it).
#if (defined(__x86_64__) || defined(__amd64__)) && \
    !defined(MOE_FORCE_PORTABLE)

#include <immintrin.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

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

    // Forward workspaces are allocated in preprocess (outside timing).
    int8_t* xq = nullptr;            // [MAX_NUM_TOKENS][D]
    float* x_scale = nullptr;        // [MAX_NUM_TOKENS]
    int* route_idx = nullptr;        // [MAX_NUM_TOKENS][K]
    float* route_gate = nullptr;     // [MAX_NUM_TOKENS][K]
    float* routed_output = nullptr;  // [MAX_NUM_TOKENS*K][D]
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
        std::free(routed_gu_comp);
        std::free(routed_down_comp);
        std::free(shared_gu_comp);
        std::free(shared_down_comp);
        std::free(router);
        std::free(xq);
        std::free(x_scale);
        std::free(route_idx);
        std::free(route_gate);
        std::free(routed_output);
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
        routed_gu_comp = routed_down_comp = nullptr;
        shared_gu_comp = shared_down_comp = nullptr;
        router = nullptr;
        xq = nullptr;
        x_scale = nullptr;
        route_idx = nullptr;
        route_gate = nullptr;
        routed_output = nullptr;
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

class PersistentPool {
public:
    using Task = void (*)(int, void*);

    struct alignas(64) WorkerSignal {
        std::atomic<uint64_t> generation{0};
    };

    void start(int requested_threads) {
        stop();
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        sched_getaffinity(0, sizeof(allowed), &allowed);
        cpus_.clear();
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &allowed)) cpus_.push_back(cpu);
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
    // VRCP14 plus one Newton step is essentially FP32-accurate and has much
    // higher throughput than a vector divide on Sapphire Rapids.
    __m512 reciprocal = _mm512_rcp14_ps(denominator);
    reciprocal = _mm512_mul_ps(
        reciprocal,
        _mm512_fnmadd_ps(denominator, reciprocal, _mm512_set1_ps(2.0f)));
    return reciprocal;
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
                _mm512_mul_ps(vg, expert_sigmoid512_ps(vg)), vu);
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
                float* dst = destination[r] + output_block * N_BLOCK;
                if (accumulate)
                    out = _mm512_add_ps(_mm512_loadu_ps(dst), out);
                _mm512_storeu_ps(dst, out);
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

static __attribute__((noinline)) void vnni_s1_gu_dot(
    const int8_t* gu, const int32_t* comp, const int8_t* xq,
    int32_t* gate_acc, int32_t* up_acc) {
    vnni_s1_gu_group<3>(gu, comp, xq, 0, gate_acc, up_acc);
    vnni_s1_gu_group<3>(gu, comp, xq, 3, gate_acc, up_acc);
    vnni_s1_gu_group<2>(gu, comp, xq, 6, gate_acc, up_acc);
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

static __attribute__((noinline)) void vnni_s1_down_dot(
    const int8_t* down, const int32_t* comp, const int8_t* hq,
    int32_t* output_acc) {
    vnni_s1_down_group<6>(down, comp, hq, 0, output_acc);
    vnni_s1_down_group<6>(down, comp, hq, 6, output_acc);
    vnni_s1_down_group<4>(down, comp, hq, 12, output_acc);
}

static inline void expert_vnni_s1(
    const int8_t* gu, const int8_t* down, const int32_t* gu_comp,
    const int32_t* down_comp, float s_gate, float s_up, float s_down,
    const int8_t* xq, float s_x, float* output) {
    alignas(64) int32_t gate_acc[128];
    alignas(64) int32_t up_acc[128];
    alignas(64) float hidden[128];
    alignas(64) int8_t hq[128];
    alignas(64) int32_t output_acc[256];
    vnni_s1_gu_dot(gu, gu_comp, xq, gate_acc, up_acc);

    const __m512 gs = _mm512_set1_ps(s_x * s_gate);
    const __m512 us = _mm512_set1_ps(s_x * s_up);
    __m512 vmax = _mm512_setzero_ps();
    for (int f = 0; f < 128; f += N_BLOCK) {
        const __m512 vg = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_load_si512(gate_acc + f)), gs);
        const __m512 vu = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_load_si512(up_acc + f)), us);
        const __m512 h = _mm512_mul_ps(
            _mm512_mul_ps(vg, expert_sigmoid512_ps(vg)), vu);
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

    vnni_s1_down_dot(down, down_comp, hq, output_acc);
    const __m512 scale = _mm512_set1_ps(sh * s_down);
    for (int d = 0; d < 256; d += N_BLOCK)
        _mm512_store_ps(output + d,
                        _mm512_mul_ps(_mm512_cvtepi32_ps(
                            _mm512_load_si512(output_acc + d)), scale));
}

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
            _mm512_store_ps(v, sigmoid512_ps(_mm512_load_ps(v)));
        }
    }
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
        std::memcpy(yr, xr, (size_t)w.d_model * sizeof(float));

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
        ? pw.shared_gu
        : pw.routed_gu + (size_t)e * pw.gu_expert_stride;
    const int8_t* down = slot == 0
        ? pw.shared_down
        : pw.routed_down + (size_t)e * pw.down_expert_stride;
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
        expert_vnni_s1(gu, down, gu_comp, down_comp, sg, su, sd, ctx.aq,
                       ctx.sx, output);
    } else {
        std::memset(output, 0, (size_t)w.d_model * sizeof(float));
        expert_vnni(gu, down, gu_comp, down_comp, sg, su, sd, ctx.aq,
                    ctx.sx, 1.0f, output, w.d_model, w.d_ff);
    }
}

struct SingleS1FusedContext {
    SinglePipelineContext* pipeline = nullptr;
    const float* x = nullptr;
    float* gates = nullptr;
    alignas(64) std::atomic<int> route_ready{0};
};

// The shared expert is independent of routing.  Run it immediately on one
// worker while the main core computes the router; the other four workers
// consume the selected routed experts as soon as their indices are published.
// This removes router latency from the critical path without another pool
// launch or a global barrier.
static void single_s1_fused_task(int id, void* opaque) {
    auto& fused = *static_cast<SingleS1FusedContext*>(opaque);
    SinglePipelineContext& ctx = *fused.pipeline;
    if (id == 0) {
        alignas(64) float affinity[ROUTER_BATCH][MAX_NUM_EXPERTS];
        router_block(fused.x, 1, ctx.w->d_model, affinity);
        select_topk(affinity[0], *ctx.w,
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
        fused.route_ready.store(1, std::memory_order_release);
        return;
    }

    const int slot = id - 1;
    if (slot != 0) {
        while (fused.route_ready.load(std::memory_order_acquire) == 0)
            _mm_pause();
    }
    single_whole_vnni_task(slot, &ctx);
}

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

static void forward_single_amx(const float* x, const MoEWeights& w, float* y) {
    if (pool.size() > 1 && w.d_model == 256 && w.d_ff == 128) {
        alignas(64) int8_t aq[256];
        float sx;
        quantize_row(x, 256, aq, sx);
        int indices[MAX_TOP_K];
        float gates[MAX_TOP_K];
        SinglePipelineContext ctx{
            &w, indices, aq, sx, w.top_k + 1, 3, 3,
            pool.size(), {}, {}};
        SingleS1FusedContext fused;
        fused.pipeline = &ctx;
        fused.x = x;
        fused.gates = gates;
        pool.dispatch(w.top_k + 2, 0, single_s1_fused_task, &fused);

        const float* shared = pw.routed_output;
        for (int d = 0; d < 256; d += N_BLOCK) {
            __m512 out = _mm512_add_ps(_mm512_loadu_ps(x + d),
                                       _mm512_load_ps(shared + d));
            for (int k = 0; k < w.top_k; ++k) {
                out = _mm512_fmadd_ps(
                    _mm512_set1_ps(gates[k]),
                    _mm512_load_ps(pw.routed_output +
                                   (size_t)(k + 1) * 256 + d),
                    out);
            }
            _mm512_storeu_ps(y + d, out);
        }
        return;
    }

    alignas(64) float affinity[ROUTER_BATCH][MAX_NUM_EXPERTS];
    router_block(x, 1, w.d_model, affinity);
    int indices[MAX_TOP_K];
    float gates[MAX_TOP_K];
    select_topk(affinity[0], w, indices, gates);

    alignas(64) int8_t aq[ROUTER_BATCH * MAX_D_MODEL];
    alignas(64) float sx[ROUTER_BATCH];
    quantize_row(x, w.d_model, aq, sx[0]);

    if (pool.size() > 1) {
        const int experts = w.top_k + 1;
        const int gu_groups = (w.d_ff / N_BLOCK + 2) / 3;
        const int down_groups = (w.d_model / N_BLOCK + 5) / 6;
        SinglePipelineContext ctx{
            &w, indices, aq, sx[0], experts, gu_groups, down_groups,
            pool.size(), {}, {}};
        if (w.d_model < 512) {
            pool.dispatch(experts, 0, single_whole_vnni_task, &ctx);
        } else {
            assign_single_tasks(ctx);
            SingleMappedContext shared;
            shared.pipeline = &ctx;
            shared.team = pool.size();
            pool.dispatch(pool.size(), 1, single_mapped_pipeline_worker,
                          &shared);
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

    std::memcpy(y, x, (size_t)w.d_model * sizeof(float));
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
    alignas(64) float affinity[ROUTER_BATCH][MAX_NUM_EXPERTS];
    router_block(x + (size_t)first * w.d_model, rows, w.d_model, affinity);
    for (int r = 0; r < rows; ++r) {
        const int t = first + r;
        select_topk(affinity[r], w, pw.route_idx + (size_t)t * w.top_k,
                    pw.route_gate + (size_t)t * w.top_k);
        quantize_row(x + (size_t)t * w.d_model, w.d_model,
                     pw.xq + (size_t)t * w.d_model, pw.x_scale[t]);
        std::memcpy(y + (size_t)t * w.d_model,
                    x + (size_t)t * w.d_model,
                    (size_t)w.d_model * sizeof(float));
    }
}

static inline void run_shared_chunk(const MoEWeights& w, float* y, int first,
                                    int rows) {
    alignas(64) int8_t aq[ROUTER_BATCH * MAX_D_MODEL];
    alignas(64) float sx[ROUTER_BATCH];
    float* destination[ROUTER_BATCH];
    for (int r = 0; r < rows; ++r) {
        std::memcpy(aq + (size_t)r * w.d_model,
                    pw.xq + (size_t)(first + r) * w.d_model, w.d_model);
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
        std::memcpy(aq + (size_t)r * w.d_model,
                    pw.xq + (size_t)token * w.d_model, w.d_model);
        sx[r] = pw.x_scale[token];
        destination[r] =
            pw.routed_output + (size_t)assignment * w.d_model;
    }
    for (int r = rows; r < ROUTER_BATCH; ++r)
        std::memset(aq + (size_t)r * w.d_model, 0, w.d_model);

    expert_amx(pw.routed_gu + (size_t)e * pw.gu_expert_stride,
               pw.routed_down + (size_t)e * pw.down_expert_stride,
               w.s_gate[e], w.s_up[e], w.s_down[e], aq, sx, destination,
               rows, 1.0f, false, w.d_model, w.d_ff);
}

static void build_expert_buckets(const MoEWeights& w, int num_tokens) {
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

}

static void forward_batch_amx(const float* x, const MoEWeights& w, float* y,
                              int num_tokens) {
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
    auto shared_task = [](int b, void* opaque) {
        auto& c = *static_cast<BatchContext*>(opaque);
        const int first = b * ROUTER_BATCH;
        run_shared_chunk(*c.w, c.y, first,
                         std::min(ROUTER_BATCH, c.num_tokens - first));
    };
    auto routed_expert_task = [](int e, void* opaque) {
        auto& c = *static_cast<BatchContext*>(opaque);
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
        const int first = b * ROUTER_BATCH;
        const int end = std::min(c.num_tokens, first + ROUTER_BATCH);
        for (int t = first; t < end; ++t) {
            float* yt = c.y + (size_t)t * weights.d_model;
            for (int d = 0; d < weights.d_model; d += N_BLOCK) {
                __m512 out = _mm512_loadu_ps(yt + d);
                for (int k = 0; k < weights.top_k; ++k) {
                    const int assignment = t * weights.top_k + k;
                    out = _mm512_fmadd_ps(
                        _mm512_set1_ps(pw.route_gate[assignment]),
                        _mm512_load_ps(pw.routed_output +
                                       (size_t)assignment * weights.d_model + d),
                        out);
                }
                _mm512_storeu_ps(yt + d, out);
            }
        }
    };

    pool.dispatch(token_blocks, 0, prepare_task, &ctx);
    build_expert_buckets(w, num_tokens);
    pool.dispatch(token_blocks, ROUTER_BATCH, shared_task, &ctx);
    // One worker owns all chunks of an expert.  Consecutive chunks therefore
    // reuse its packed weights from private L2 instead of migrating them
    // between cores through the dynamic OpenMP queue.
    pool.dispatch(w.num_experts, ROUTER_BATCH, routed_expert_task, &ctx);
    pool.dispatch(token_blocks, 0, combine_task, &ctx);
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
    pw.routed_gu_comp =
        alloc_aligned<int32_t>((size_t)w.num_experts * 2 * w.d_ff);
    pw.routed_down_comp =
        alloc_aligned<int32_t>((size_t)w.num_experts * w.d_model);
    pw.shared_gu_comp = alloc_aligned<int32_t>((size_t)2 * w.d_ff);
    pw.shared_down_comp = alloc_aligned<int32_t>(w.d_model);
    pw.router = alloc_aligned<float>((size_t)padded_e * w.d_model);
    const size_t max_assignments = (size_t)MAX_NUM_TOKENS * w.top_k;
    pw.xq = alloc_aligned<int8_t>((size_t)MAX_NUM_TOKENS * w.d_model);
    pw.x_scale = alloc_aligned<float>(MAX_NUM_TOKENS);
    pw.route_idx = alloc_aligned<int>(max_assignments);
    pw.route_gate = alloc_aligned<float>(max_assignments);
    pw.routed_output =
        alloc_aligned<float>(max_assignments * w.d_model);
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
    if (!pw.routed_gu || !pw.routed_down || !pw.shared_gu ||
        !pw.shared_down || !pw.routed_gu_comp || !pw.routed_down_comp ||
        !pw.shared_gu_comp || !pw.shared_down_comp || !pw.router || !pw.xq ||
        !pw.x_scale || !pw.route_idx || !pw.route_gate ||
        !pw.routed_output || !pw.expert_counts || !pw.expert_offsets ||
        !pw.bucket_entries || !pw.single_gate_acc || !pw.single_up_acc ||
        !pw.single_hidden || !pw.single_hq || !pw.single_hscale ||
        !pw.single_chunk_max || !pw.single_down_acc) {
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
        make_compensation(w.w_gate + gu_off, w.d_ff, w.d_model,
                          pw.routed_gu_comp + (size_t)e * 2 * w.d_ff);
        make_compensation(w.w_up + gu_off, w.d_ff, w.d_model,
                          pw.routed_gu_comp + (size_t)e * 2 * w.d_ff + w.d_ff);
        make_compensation(w.w_down + down_off, w.d_model, w.d_ff,
                          pw.routed_down_comp + (size_t)e * w.d_model);
    }
    pack_gate_up(w.sh_gate, w.sh_up, w.d_model, w.d_ff, pw.shared_gu);
    pack_matrix_grouped(w.sh_down, w.d_model, w.d_ff, 6, pw.shared_down);
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
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    if (!pw.ready || pw.d_model != w.d_model || pw.d_ff != w.d_ff ||
        pw.num_experts != w.num_experts) {
        scalar_forward(x, w, y, num_tokens);
        return;
    }

    pool.set_enabled(num_tokens == 1 && w.d_model < 512
                         ? w.top_k + 2
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
#pragma omp parallel for schedule(static)
    for (int b = 0; b < blocks; ++b) {
        const int first = b * ROUTER_BATCH;
        const int rows = std::min(ROUTER_BATCH, num_tokens - first);
        forward_block(x + (size_t)first * w.d_model, w,
                      y + (size_t)first * w.d_model, rows);
    }
}

#else  // !x86_64 (or MOE_FORCE_PORTABLE): portable backend for RISC-V and
       // other ISAs.

// ---------------------------------------------------------------------------
// RISC-V backend.
//
// The AVX-512/AMX code above is Intel-specific.  On RISC-V the hot kernels
// are reimplemented with RVV 1.0 intrinsics (Muse Pi Pro: VLEN=256, so a
// 16-lane tile is the working width we use):
//   * s8 x s8 -> s32 GEMM (the dominant hotspot): weights are pre-transposed
//     into 16-row tiles in preprocess; each K step sign-extends the 16 weight
//     bytes to s16 once and keeps 8 token accumulators live (LMUL=2), so
//     vwmacc.vx issues 16 MACs per instruction and every weight byte is
//     reused by 8 tokens before it leaves L2.
//   * router: fp32 dot with vfmacc.vf over 16-expert tiles + vectorized
//     sigmoid (the same Cephes-style exp polynomial as the x86 path).
//   * quantization: vfcvt (RNE) + saturating vnclip chain, and a vector
//     vfredmax for the per-row scale.
// If the compiler has no RVV (-march without v), the scalar backend at the
// bottom of this block is used instead.
// ---------------------------------------------------------------------------

namespace {

// One SwiGLU expert on one already-quantized token: scalar reference math
// (identical formulas to src/moe_ref.cpp), used as the fallback.
static void riscv_expert_ffn(const int8_t* w_gate, const int8_t* w_up,
                             const int8_t* w_down, float s_gate, float s_up,
                             float s_down, const int8_t* xq, float s_x,
                             float* out, int d_model, int d_ff) {
    float h[MAX_D_FF];
    float h_amax = 0.0f;
    for (int f = 0; f < d_ff; ++f) {
        int32_t acc_g = 0;
        int32_t acc_u = 0;
        for (int d = 0; d < d_model; ++d) {
            acc_g += (int32_t)w_gate[(size_t)f * d_model + d] * (int32_t)xq[d];
            acc_u += (int32_t)w_up[(size_t)f * d_model + d] * (int32_t)xq[d];
        }
        const float vg = (float)acc_g * (s_x * s_gate);
        const float vu = (float)acc_u * (s_x * s_up);
        const float silu = vg / (1.0f + expf(-vg));
        h[f] = silu * vu;
        const float a = fabsf(h[f]);
        if (a > h_amax) h_amax = a;
    }
    const float s_h = (h_amax > 0.0f) ? h_amax / 127.0f : 1.0f;
    int8_t hq[MAX_D_FF];
    for (int f = 0; f < d_ff; ++f) hq[f] = (int8_t)lrintf(h[f] / s_h);
    for (int d = 0; d < d_model; ++d) {
        int32_t acc = 0;
        for (int f = 0; f < d_ff; ++f) {
            acc += (int32_t)w_down[(size_t)d * d_ff + f] * (int32_t)hq[f];
        }
        out[d] = (float)acc * (s_h * s_down);
    }
}

static void riscv_forward_scalar(const float* x, const MoEWeights& w,
                                 float* y, int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;

#pragma omp parallel for schedule(static)
    for (int t = 0; t < num_tokens; ++t) {
        const float* xt = x + (size_t)t * d_model;
        float* yt = y + (size_t)t * d_model;
        float s[MAX_NUM_EXPERTS];
        for (int e = 0; e < num_experts; ++e) {
            float acc = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                acc += w.w_router[(size_t)e * d_model + d] * xt[d];
            }
            s[e] = 1.0f / (1.0f + expf(-acc));
        }
        int topk_idx[MAX_TOP_K];
        bool used[MAX_NUM_EXPERTS] = {};
        for (int k = 0; k < top_k; ++k) {
            int best = -1;
            for (int e = 0; e < num_experts; ++e) {
                if (used[e]) continue;
                if (best < 0 || s[e] + w.bias[e] > s[best] + w.bias[best]) {
                    best = e;
                }
            }
            used[best] = true;
            topk_idx[k] = best;
        }
        float gate_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) gate_sum += s[topk_idx[k]];
        float x_amax = 0.0f;
        for (int d = 0; d < d_model; ++d) {
            const float a = fabsf(xt[d]);
            if (a > x_amax) x_amax = a;
        }
        const float s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f;
        int8_t xq[MAX_D_MODEL];
        for (int d = 0; d < d_model; ++d) xq[d] = (int8_t)lrintf(xt[d] / s_x);
        float o[MAX_D_MODEL];
        riscv_expert_ffn(w.sh_gate, w.sh_up, w.sh_down, w.sh_s_gate,
                         w.sh_s_up, w.sh_s_down, xq, s_x, o, d_model, d_ff);
        for (int d = 0; d < d_model; ++d) yt[d] = xt[d] + o[d];
        for (int k = 0; k < top_k; ++k) {
            const int e = topk_idx[k];
            const float gate = s[e] / gate_sum;
            riscv_expert_ffn(
                w.w_gate + (size_t)e * d_ff * d_model,
                w.w_up + (size_t)e * d_ff * d_model,
                w.w_down + (size_t)e * d_model * d_ff, w.s_gate[e],
                w.s_up[e], w.s_down[e], xq, s_x, o, d_model, d_ff);
            for (int d = 0; d < d_model; ++d) yt[d] += gate * o[d];
        }
    }
}

}  // namespace

#if defined(__riscv_v)

#include <riscv_vector.h>

namespace {

constexpr int RVV_TB = 16;      // tokens per processing block
constexpr int RVV_FB = 16;      // output rows per s8 tile (= vector lanes)
constexpr int RVV_EB = 16;      // router experts per tile
constexpr int RVV_TLIVE = 8;    // live token accumulators per GEMM sweep
constexpr size_t RVV_VL = 16;   // active lanes for every RVV kernel

template <typename T>
static T* rvv_alloc(size_t count) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, count * sizeof(T)) != 0) return nullptr;
    return static_cast<T*>(p);
}

// Packed, preprocessed weights (built once in preprocess, outside timing).
struct RvvPacked {
    int8_t* gu = nullptr;         // [E+1][2][d_ff/16][d_model][16]
    int8_t* down = nullptr;       // [E+1][d_model/16][d_ff][16]
    float* router = nullptr;      // [padded_e/16][d_model][16]
    int8_t* xq = nullptr;         // [MAX_NUM_TOKENS][d_model]
    float* x_scale = nullptr;     // [MAX_NUM_TOKENS]
    int* route_idx = nullptr;     // [MAX_NUM_TOKENS][top_k]
    float* route_gate = nullptr;  // [MAX_NUM_TOKENS][top_k]
    int* expert_counts = nullptr; // [E+1]
    int* expert_offsets = nullptr;// [E+1]
    int* bucket_entries = nullptr;// [MAX_NUM_TOKENS*top_k]
    float* routed_output = nullptr;  // [MAX_NUM_TOKENS*top_k][d_model]
    float* sh_out = nullptr;         // [MAX_NUM_TOKENS][d_model]
    size_t gu_expert = 0;
    size_t down_expert = 0;
    int d_model = 0, d_ff = 0;
    int num_experts = 0, padded_e = 0;
    bool rvv_ok = false;
    bool ready = false;

    void clear() {
        std::free(gu);
        std::free(down);
        std::free(router);
        std::free(xq);
        std::free(x_scale);
        std::free(route_idx);
        std::free(route_gate);
        std::free(expert_counts);
        std::free(expert_offsets);
        std::free(bucket_entries);
        std::free(routed_output);
        std::free(sh_out);
        gu = down = nullptr;
        router = nullptr;
        xq = nullptr;
        x_scale = nullptr;
        route_idx = nullptr;
        route_gate = nullptr;
        expert_counts = nullptr;
        expert_offsets = nullptr;
        bucket_entries = nullptr;
        routed_output = nullptr;
        sh_out = nullptr;
        gu_expert = down_expert = 0;
        d_model = d_ff = num_experts = padded_e = 0;
        rvv_ok = false;
        ready = false;
    }
    ~RvvPacked() { clear(); }
};

RvvPacked rv;

// Per-thread scratch for the expert kernels (kept out of the hot stack
// frame; thread-local so OpenMP expert iterations never race).
static thread_local int32_t rvv_gu_acc[2 * RVV_TB * MAX_D_FF];
static thread_local int32_t rvv_down_acc[RVV_TB * MAX_D_MODEL];
static thread_local int8_t rvv_hq[RVV_TB * MAX_D_FF];
static thread_local float rvv_hscale[RVV_TB];
static thread_local float rvv_hidden[MAX_D_FF];

// ---------- packing (once, in preprocess) ----------

// Transpose every 16-row block: dst[(fb, k)][lane] = src[fb*16+lane][k].
// Gate tiles occupy the first half of the buffer, up tiles the second half,
// so each tile has a uniform 16-byte K stride.
static void pack_gu_tiles(const int8_t* gate, const int8_t* up, int rows,
                          int cols, int8_t* dst) {
    const int fb = rows / RVV_FB;
    const size_t half = (size_t)fb * cols * RVV_FB;
    for (int b = 0; b < fb; ++b)
        for (int k = 0; k < cols; ++k)
            for (int l = 0; l < RVV_FB; ++l) {
                dst[(size_t)(b * cols + k) * RVV_FB + l] =
                    gate[(size_t)(b * RVV_FB + l) * cols + k];
                dst[half + (size_t)(b * cols + k) * RVV_FB + l] =
                    up[(size_t)(b * RVV_FB + l) * cols + k];
            }
}

static void pack_down_tiles(const int8_t* w, int rows, int cols, int8_t* dst) {
    const int fb = rows / RVV_FB;
    for (int b = 0; b < fb; ++b)
        for (int k = 0; k < cols; ++k)
            for (int l = 0; l < RVV_FB; ++l)
                dst[((size_t)b * cols + k) * RVV_FB + l] =
                    w[(size_t)(b * RVV_FB + l) * cols + k];
}

static void pack_router_tiles(const float* src, int rows, int cols, int padded,
                              float* dst) {
    const int eb = padded / RVV_EB;
    for (int b = 0; b < eb; ++b)
        for (int k = 0; k < cols; ++k)
            for (int l = 0; l < RVV_EB; ++l) {
                const int e = b * RVV_EB + l;
                dst[((size_t)b * cols + k) * RVV_EB + l] =
                    e < rows ? src[(size_t)e * cols + k] : 0.0f;
            }
}

// ---------- RVV kernels ----------

// Cephes-style vector exp (same polynomial as the x86 AVX-512 path).
static inline vfloat32m2_t rvv_exp(vfloat32m2_t x, size_t vl) {
    x = __riscv_vfmin_vf_f32m2(x, 88.3762626647949f, vl);
    x = __riscv_vfmax_vf_f32m2(x, -88.3762626647949f, vl);
    vfloat32m2_t fx = __riscv_vfmacc_vf_f32m2(
        __riscv_vfmv_v_f_f32m2(0.5f, vl), 1.44269504088896341f, x, vl);
    vint32m2_t fxi = __riscv_vfcvt_x_f_v_i32m2(fx, vl);
    vfloat32m2_t fxf = __riscv_vfcvt_f_x_v_f32m2(fxi, vl);
    // vfnmsac: vd = vd - rs1*vs2  (RVV's vfnmacc negates the whole sum).
    x = __riscv_vfnmsac_vf_f32m2(x, 0.693359375f, fxf, vl);
    x = __riscv_vfnmsac_vf_f32m2(x, -2.12194440e-4f, fxf, vl);
    vfloat32m2_t z = __riscv_vfmul_vv_f32m2(x, x, vl);
    vfloat32m2_t y = __riscv_vfmv_v_f_f32m2(1.9875691500E-4f, vl);
    // Horner form: y = y*x + c  (== vfmacc_vv(c, y, x)).
    y = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.3981999507E-3f, vl),
                                y, x, vl);
    y = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(8.3334519073E-3f, vl),
                                y, x, vl);
    y = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(4.1665795894E-2f, vl),
                                y, x, vl);
    y = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.6666665459E-1f, vl),
                                y, x, vl);
    y = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(5.0000001201E-1f, vl),
                                y, x, vl);
    y = __riscv_vfmacc_vv_f32m2(x, y, z, vl);  // y = y*z + x
    y = __riscv_vfadd_vf_f32m2(y, 1.0f, vl);   // y = y*z + x + 1
    vint32m2_t emm = __riscv_vadd_vx_i32m2(fxi, 127, vl);
    emm = __riscv_vsll_vx_i32m2(emm, 23, vl);
    return __riscv_vfmul_vv_f32m2(
        y, __riscv_vreinterpret_v_i32m2_f32m2(emm), vl);
}

static inline vfloat32m2_t rvv_sigmoid(vfloat32m2_t x, size_t vl) {
    vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
    vfloat32m2_t d = __riscv_vfadd_vv_f32m2(
        one, rvv_exp(__riscv_vfneg_v_f32m2(x, vl), vl), vl);
    return __riscv_vfdiv_vv_f32m2(one, d, vl);
}

// Per-token symmetric int8 quantization: scale = max|x|/127, RNE rounding,
// saturating narrowing i32 -> i16 -> i8.
static void rvv_quantize_row(const float* x, int n, int8_t* q, float& scale) {
    const size_t vl = RVV_VL;
    vfloat32m1_t maxv = __riscv_vfmv_v_f_f32m1(0.0f, 8);
    for (int i = 0; i < n; i += RVV_FB) {
        vfloat32m2_t v = __riscv_vle32_v_f32m2(x + i, vl);
        maxv = __riscv_vfredmax_vs_f32m2_f32m1(
            __riscv_vfabs_v_f32m2(v, vl), maxv, vl);
    }
    const float amax = __riscv_vfmv_f_s_f32m1_f32(maxv);
    scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / scale;
    for (int i = 0; i < n; i += RVV_FB) {
        vfloat32m2_t v = __riscv_vfmul_vf_f32m2(
            __riscv_vle32_v_f32m2(x + i, vl), inv, vl);
        vint32m2_t vi = __riscv_vfcvt_x_f_v_i32m2(v, vl);
        vint16m1_t v16 = __riscv_vnclip_wx_i16m1(vi, 0, __RISCV_VXRM_RNU, vl);
        vint8mf2_t v8 = __riscv_vnclip_wx_i8mf2(v16, 0, __RISCV_VXRM_RNU, vl);
        __riscv_vse8_v_i8mf2(q + i, v8, vl);
    }
}

// One 16-row weight tile: out[t][f_off + lane] += sum_k w[k][lane] * xq[row][k].
// token_ids == nullptr means the rows are contiguous 0..t_count-1.
// The tile bytes are sign-extended to s16 once per K step; vwmacc.vx keeps
// RVV_TLIVE token accumulators (LMUL=2, 16 int32 lanes each) live, so 16 MACs
// issue per instruction and the weight vector is reused by all live tokens.
static inline void gemm_tile_s8(const int8_t* w, const int8_t* xq,
                                const int* token_ids, int k_dim, int t_count,
                                int32_t* out, int f_off, int out_stride) {
    const size_t vl = RVV_VL;
    const int groups = (t_count + RVV_TLIVE - 1) / RVV_TLIVE;
    for (int g = 0; g < groups; ++g) {
        vint32m2_t a0 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t a1 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t a2 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t a3 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t a4 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t a5 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t a6 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t a7 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t* acc[RVV_TLIVE] = {&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7};
        for (int k = 0; k < k_dim; ++k) {
            vint8mf2_t w8 = __riscv_vle8_v_i8mf2(w + (size_t)k * RVV_FB, vl);
            vint16m1_t w16 = __riscv_vsext_vf2_i16m1(w8, vl);
            for (int t = 0; t < RVV_TLIVE; ++t) {
                const int idx = std::min(g * RVV_TLIVE + t, t_count - 1);
                const int row = token_ids ? token_ids[idx] : idx;
                *acc[t] = __riscv_vwmacc_vx_i32m2(
                    *acc[t], xq[(size_t)row * k_dim + k], w16, vl);
            }
        }
        for (int t = 0; t < RVV_TLIVE; ++t)
            __riscv_vse32_v_i32m2(
                out + (size_t)(g * RVV_TLIVE + t) * out_stride + f_off,
                *acc[t], vl);
    }
}

// Router: affinity[t][e] = sigmoid(sum_d x[t][d] * r[e][d]) over 16-expert
// tiles; the packed router is [eb][d][16] so one vle32 loads 16 experts.
static void rvv_router_block(const float* x, int d_model, int num_experts,
                             int t_count, float* affinity) {
    const size_t vl = RVV_VL;
    const int eb = num_experts / RVV_EB;
    for (int b = 0; b < eb; ++b) {
        const int groups = (t_count + RVV_TLIVE - 1) / RVV_TLIVE;
        for (int g = 0; g < groups; ++g) {
            vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t a1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t a2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t a3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t a4 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t a5 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t a6 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t a7 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t* acc[RVV_TLIVE] = {&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7};
            for (int d = 0; d < d_model; ++d) {
                vfloat32m2_t wv = __riscv_vle32_v_f32m2(
                    rv.router + ((size_t)b * d_model + d) * RVV_EB, vl);
                for (int t = 0; t < RVV_TLIVE; ++t) {
                    const int idx = std::min(g * RVV_TLIVE + t, t_count - 1);
                    *acc[t] = __riscv_vfmacc_vf_f32m2(
                        *acc[t], x[(size_t)idx * d_model + d], wv, vl);
                }
            }
            for (int t = 0; t < RVV_TLIVE; ++t) {
                const int idx = std::min(g * RVV_TLIVE + t, t_count - 1);
                vfloat32m2_t s = rvv_sigmoid(*acc[t], vl);
                __riscv_vse32_v_f32m2(
                    affinity + (size_t)idx * MAX_NUM_EXPERTS + b * RVV_EB,
                    s, vl);
            }
        }
    }
    // Scalar tail for num_experts % 16 != 0 (not hit by the lab shapes).
    for (int e = eb * RVV_EB; e < num_experts; ++e)
        for (int t = 0; t < t_count; ++t) {
            float acc = 0.0f;
            for (int d = 0; d < d_model; ++d)
                acc += x[(size_t)t * d_model + d] *
                       rv.router[((size_t)eb * d_model + d) * RVV_EB +
                                 (e - eb * RVV_EB)];
            affinity[(size_t)t * MAX_NUM_EXPERTS + e] =
                1.0f / (1.0f + expf(-acc));
        }
}

// SwiGLU activation + hidden requantization for one expert call.
// gate_acc/up_acc are [t_count][d_ff] int32; writes hq[t][f] and hscale[t].
static void rvv_swiglu_quantize(const int32_t* gate_acc,
                                const int32_t* up_acc, int t_count, int d_ff,
                                const float* sx, float s_gate, float s_up,
                                int8_t* hq, float* hscale) {
    const size_t vl = RVV_VL;
    for (int t = 0; t < t_count; ++t) {
        vfloat32m1_t maxv = __riscv_vfmv_v_f_f32m1(0.0f, 8);
        for (int f = 0; f < d_ff; f += RVV_FB) {
            vfloat32m2_t vg = __riscv_vfmul_vf_f32m2(
                __riscv_vfcvt_f_x_v_f32m2(
                    __riscv_vle32_v_i32m2(gate_acc + (size_t)t * d_ff + f, vl),
                    vl),
                sx[t] * s_gate, vl);
            vfloat32m2_t vu = __riscv_vfmul_vf_f32m2(
                __riscv_vfcvt_f_x_v_f32m2(
                    __riscv_vle32_v_i32m2(up_acc + (size_t)t * d_ff + f, vl),
                    vl),
                sx[t] * s_up, vl);
            // SwiGLU: h = silu(vg) * vu = vg * sigmoid(vg) * vu.
            vfloat32m2_t hv = __riscv_vfmul_vv_f32m2(
                __riscv_vfmul_vv_f32m2(vg, rvv_sigmoid(vg, vl), vl), vu, vl);
            __riscv_vse32_v_f32m2(rvv_hidden + f, hv, vl);
            maxv = __riscv_vfredmax_vs_f32m2_f32m1(
                __riscv_vfabs_v_f32m2(hv, vl), maxv, vl);
        }
        const float amax = __riscv_vfmv_f_s_f32m1_f32(maxv);
        hscale[t] = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float inv = 1.0f / hscale[t];
        for (int f = 0; f < d_ff; f += RVV_FB) {
            vfloat32m2_t v = __riscv_vfmul_vf_f32m2(
                __riscv_vle32_v_f32m2(rvv_hidden + f, vl), inv, vl);
            vint32m2_t vi = __riscv_vfcvt_x_f_v_i32m2(v, vl);
            vint16m1_t v16 = __riscv_vnclip_wx_i16m1(vi, 0, __RISCV_VXRM_RNU, vl);
            vint8mf2_t v8 = __riscv_vnclip_wx_i8mf2(v16, 0, __RISCV_VXRM_RNU, vl);
            __riscv_vse8_v_i8mf2(hq + (size_t)t * d_ff + f, v8, vl);
        }
    }
}

// Down projection: out_rows[t] += down @ hq[t] * (hscale[t]*s_down*coeff[t]).
static void rvv_down_accumulate(const int8_t* down, const int8_t* hq,
                                int t_count, int k_dim, int out_dim,
                                const float* hscale, float s_down,
                                const float* coeff, float* const* out_rows) {
    const size_t vl = RVV_VL;
    const int fb = out_dim / RVV_FB;
    for (int b = 0; b < fb; ++b) {
        gemm_tile_s8(down + (size_t)b * k_dim * RVV_FB, hq, nullptr, k_dim,
                     t_count, rvv_down_acc, b * RVV_FB, out_dim);
        for (int t = 0; t < t_count; ++t) {
            vfloat32m2_t a = __riscv_vfcvt_f_x_v_f32m2(
                __riscv_vle32_v_i32m2(rvv_down_acc + (size_t)t * out_dim +
                                          b * RVV_FB,
                                      vl),
                vl);
            vfloat32m2_t o = __riscv_vfmul_vf_f32m2(
                a, hscale[t] * s_down * coeff[t], vl);
            float* yt = out_rows[t] + b * RVV_FB;
            __riscv_vse32_v_f32m2(
                yt,
                __riscv_vfadd_vv_f32m2(__riscv_vle32_v_f32m2(yt, vl), o, vl),
                vl);
        }
    }
}

// One expert FFN on up to RVV_TB tokens: gu GEMM -> SwiGLU -> requantize ->
// down GEMM, accumulated into out_rows[t] scaled by coeff[t].  e == -1 means
// the shared expert.
static void rvv_expert_ffn(const MoEWeights& w, int e, const int* token_ids,
                           int t_count, const float* coeff,
                           float* const* out_rows) {
    const int d_model = w.d_model, d_ff = w.d_ff;
    const int8_t* gu = rv.gu + (size_t)(e + 1) * rv.gu_expert;
    const int8_t* down = rv.down + (size_t)(e + 1) * rv.down_expert;
    const float s_gate = e < 0 ? w.sh_s_gate : w.s_gate[e];
    const float s_up = e < 0 ? w.sh_s_up : w.s_up[e];
    const float s_down = e < 0 ? w.sh_s_down : w.s_down[e];
    float sx[RVV_TB];
    for (int t = 0; t < t_count; ++t) sx[t] = rv.x_scale[token_ids[t]];
    const int32_t* up_base = rvv_gu_acc + (size_t)RVV_TB * d_ff;
    const int fb = d_ff / RVV_FB;
    const int8_t* gate = gu;
    const int8_t* up = gu + (size_t)fb * d_model * RVV_FB;
    for (int b = 0; b < fb; ++b) {
        gemm_tile_s8(gate + (size_t)b * d_model * RVV_FB, rv.xq, token_ids,
                     d_model, t_count, rvv_gu_acc,
                     b * RVV_FB, d_ff);
        gemm_tile_s8(up + (size_t)b * d_model * RVV_FB, rv.xq, token_ids,
                     d_model, t_count,
                     const_cast<int32_t*>(up_base), b * RVV_FB, d_ff);
    }
    rvv_swiglu_quantize(rvv_gu_acc, up_base, t_count, d_ff, sx, s_gate, s_up,
                        rvv_hq, rvv_hscale);
    rvv_down_accumulate(down, rvv_hq, t_count, d_ff, d_model, rvv_hscale,
                        s_down, coeff, out_rows);
}

// Top-K selection + gate normalization: same semantics as the reference
// (bias participates in selection only; gates use the raw affinities).
static void select_topk_rvv(const float* affinity, const MoEWeights& w,
                            int* indices, float* gates) {
    const int top_k = w.top_k;
    int topk_idx[MAX_TOP_K];
    bool used[MAX_NUM_EXPERTS] = {};
    for (int k = 0; k < top_k; ++k) {
        int best = -1;
        for (int e = 0; e < w.num_experts; ++e) {
            if (used[e]) continue;
            if (best < 0 || affinity[e] + w.bias[e] > affinity[best] + w.bias[best])
                best = e;
        }
        used[best] = true;
        topk_idx[k] = best;
    }
    float gate_sum = 0.0f;
    for (int k = 0; k < top_k; ++k) gate_sum += affinity[topk_idx[k]];
    for (int k = 0; k < top_k; ++k) {
        indices[k] = topk_idx[k];
        gates[k] = affinity[topk_idx[k]] / gate_sum;
    }
}

static void rvv_prepare_block(const MoEWeights& w, const float* x, int first,
                              int rows) {
    for (int r = 0; r < rows; ++r) {
        const int t = first + r;
        rvv_quantize_row(x + (size_t)t * w.d_model, w.d_model,
                         rv.xq + (size_t)t * w.d_model, rv.x_scale[t]);
    }
    alignas(64) float affinity[RVV_TB][MAX_NUM_EXPERTS];
    rvv_router_block(x + (size_t)first * w.d_model, w.d_model, w.num_experts,
                     rows, &affinity[0][0]);
    for (int r = 0; r < rows; ++r) {
        const int t = first + r;
        select_topk_rvv(affinity[r], w, rv.route_idx + (size_t)t * w.top_k,
                        rv.route_gate + (size_t)t * w.top_k);
    }
}

static void rvv_build_buckets(const MoEWeights& w, int num_tokens) {
    std::memset(rv.expert_counts, 0, (size_t)(w.num_experts + 1) * sizeof(int));
    const int assignments = num_tokens * w.top_k;
    for (int a = 0; a < assignments; ++a) ++rv.expert_counts[rv.route_idx[a]];
    rv.expert_offsets[0] = 0;
    for (int e = 0; e < w.num_experts; ++e)
        rv.expert_offsets[e + 1] = rv.expert_offsets[e] + rv.expert_counts[e];
    std::memcpy(rv.expert_counts, rv.expert_offsets,
                (size_t)w.num_experts * sizeof(int));
    for (int a = 0; a < assignments; ++a) {
        const int e = rv.route_idx[a];
        rv.bucket_entries[rv.expert_counts[e]++] = a;
    }
}

// sh_out / routed_output are accumulated into by rvv_down_accumulate, so the
// rows used by this call must start at zero every forward pass.
static void rvv_zero_outputs(const MoEWeights& w, int num_tokens) {
    const int assignments = num_tokens * w.top_k;
    std::memset(rv.routed_output, 0,
                (size_t)assignments * w.d_model * sizeof(float));
    std::memset(rv.sh_out, 0, (size_t)num_tokens * w.d_model * sizeof(float));
}

// y[t] = x[t] + shared_out[t] + sum_k gate_k * routed_out[t*top_k+k].
static void rvv_combine(const MoEWeights& w, const float* x, float* y,
                        int num_tokens) {
    const size_t vl = RVV_VL;
#pragma omp parallel for schedule(static) if (num_tokens >= 2 * RVV_TB)
    for (int t = 0; t < num_tokens; ++t) {
        const float* xt = x + (size_t)t * w.d_model;
        const float* st = rv.sh_out + (size_t)t * w.d_model;
        float* yt = y + (size_t)t * w.d_model;
        for (int d = 0; d < w.d_model; d += RVV_FB) {
            vfloat32m2_t v = __riscv_vfadd_vv_f32m2(
                __riscv_vle32_v_f32m2(xt + d, vl),
                __riscv_vle32_v_f32m2(st + d, vl), vl);
            for (int k = 0; k < w.top_k; ++k) {
                const int a = t * w.top_k + k;
                // routed_output already carries the per-assignment gate
                // (folded in rvv_down_accumulate via coeff).
                v = __riscv_vfadd_vv_f32m2(
                    v, __riscv_vle32_v_f32m2(
                           rv.routed_output + (size_t)a * w.d_model + d, vl),
                    vl);
            }
            __riscv_vse32_v_f32m2(yt + d, v, vl);
        }
    }
}

static void rvv_forward(const float* x, const MoEWeights& w, float* y,
                        int num_tokens) {
    const int blocks = (num_tokens + RVV_TB - 1) / RVV_TB;

#pragma omp parallel for schedule(static) if (num_tokens >= 2 * RVV_TB)
    for (int b = 0; b < blocks; ++b) {
        const int first = b * RVV_TB;
        const int rows = std::min(RVV_TB, num_tokens - first);
        rvv_prepare_block(w, x, first, rows);
    }
    rvv_build_buckets(w, num_tokens);
    rvv_zero_outputs(w, num_tokens);

    // Shared expert: all tokens, in 16-token chunks.
    for (int b = 0; b < blocks; ++b) {
        const int first = b * RVV_TB;
        const int rows = std::min(RVV_TB, num_tokens - first);
        int ids[RVV_TB];
        float coeff[RVV_TB];
        float* rows_ptr[RVV_TB];
        for (int r = 0; r < rows; ++r) {
            ids[r] = first + r;
            coeff[r] = 1.0f;
            rows_ptr[r] = rv.sh_out + (size_t)(first + r) * w.d_model;
        }
        rvv_expert_ffn(w, -1, ids, rows, coeff, rows_ptr);
    }

    // Routed experts: one OpenMP iteration per expert over its prebuilt
    // bucket.  Each assignment appears in exactly one bucket, so the
    // routed_output rows are written without any race.
#pragma omp parallel for schedule(static) if (num_tokens >= 2 * RVV_TB)
    for (int e = 0; e < w.num_experts; ++e) {
        const int start = rv.expert_offsets[e];
        const int end = rv.expert_offsets[e + 1];
        for (int s = start; s < end; s += RVV_TB) {
            const int cnt = std::min(RVV_TB, end - s);
            int ids[RVV_TB];
            float coeff[RVV_TB];
            float* rows_ptr[RVV_TB];
            for (int i = 0; i < cnt; ++i) {
                const int a = rv.bucket_entries[s + i];
                ids[i] = a / w.top_k;
                coeff[i] = rv.route_gate[a];
                rows_ptr[i] = rv.routed_output + (size_t)a * w.d_model;
            }
            rvv_expert_ffn(w, e, ids, cnt, coeff, rows_ptr);
        }
    }

    rvv_combine(w, x, y, num_tokens);
}

}  // namespace

void preprocess(MoEWeights& w) {
    rv.clear();
    const int d_model = w.d_model, d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int padded_e = (num_experts + RVV_EB - 1) & -RVV_EB;
    rv.gu_expert = (size_t)2 * (d_ff / RVV_FB) * d_model * RVV_FB;
    rv.down_expert = (size_t)(d_model / RVV_FB) * d_ff * RVV_FB;
    rv.gu = rvv_alloc<int8_t>((size_t)(num_experts + 1) * rv.gu_expert);
    rv.down = rvv_alloc<int8_t>((size_t)(num_experts + 1) * rv.down_expert);
    rv.router = rvv_alloc<float>((size_t)padded_e * d_model);
    rv.xq = rvv_alloc<int8_t>((size_t)MAX_NUM_TOKENS * d_model);
    rv.x_scale = rvv_alloc<float>(MAX_NUM_TOKENS);
    const size_t max_assignments = (size_t)MAX_NUM_TOKENS * w.top_k;
    rv.route_idx = rvv_alloc<int>(max_assignments);
    rv.route_gate = rvv_alloc<float>(max_assignments);
    rv.expert_counts = rvv_alloc<int>((size_t)num_experts + 1);
    rv.expert_offsets = rvv_alloc<int>((size_t)num_experts + 1);
    rv.bucket_entries = rvv_alloc<int>(max_assignments);
    rv.routed_output = rvv_alloc<float>(max_assignments * d_model);
    rv.sh_out = rvv_alloc<float>((size_t)MAX_NUM_TOKENS * d_model);
    if (!rv.gu || !rv.down || !rv.router || !rv.xq || !rv.x_scale ||
        !rv.route_idx || !rv.route_gate || !rv.expert_counts ||
        !rv.expert_offsets || !rv.bucket_entries || !rv.routed_output ||
        !rv.sh_out) {
        rv.clear();
        return;
    }
    pack_gu_tiles(w.sh_gate, w.sh_up, d_ff, d_model, rv.gu);
    pack_down_tiles(w.sh_down, d_model, d_ff, rv.down);
    for (int e = 0; e < num_experts; ++e) {
        pack_gu_tiles(w.w_gate + (size_t)e * d_ff * d_model,
                      w.w_up + (size_t)e * d_ff * d_model, d_ff, d_model,
                      rv.gu + (size_t)(e + 1) * rv.gu_expert);
        pack_down_tiles(w.w_down + (size_t)e * d_model * d_ff, d_model, d_ff,
                        rv.down + (size_t)(e + 1) * rv.down_expert);
    }
    pack_router_tiles(w.w_router, num_experts, d_model, padded_e, rv.router);
    rv.d_model = d_model;
    rv.d_ff = d_ff;
    rv.num_experts = num_experts;
    rv.padded_e = padded_e;
    // The kernels below use a fixed 16-lane working width (VLEN=256) and
    // assume d_model/d_ff are multiples of 16.  Fall back to the scalar
    // backend on any other hardware or shape.
    rv.rvv_ok = __riscv_vsetvlmax_e8m1() >= 32 && d_model % RVV_FB == 0 &&
                d_ff % RVV_FB == 0;
    rv.ready = true;
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    if (!rv.ready || !rv.rvv_ok || rv.d_model != w.d_model ||
        rv.d_ff != w.d_ff || rv.num_experts != w.num_experts) {
        riscv_forward_scalar(x, w, y, num_tokens);
        return;
    }
    rvv_forward(x, w, y, num_tokens);
}

#else  // !defined(__riscv_v): scalar backend (also used by MOE_FORCE_PORTABLE
       // on hosts without RVV)

void preprocess(MoEWeights& w) {
    // No repacking needed: the scalar backend consumes the weights in the
    // row-major layout provided by the framework.
    (void)w;
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    riscv_forward_scalar(x, w, y, num_tokens);
}

#endif  // defined(__riscv_v)

#endif  // (defined(__x86_64__) || defined(__amd64__)) &&
        // !defined(MOE_FORCE_PORTABLE)
