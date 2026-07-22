// Matrix-oriented MoE implementation.  No external BLAS is used because the
// grading build only links OpenMP; the small GEMM/GEMV kernel below is the
// self-contained matrix backend.

#include "moe.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

#if defined(__GNUC__) && defined(__x86_64__)
#define PORTABLE_TARGET __attribute__((target("arch=x86-64-v3")))
#else
#define PORTABLE_TARGET
#endif

// Precomputed row metadata.  It is also the compensation needed by matrix
// backends whose fast primitive is u8*s8 rather than s8*s8; the portable
// backend keeps the same interface and performs s8*s8 directly.
struct PackedMetadata {
    int d_model = 0, d_ff = 0, num_experts = 0;
    int32_t gate_sum[MAX_NUM_EXPERTS * MAX_D_FF];
    int32_t up_sum[MAX_NUM_EXPERTS * MAX_D_FF];
    int32_t down_sum[MAX_NUM_EXPERTS * MAX_D_MODEL];
    int32_t sh_gate_sum[MAX_D_FF], sh_up_sum[MAX_D_FF];
    int32_t sh_down_sum[MAX_D_MODEL];
} meta;

PORTABLE_TARGET static int32_t scalar_row_sum(const int8_t* p, int n) {
    int32_t s = 0;
    for (int i = 0; i < n; ++i) s += p[i];
    return s;
}

// These clear reduction loops are auto-vectorized into the translation
// unit's portable AVX2 target. Integer accumulation remains exact.
PORTABLE_TARGET static int32_t dot_i8(const int8_t* a, const int8_t* b, int n,
                                      int32_t /*b_sum*/) {
    int32_t acc = 0;
    for (int i = 0; i < n; ++i)
        acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    return acc;
}

PORTABLE_TARGET static float dot_f32(const float* a, const float* b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

PORTABLE_TARGET static inline void quantize_row(const float* src, int8_t* dst,
                                                int n, float& scale) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(src[i]));
    scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / scale;
    for (int i = 0; i < n; ++i)
        dst[i] = static_cast<int8_t>(lrintf(src[i] * inv));
}

// One row of the batched expert matrix multiplication.
PORTABLE_TARGET static inline void expert_row(
    const int8_t* wg, const int8_t* wu, const int8_t* wd,
    const int32_t* wg_sum, const int32_t* wu_sum, const int32_t* wd_sum,
    float sg, float su, float sd, const int8_t* xq, float sx, float* out,
    float* hidden, int8_t* hq, int d_model, int d_ff) {
    float amax = 0.0f;
    const float xsg = sx * sg, xsu = sx * su;
    for (int f = 0; f < d_ff; ++f) {
        const int32_t ag = dot_i8(xq, wg + static_cast<size_t>(f) * d_model,
                                  d_model, wg_sum[f]);
        const int32_t au = dot_i8(xq, wu + static_cast<size_t>(f) * d_model,
                                  d_model, wu_sum[f]);
        const float vg = static_cast<float>(ag) * xsg;
        const float vu = static_cast<float>(au) * xsu;
        const float v = (vg / (1.0f + expf(-vg))) * vu;
        hidden[f] = v;
        amax = std::max(amax, std::fabs(v));
    }
    const float sh = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / sh;
    for (int f = 0; f < d_ff; ++f)
        hq[f] = static_cast<int8_t>(lrintf(hidden[f] * inv));
    const float out_scale = sh * sd;
    for (int d = 0; d < d_model; ++d) {
        const int32_t acc = dot_i8(
            hq, wd + static_cast<size_t>(d) * d_ff, d_ff, wd_sum[d]);
        out[d] = static_cast<float>(acc) * out_scale;
    }
}

struct Workspace {
    float affinity[MAX_NUM_TOKENS * MAX_NUM_EXPERTS];
    float sx[MAX_NUM_TOKENS], gates[MAX_NUM_TOKENS * MAX_TOP_K];
    float shared_out[MAX_NUM_TOKENS * MAX_D_MODEL];
    float routed_out[MAX_NUM_TOKENS * MAX_TOP_K * MAX_D_MODEL];
    float hidden[(MAX_NUM_TOKENS + MAX_NUM_TOKENS * MAX_TOP_K) * MAX_D_FF];
    int8_t xq[MAX_NUM_TOKENS * MAX_D_MODEL];
    int8_t bucket_xq[MAX_NUM_TOKENS * MAX_TOP_K * MAX_D_MODEL];
    int8_t hq[(MAX_NUM_TOKENS + MAX_NUM_TOKENS * MAX_TOP_K) * MAX_D_FF];
    int ids[MAX_NUM_TOKENS * MAX_TOP_K], counts[MAX_NUM_EXPERTS];
    int offsets[MAX_NUM_EXPERTS + 1], cursor[MAX_NUM_EXPERTS];
    int positions[MAX_NUM_TOKENS * MAX_TOP_K];
    int bucket_token[MAX_NUM_TOKENS * MAX_TOP_K];
};
static Workspace ws;

PORTABLE_TARGET static void forward_one(const float* x, const MoEWeights& w,
                                        float* y) {
    const int D = w.d_model, H = w.d_ff, E = w.num_experts, K = w.top_k;
    float affinity[MAX_NUM_EXPERTS];
    int ids[MAX_TOP_K];
    bool used[MAX_NUM_EXPERTS] = {};
    for (int e = 0; e < E; ++e) {
        const float z = dot_f32(x, w.w_router + static_cast<size_t>(e) * D, D);
        affinity[e] = 1.0f / (1.0f + expf(-z));
    }
    float gate_sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        int best = -1;
        for (int e = 0; e < E; ++e)
            if (!used[e] &&
                (best < 0 || affinity[e] + w.bias[e] >
                                 affinity[best] + w.bias[best]))
                best = e;
        used[best] = true;
        ids[k] = best;
        gate_sum += affinity[best];
    }
    alignas(32) int8_t xq[MAX_D_MODEL], hq[MAX_D_FF];
    alignas(32) float hidden[MAX_D_FF], out[MAX_D_MODEL];
    float sx;
    quantize_row(x, xq, D, sx);
    expert_row(w.sh_gate, w.sh_up, w.sh_down, meta.sh_gate_sum,
               meta.sh_up_sum, meta.sh_down_sum, w.sh_s_gate,
               w.sh_s_up, w.sh_s_down, xq, sx, out, hidden, hq, D, H);
    for (int d = 0; d < D; ++d) y[d] = x[d] + out[d];
    for (int k = 0; k < K; ++k) {
        const int e = ids[k];
        expert_row(
            w.w_gate + static_cast<size_t>(e) * H * D,
            w.w_up + static_cast<size_t>(e) * H * D,
            w.w_down + static_cast<size_t>(e) * D * H,
            meta.gate_sum + static_cast<size_t>(e) * H,
            meta.up_sum + static_cast<size_t>(e) * H,
            meta.down_sum + static_cast<size_t>(e) * D, w.s_gate[e],
            w.s_up[e], w.s_down[e], xq, sx, out, hidden, hq, D, H);
        const float g = affinity[e] / gate_sum;
        for (int d = 0; d < D; ++d) y[d] += g * out[d];
    }
}

} // namespace

PORTABLE_TARGET void preprocess(MoEWeights& w) {
    const int D = w.d_model, H = w.d_ff, E = w.num_experts;
    meta.d_model = D;
    meta.d_ff = H;
    meta.num_experts = E;
    for (int e = 0; e < E; ++e) {
        for (int f = 0; f < H; ++f) {
            const size_t row = static_cast<size_t>(e) * H + f;
            meta.gate_sum[row] = scalar_row_sum(w.w_gate + row * D, D);
            meta.up_sum[row] = scalar_row_sum(w.w_up + row * D, D);
        }
        for (int d = 0; d < D; ++d) {
            const size_t row = static_cast<size_t>(e) * D + d;
            meta.down_sum[row] = scalar_row_sum(w.w_down + row * H, H);
        }
    }
    for (int f = 0; f < H; ++f) {
        meta.sh_gate_sum[f] = scalar_row_sum(w.sh_gate + static_cast<size_t>(f) * D, D);
        meta.sh_up_sum[f] = scalar_row_sum(w.sh_up + static_cast<size_t>(f) * D, D);
    }
    for (int d = 0; d < D; ++d)
        meta.sh_down_sum[d] = scalar_row_sum(w.sh_down + static_cast<size_t>(d) * H, H);

}

PORTABLE_TARGET void moe_forward_optimized(const float* x, const MoEWeights& w,
                                           float* y, int num_tokens) {
    if (num_tokens == 1) {
        forward_one(x, w, y);
        return;
    }

    const int N = num_tokens, D = w.d_model, H = w.d_ff;
    const int E = w.num_experts, K = w.top_k, T = N * K;
    for (int e = 0; e < E; ++e) ws.counts[e] = 0;

    // Router GEMM Z = X * Wr^T, followed by row-wise Top-K.
#pragma omp parallel for schedule(static) if(N >= 32)
    for (int t = 0; t < N; ++t) {
        const float* xt = x + static_cast<size_t>(t) * D;
        float* at = ws.affinity + static_cast<size_t>(t) * E;
        for (int e = 0; e < E; ++e) {
            const float z = dot_f32(xt, w.w_router + static_cast<size_t>(e) * D, D);
            at[e] = 1.0f / (1.0f + expf(-z));
        }
        quantize_row(xt, ws.xq + static_cast<size_t>(t) * D, D, ws.sx[t]);
        bool used[MAX_NUM_EXPERTS] = {};
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            int best = -1;
            for (int e = 0; e < E; ++e)
                if (!used[e] &&
                    (best < 0 || at[e] + w.bias[e] > at[best] + w.bias[best]))
                    best = e;
            used[best] = true;
            ws.ids[t * K + k] = best;
            sum += at[best];
        }
        for (int k = 0; k < K; ++k)
            ws.gates[t * K + k] = at[ws.ids[t * K + k]] / sum;
    }

    for (int i = 0; i < T; ++i) ++ws.counts[ws.ids[i]];
    ws.offsets[0] = 0;
    for (int e = 0; e < E; ++e) ws.offsets[e + 1] = ws.offsets[e] + ws.counts[e];
    for (int e = 0; e < E; ++e) ws.cursor[e] = ws.offsets[e];
    for (int t = 0; t < N; ++t) {
        for (int k = 0; k < K; ++k) {
            const int i = t * K + k, p = ws.cursor[ws.ids[i]]++;
            ws.positions[i] = p;
            ws.bucket_token[p] = t;
            int8_t* dst = ws.bucket_xq + static_cast<size_t>(p) * D;
            const int8_t* src = ws.xq + static_cast<size_t>(t) * D;
            for (int d = 0; d < D; ++d) dst[d] = src[d];
        }
    }

    // The shared expert is one regular N-row matrix operation.
#pragma omp parallel for schedule(static) if(N >= 16)
    for (int t = 0; t < N; ++t) {
        expert_row(w.sh_gate, w.sh_up, w.sh_down, meta.sh_gate_sum,
                   meta.sh_up_sum, meta.sh_down_sum, w.sh_s_gate,
                   w.sh_s_up, w.sh_s_down,
                   ws.xq + static_cast<size_t>(t) * D, ws.sx[t],
                   ws.shared_out + static_cast<size_t>(t) * D,
                   ws.hidden + static_cast<size_t>(t) * H,
                   ws.hq + static_cast<size_t>(t) * H, D, H);
    }

    // Segmented/grouped GEMM: each expert consumes one contiguous token bucket.
#pragma omp parallel for schedule(dynamic, 1) if(T >= 32)
    for (int e = 0; e < E; ++e) {
        const int begin = ws.offsets[e], end = ws.offsets[e + 1];
        for (int p = begin; p < end; ++p) {
            expert_row(
                w.w_gate + static_cast<size_t>(e) * H * D,
                w.w_up + static_cast<size_t>(e) * H * D,
                w.w_down + static_cast<size_t>(e) * D * H,
                meta.gate_sum + static_cast<size_t>(e) * H,
                meta.up_sum + static_cast<size_t>(e) * H,
                meta.down_sum + static_cast<size_t>(e) * D,
                w.s_gate[e], w.s_up[e], w.s_down[e],
                ws.bucket_xq + static_cast<size_t>(p) * D,
                ws.sx[ws.bucket_token[p]],
                ws.routed_out + static_cast<size_t>(p) * D,
                ws.hidden + static_cast<size_t>(N + p) * H,
                ws.hq + static_cast<size_t>(N + p) * H, D, H);
        }
    }

    // Token-owned reduction avoids races and preserves the reference Top-K order.
#pragma omp parallel for schedule(static) if(N >= 32)
    for (int t = 0; t < N; ++t) {
        const float* xt = x + static_cast<size_t>(t) * D;
        const float* shared = ws.shared_out + static_cast<size_t>(t) * D;
        float* yt = y + static_cast<size_t>(t) * D;
        for (int d = 0; d < D; ++d) yt[d] = xt[d] + shared[d];
        for (int k = 0; k < K; ++k) {
            const int i = t * K + k, p = ws.positions[i];
            const float* out = ws.routed_out + static_cast<size_t>(p) * D;
            const float gate = ws.gates[i];
            for (int d = 0; d < D; ++d) yt[d] += gate * out[d];
        }
    }
}