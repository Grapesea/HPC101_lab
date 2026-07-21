// Main task: optimize the MoE forward pass.

#include "moe.h"
#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <cstddef>
#include <cmath>

// WSL2 does not expose AVX-512 to the guest.  -march=sapphirerapids (set by
// CMakeLists.txt) generates AVX-512 instructions that crash on this VM.
// Override down to x86-64-v3 (AVX2 + FMA) so the binary runs everywhere that
// has AVX2, while still allowing the compiler to auto-vectorize with 256-bit
// vectors.  Remove this line on a bare-metal machine with AVX-512 support to
// unlock the full 512-bit width.
#pragma GCC target("arch=x86-64-v3")

void preprocess(MoEWeights& w) {
    (void)w; // suppress unused parameter warning
}

// One SwiGLU expert on one already-quantized token
static void expert_ffn(const int8_t* w_gate, const int8_t* w_up,
                       const int8_t* w_down, float s_gate, float s_up,
                       float s_down, const int8_t* xq, float s_x, float* out,
                       int d_model, int d_ff) {
    // Gate / up projections + SwiGLU activation
    float h[MAX_D_FF];
    float h_amax = 0.0f;

    for (int f = 0; f < d_ff; f++) {
        int32_t acc_g = 0;
        int32_t acc_u = 0;
        for (int d = 0; d < d_model; d++) {
            acc_g += (int32_t)w_gate[f * d_model + d] * (int32_t)xq[d];
            acc_u += (int32_t)w_up[f * d_model + d] * (int32_t)xq[d];
        }
        float vg = (float)acc_g * (s_x * s_gate);
        float vu = (float)acc_u * (s_x * s_up);
        float silu = vg / (1.0f + expf(-vg));
        h[f] = silu * vu;
        float a = fabsf(h[f]);
        if (a > h_amax) h_amax = a;
    }

    // Requantize hidden activation to int8
    float s_h = (h_amax > 0.0f) ? h_amax / 127.0f : 1.0f;
    int8_t hq[MAX_D_FF];
    for (int f = 0; f < d_ff; f++) {
        hq[f] = (int8_t)lrintf(h[f] / s_h);
    }

    // Down projection
    for (int d = 0; d < d_model; d++) {
        int32_t acc = 0;
        for (int f = 0; f < d_ff; f++) {
            acc += (int32_t)w_down[d * d_ff + f] * (int32_t)hq[f];
        }
        out[d] = (float)acc * (s_h * s_down);
    }
}

void moe_forward(const float* x, const MoEWeights& w, float* y,
                     int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;

    for (int t = 0; t < num_tokens; t++) {
        const float* xt = x + (size_t)t * d_model;
        float* yt = y + (size_t)t * d_model;

        // 1. Affinity scores
        float s[MAX_NUM_EXPERTS];

        // 优化：并行打分
        for (int e = 0; e < num_experts; e++) {
            float acc = 0.0f;
            #pragma omp parallel for schedule(static) reduction(+:acc)
            for (int d = 0; d < d_model; d++) {
                acc += w.w_router[(size_t)e * d_model + d] * xt[d];
            }
            s[e] = 1.0f / (1.0f + expf(-acc));
        }

        // 2. Top-K selection by biased score (ties broken by smaller index)
        int topk_idx[MAX_TOP_K];
        bool used[MAX_NUM_EXPERTS] = {};
        for (int k = 0; k < top_k; k++) {
            int best = -1;
            for (int e = 0; e < num_experts; e++) {
                if (used[e]) continue;
                if (best < 0 || s[e] + w.bias[e] > s[best] + w.bias[best]) {
                    best = e;
                }
            }
            used[best] = true;
            topk_idx[k] = best;
        }

        // 3. Gate values: normalize the ORIGINAL affinities of the selected
        //    experts (the bias never enters the gate values)
        float gate_sum = 0.0f;
        for (int k = 0; k < top_k; k++) gate_sum += s[topk_idx[k]];

        // 4. Quantize the token to int8 (symmetric, per-token scale)
        float x_amax = 0.0f;
        #pragma omp parallel for schedule(static) reduction(max:x_amax)
        for (int d = 0; d < d_model; d++) {
            float a = fabsf(xt[d]);
            if (a > x_amax) x_amax = a;
        }
        float s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f;
        int8_t xq[MAX_D_MODEL];
        #pragma omp parallel for schedule(static) 
        for (int d = 0; d < d_model; d++) {
            xq[d] = (int8_t)lrintf(xt[d] / s_x);
        }

        // 5+6. Shared expert (always on), then selected routed experts,
        //      combined on top of the residual connection
        float o[MAX_D_MODEL];
        expert_ffn(w.sh_gate, w.sh_up, w.sh_down, w.sh_s_gate, w.sh_s_up,
                   w.sh_s_down, xq, s_x, o, d_model, d_ff);
        #pragma omp parallel for schedule(static) 
        for (int d = 0; d < d_model; d++) {
            yt[d] = xt[d] + o[d];
        }

        for (int k = 0; k < top_k; k++) {
            int e = topk_idx[k];
            float gate = s[e] / gate_sum;
            expert_ffn(w.w_gate + (size_t)e * d_ff * d_model,
                       w.w_up + (size_t)e * d_ff * d_model,
                       w.w_down + (size_t)e * d_model * d_ff, w.s_gate[e],
                       w.s_up[e], w.s_down[e], xq, s_x, o, d_model, d_ff);
            for (int d = 0; d < d_model; d++) {
                yt[d] += gate * o[d];
            }
        }
    }
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    if (num_tokens > 1){
      #pragma omp parallel for schedule(static)
      for (int t = 0; t < num_tokens; ++t) {
          const float* xt = x + (size_t)t * w.d_model;
          float* yt = y + (size_t)t * w.d_model;
          moe_forward(xt, w, yt, 1);
      } 
    }

    else 
      moe_forward(x, w, y, 1);
}
