#include "prolongrestrict.h"

#include "fmisc.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// ==========================================
// 1. Constants & Helper Functions
// ==========================================

// Prolongation Coefficients (5th order)
__constant__ double C_PROLONG[6] = {
    77.0 / 8192.0,    // C1
    -693.0 / 8192.0,  // C2
    3465.0 / 4096.0,  // C3
    1155.0 / 4096.0,  // C4
    -495.0 / 8192.0,  // C5
    63.0 / 8192.0     // C6
};

// Restriction Coefficients
__constant__ double C_RESTRICT[3] = {
    3.0 / 256.0,      // C1
    -25.0 / 256.0,    // C2
    75.0 / 128.0      // C3
};

// Fortran IDINT equivalent
__device__ int d_idint(double a) {
    if (fabs(a) < 1.0) return 0;
    return (int)(a);
}

// Memory Access Helper: Column-Major (Fortran Layout)
// i, j, k are 0-based indices
__device__ __forceinline__ int get_col_major_idx(int i, int j, int k, int nx, int ny, int nz) {
    return k * (nx * ny) + j * nx + i;
}

// ==========================================
// 2. Prolongation Device Function
// ==========================================

// Calculate prolonged value for a single point (i, j, k) on the FINE grid.
// i, j, k are 0-based indices [0, extf-1]
__device__ void d_prolong3_device(
    int i, int j, int k,
    const int* lbc, const int* lbf, const int* extc, const double* func,
    const int* extf, double* funf,
    const double* SoA, int Symmetry
) {
    // The host launch is already narrowed to this patch pair's valid box.
    // CD/FD/base and the bounds therefore do not need recomputing per point.
    int i_1b = i + 1;
    int j_1b = j + 1;
    int k_1b = k + 1;
    
    // Global index (Still effectively 1-based logic for parity check)
    // Fortran: ii = i + lbf - 1. Since our i is 0-based, ii = (i+1) + lbf - 1 = i + lbf
    int ii = i + lbf[0];
    int jj = j + lbf[1];
    int kk = k + lbf[2];

    // Coarse Index Calculation
    // Fortran: cxI = i; cxI = (cxI + lbf - 1)/2; cxI = cxI - lbc + 1
    // CUDA: use i_1b for 'i'
    int cxI_i = (i_1b + lbf[0] - 1) / 2 - lbc[0] + 1;
    int cxI_j = (j_1b + lbf[1] - 1) / 2 - lbc[1] + 1;
    int cxI_k = (k_1b + lbf[2] - 1) / 2 - lbc[2] + 1;

    // Parity Checks (Even/Odd)
    bool k_even = ((kk / 2) * 2 == kk);
    bool j_even = ((jj / 2) * 2 == jj);
    bool i_even = ((ii / 2) * 2 == ii);

    // Most prolongation targets are away from the coarse-grid boundary.  In
    // that case every 6-point sample is an ordinary contiguous read; going
    // through d_symmetry_bd_1b() 216 times only repeats bounds/reflection
    // checks.  Keep the coefficient order identical to the generic path.
    if (cxI_i >= 3 && cxI_i + 3 <= extc[0] &&
        cxI_j >= 3 && cxI_j + 3 <= extc[1] &&
        cxI_k >= 3 && cxI_k + 3 <= extc[2]) {
        double tmp1[6];
        const int sx = 1;
        const int sy = extc[0];
        const int sz = extc[0] * extc[1];
        const int origin = (cxI_i - 3) + sy * (cxI_j - 3) + sz * (cxI_k - 3);
        const double* wz = k_even ? C_PROLONG : C_PROLONG + 5;
        // The original separable interpolation materialised all 36 z-line
        // results in tmp2[m][n].  A given n is consumed immediately by the
        // following y reduction, so keeping only the six final y values
        // preserves every z/y/x accumulation order while removing a large
        // per-thread local array (and its local-memory traffic).
#pragma unroll
        for (int n = 0; n < 6; n++) {
            double yval = 0.0;
#pragma unroll
            for (int m = 0; m < 6; m++) {
                const int p = origin + n * sx + m * sy;
                double val = 0.0;
                if (k_even) {
                    val += wz[0] * func[p];
                    val += wz[1] * func[p + sz];
                    val += wz[2] * func[p + 2 * sz];
                    val += wz[3] * func[p + 3 * sz];
                    val += wz[4] * func[p + 4 * sz];
                    val += wz[5] * func[p + 5 * sz];
                } else {
                    val += wz[0] * func[p];
                    val += wz[-1] * func[p + sz];
                    val += wz[-2] * func[p + 2 * sz];
                    val += wz[-3] * func[p + 3 * sz];
                    val += wz[-4] * func[p + 4 * sz];
                    val += wz[-5] * func[p + 5 * sz];
                }
                if (j_even) {
                    yval += C_PROLONG[m] * val;
                } else {
                    yval += C_PROLONG[5 - m] * val;
                }
            }
            tmp1[n] = yval;
        }
        double final_val = 0.0;
        if (i_even) {
            final_val += C_PROLONG[0] * tmp1[0] + C_PROLONG[1] * tmp1[1] + C_PROLONG[2] * tmp1[2] +
                         C_PROLONG[3] * tmp1[3] + C_PROLONG[4] * tmp1[4] + C_PROLONG[5] * tmp1[5];
        } else {
            final_val += C_PROLONG[5] * tmp1[0] + C_PROLONG[4] * tmp1[1] + C_PROLONG[3] * tmp1[2] +
                         C_PROLONG[2] * tmp1[3] + C_PROLONG[1] * tmp1[4] + C_PROLONG[0] * tmp1[5];
        }
        funf[get_col_major_idx(i, j, k, extf[0], extf[1], extf[2])] = final_val;
        return;
    }

    // --- 3. Interpolation ---
    double tmp1[6];

    // A z-line is consumed immediately by the y reduction.  Streaming this
    // reduction keeps the original m/n accumulation order but avoids the
    // 6x6 local array (which otherwise becomes per-thread local memory).
    for (int n = 0; n < 6; n++) {
        double yval = 0.0;
        for (int m = 0; m < 6; m++) {
            int cur_ic = cxI_i - 2 + n;
            int cur_jc = cxI_j - 2 + m;
            
            double val = 0.0;
            // 1-based indices passed to d_get_sym_val
            if (k_even) {
                val += C_PROLONG[0] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 2, SoA);
                val += C_PROLONG[1] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 1, SoA);
                val += C_PROLONG[2] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k    , SoA);
                val += C_PROLONG[3] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 1, SoA);
                val += C_PROLONG[4] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 2, SoA);
                val += C_PROLONG[5] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 3, SoA);
            } else {
                val += C_PROLONG[5] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 2, SoA);
                val += C_PROLONG[4] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 1, SoA);
                val += C_PROLONG[3] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k    , SoA);
                val += C_PROLONG[2] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 1, SoA);
                val += C_PROLONG[1] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 2, SoA);
                val += C_PROLONG[0] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 3, SoA);
            }
            if (j_even) {
                yval += C_PROLONG[m] * val;
            } else {
                yval += C_PROLONG[5 - m] * val;
            }
        }
        tmp1[n] = yval;
    }

    // X-Direction Interpolation
    double final_val = 0.0;
    if (i_even) {
        final_val += C_PROLONG[0] * tmp1[0] + C_PROLONG[1] * tmp1[1] + C_PROLONG[2] * tmp1[2] +
                     C_PROLONG[3] * tmp1[3] + C_PROLONG[4] * tmp1[4] + C_PROLONG[5] * tmp1[5];
    } else {
        final_val += C_PROLONG[5] * tmp1[0] + C_PROLONG[4] * tmp1[1] + C_PROLONG[3] * tmp1[2] +
                     C_PROLONG[2] * tmp1[3] + C_PROLONG[1] * tmp1[4] + C_PROLONG[0] * tmp1[5];
    }

    // Write Output (0-based index)
    int out_idx = get_col_major_idx(i, j, k, extf[0], extf[1], extf[2]);
    funf[out_idx] = final_val;
}

// ==========================================
// 3. Restriction Device Function
// ==========================================

// Calculate restricted value for a single point (i, j, k) on the COARSE grid.
// i, j, k are 0-based indices [0, extc-1] (conceptually within the valid restricted range)
__device__ void d_restrict3_device(
    int i, int j, int k,
    const int* lbc, const int* lbf, const int* extc, double* func,
    const int* extf, const double* funf,
    const double* SoA, int Symmetry
) {
    // The host has already supplied the constant mapping and valid box.
    int i_1b = i + 1;
    int j_1b = j + 1;
    int k_1b = k + 1;
    
    // Coarse to Fine mapping
    // Fortran: cxI = i; cxI = 2*(cxI+lbc-1) - 1; cxI = cxI - lbf + 1
    // CUDA: use i_1b for 'i'
    int if_fine = 2 * (i_1b + lbc[0] - 1) - 1 - lbf[0] + 1;
    int jf_fine = 2 * (j_1b + lbc[1] - 1) - 1 - lbf[1] + 1;
    int kf_fine = 2 * (k_1b + lbc[2] - 1) - 1 - lbf[2] + 1;

    // The same direct-access rule applies to restriction.  Preserve the
    // original 6-point filter and accumulation order when all samples are
    // inside the fine grid; only boundary points retain symmetry handling.
    if (if_fine >= 3 && if_fine + 3 <= extf[0] &&
        jf_fine >= 3 && jf_fine + 3 <= extf[1] &&
        kf_fine >= 3 && kf_fine + 3 <= extf[2]) {
        double tmp1[6];
        const int sy = extf[0];
        const int sz = extf[0] * extf[1];
        const int origin = (if_fine - 3) + sy * (jf_fine - 3) + sz * (kf_fine - 3);
        for (int n = 0; n < 6; n++) {
            double yval = 0.0;
            for (int m = 0; m < 6; m++) {
                const int p = origin + n + m * sy;
                double val = 0.0;
                val += C_RESTRICT[0] * (funf[p] + funf[p + 5 * sz]);
                val += C_RESTRICT[1] * (funf[p + sz] + funf[p + 4 * sz]);
                val += C_RESTRICT[2] * (funf[p + 2 * sz] + funf[p + 3 * sz]);
                if (m == 0 || m == 5) yval += C_RESTRICT[0] * val;
                else if (m == 1 || m == 4) yval += C_RESTRICT[1] * val;
                else yval += C_RESTRICT[2] * val;
            }
            tmp1[n] = yval;
        }
        double final_val = 0.0;
        final_val += C_RESTRICT[0] * (tmp1[0] + tmp1[5]);
        final_val += C_RESTRICT[1] * (tmp1[1] + tmp1[4]);
        final_val += C_RESTRICT[2] * (tmp1[2] + tmp1[3]);
        func[get_col_major_idx(i, j, k, extc[0], extc[1], extc[2])] = final_val;
        return;
    }

    // --- 3. Restriction ---
    double tmp1[6];

    // As in prolongation, immediately reduce each z-filtered value along y
    // so the generic symmetry fallback does not force a 6x6 local array on
    // the common direct-access path.
    for (int n = 0; n < 6; n++) {
        double yval = 0.0;
        for (int m = 0; m < 6; m++) {
            int cur_jf = jf_fine - 2 + m;
            int cur_if = if_fine - 2 + n;
            
            double val = 0.0;
            // Ord=2 passed to symmetry_bd as per Fortran restrict3
            // Indices: -2, -1, 0, 1, 2, 3 relative to fine center
            val += C_RESTRICT[0] * (
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine - 2, SoA) + 
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 3, SoA)
            );
            val += C_RESTRICT[1] * (
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine - 1, SoA) + 
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 2, SoA)
            );
            val += C_RESTRICT[2] * (
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine    , SoA) + 
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 1, SoA)
            );
            
            if (m == 0 || m == 5) yval += C_RESTRICT[0] * val;
            else if (m == 1 || m == 4) yval += C_RESTRICT[1] * val;
            else yval += C_RESTRICT[2] * val;
        }
        tmp1[n] = yval;
    }

    // X-Direction Restriction
    double final_val = 0.0;
    final_val += C_RESTRICT[0] * (tmp1[0] + tmp1[5]);
    final_val += C_RESTRICT[1] * (tmp1[1] + tmp1[4]);
    final_val += C_RESTRICT[2] * (tmp1[2] + tmp1[3]);

    // Write Output (0-based index)
    int out_idx = get_col_major_idx(i, j, k, extc[0], extc[1], extc[2]);
    func[out_idx] = final_val;
}

// ++++++++++++++ Kernel Implementation ++++++++++++++
// ---------------------------------------------------------
// 1. 直接面向显存的单任务 Prolong Kernel
// ---------------------------------------------------------
__global__ void prolong3_kernel(
    int ni, int nj, int nk,
    int i_start, int j_start, int k_start,
    int lbc0, int lbc1, int lbc2, int lbf0, int lbf1, int lbf2,
    double llbc0, double llbc1, double llbc2,
    double uubc0, double uubc1, double uubc2,
    int extc0, int extc1, int extc2,
    const double* __restrict__ d_src_c,
    double llbf0, double llbf1, double llbf2,
    double uubf0, double uubf1, double uubf2,
    int extf0, int extf1, int extf2,
    double* __restrict__ d_dst_f,
    double llbt0, double llbt1, double llbt2,
    double uubt0, double uubt1, double uubt2,
    double SoA0, double SoA1, double SoA2,
    int Symmetry
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ni * nj * nk;
    if (idx >= total) return;

    // 1D to 3D mapping
    int k_local = idx / (ni * nj);
    int rem     = idx % (ni * nj);
    int j_local = rem / ni;
    int i_local = rem % ni;

    // 0-based Fortran equivalent loop indices
    int i = i_start + i_local;
    int j = j_start + j_local;
    int k = k_start + k_local;

    // The launch-side geometry is constant for all points in this patch.
    int    arr_lbc[3]  = {lbc0, lbc1, lbc2};
    int    arr_lbf[3]  = {lbf0, lbf1, lbf2};
    int    arr_extc[3] = {extc0, extc1, extc2};
    int    arr_extf[3] = {extf0, extf1, extf2};
    double arr_SoA[3]  = {SoA0, SoA1, SoA2};

    d_prolong3_device(
        i, j, k,
        arr_lbc, arr_lbf, arr_extc, d_src_c,
        arr_extf, d_dst_f,
        arr_SoA, Symmetry
    );
}

// ---------------------------------------------------------
// 3. 直接面向显存的单任务 Restrict Kernel
// ---------------------------------------------------------
__global__ void restrict3_kernel(
    int ni, int nj, int nk,
    int i_start, int j_start, int k_start,
    int lbc0, int lbc1, int lbc2, int lbf0, int lbf1, int lbf2,
    double llbc0, double llbc1, double llbc2,
    double uubc0, double uubc1, double uubc2,
    int extc0, int extc1, int extc2,
    double* __restrict__ d_dst_c, // Restrict 目标是粗网格
    double llbf0, double llbf1, double llbf2,
    double uubf0, double uubf1, double uubf2,
    int extf0, int extf1, int extf2,
    const double* __restrict__ d_src_f,
    double llbt0, double llbt1, double llbt2,
    double uubt0, double uubt1, double uubt2,
    double SoA0, double SoA1, double SoA2,
    int Symmetry
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ni * nj * nk;
    if (idx >= total) return;

    int k_local = idx / (ni * nj);
    int rem     = idx % (ni * nj);
    int j_local = rem / ni;
    int i_local = rem % ni;

    int i = i_start + i_local;
    int j = j_start + j_local;
    int k = k_start + k_local;

    int    arr_lbc[3]  = {lbc0, lbc1, lbc2};
    int    arr_lbf[3]  = {lbf0, lbf1, lbf2};
    int    arr_extc[3] = {extc0, extc1, extc2};
    int    arr_extf[3] = {extf0, extf1, extf2};
    double arr_SoA[3]  = {SoA0, SoA1, SoA2};

    d_restrict3_device(
        i, j, k,
        arr_lbc, arr_lbf, arr_extc, d_dst_c,
        arr_extf, d_src_f,
        arr_SoA, Symmetry
    );
}

// A transfer segment has identical geometry for all state variables.  Launch
// the fields as a second grid dimension and fetch each field's pointers/SoA
// once per block.  The per-point interpolation stays exactly in
// d_prolong3_device/d_restrict3_device, including the generic symmetry path.
__global__ void prolong3_batch_kernel(
    int ni, int nj, int nk,
    int i_start, int j_start, int k_start,
    int lbc0, int lbc1, int lbc2, int lbf0, int lbf1, int lbf2,
    int extc0, int extc1, int extc2,
    double* const* d_src_c, double* const* d_dst_f, const double* d_soa,
    int extf0, int extf1, int extf2,
    int Symmetry, int nfields
) {
    const int field = blockIdx.y;
    if (field >= nfields) return;
    __shared__ const double* src;
    __shared__ double* dst;
    __shared__ double soa[3];
    if (threadIdx.x == 0) {
        src = d_src_c[field];
        dst = d_dst_f[field];
        soa[0] = d_soa[3 * field];
        soa[1] = d_soa[3 * field + 1];
        soa[2] = d_soa[3 * field + 2];
    }
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ni * nj * nk;
    if (idx >= total) return;
    const int k = k_start + idx / (ni * nj);
    const int rem = idx % (ni * nj);
    const int j = j_start + rem / ni;
    const int i = i_start + rem % ni;
    const int lbc[3] = {lbc0, lbc1, lbc2};
    const int lbf[3] = {lbf0, lbf1, lbf2};
    const int extc[3] = {extc0, extc1, extc2};
    const int extf[3] = {extf0, extf1, extf2};
    d_prolong3_device(i, j, k, lbc, lbf, extc, src, extf, dst, soa, Symmetry);
}

__global__ void restrict3_batch_kernel(
    int ni, int nj, int nk,
    int i_start, int j_start, int k_start,
    int lbc0, int lbc1, int lbc2, int lbf0, int lbf1, int lbf2,
    int extc0, int extc1, int extc2,
    double* const* d_src_f, double* const* d_dst_c, const double* d_soa,
    int extf0, int extf1, int extf2,
    int Symmetry, int nfields
) {
    const int field = blockIdx.y;
    if (field >= nfields) return;
    __shared__ const double* src;
    __shared__ double* dst;
    __shared__ double soa[3];
    if (threadIdx.x == 0) {
        src = d_src_f[field];
        dst = d_dst_c[field];
        soa[0] = d_soa[3 * field];
        soa[1] = d_soa[3 * field + 1];
        soa[2] = d_soa[3 * field + 2];
    }
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ni * nj * nk;
    if (idx >= total) return;
    const int k = k_start + idx / (ni * nj);
    const int rem = idx % (ni * nj);
    const int j = j_start + rem / ni;
    const int i = i_start + rem % ni;
    const int lbc[3] = {lbc0, lbc1, lbc2};
    const int lbf[3] = {lbf0, lbf1, lbf2};
    const int extc[3] = {extc0, extc1, extc2};
    const int extf[3] = {extf0, extf1, extf2};
    d_restrict3_device(i, j, k, lbc, lbf, extc, dst, extf, src, soa, Symmetry);
}

// ---------------------------------------------------------
// 4. Host 端启动接口
// ---------------------------------------------------------
void gpu_prolong3_launch(
    cudaStream_t stream,
    const double* d_src_c, double* d_dst_f,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry
) {
    double CD[3], FD[3], base[3];
    for(int d = 0; d < 3; d++) {
        CD[d] = (uubc[d] - llbc[d]) / (double)extc[d];
        FD[d] = (uubf[d] - llbf[d]) / (double)extf[d];
        if (llbc[d] <= llbf[d]) {
            base[d] = llbc[d];
        } else {
            // 修正：使用 std::trunc 完美对齐 Fortran 的 idint (向零取整)
            int j_val = (int)std::trunc((llbc[d] - llbf[d]) / FD[d] + 0.4);
            if ((j_val / 2) * 2 == j_val) base[d] = llbf[d];
            else base[d] = llbf[d] - CD[d] / 2.0;
        }
    }

    int lbc_idx[3], lbf_idx[3];
    int i_start, i_end, j_start, j_end, k_start, k_end;
    for(int d = 0; d < 3; d++) {
        // 修正：使用 std::trunc
        int lbp = (int)std::trunc((llbt[d] - base[d]) / FD[d] + 0.4) + 1;
        int ubp = (int)std::trunc((uubt[d] - base[d]) / FD[d] + 0.4);
        lbc_idx[d] = (int)std::trunc((llbc[d] - base[d]) / CD[d] + 0.4) + 1;
        lbf_idx[d] = (int)std::trunc((llbf[d] - base[d]) / FD[d] + 0.4) + 1;
        
        if (d == 0) { i_start = lbp - lbf_idx[d]; i_end = ubp - lbf_idx[d]; }
        if (d == 1) { j_start = lbp - lbf_idx[d]; j_end = ubp - lbf_idx[d]; }
        if (d == 2) { k_start = lbp - lbf_idx[d]; k_end = ubp - lbf_idx[d]; }
    }

    int ni = i_end - i_start + 1;
    int nj = j_end - j_start + 1;
    int nk = k_end - k_start + 1;

    if (ni <= 0 || nj <= 0 || nk <= 0) return; // 剔除空操作

    int total_points = ni * nj * nk;
    int block = 256;
    int grid = (total_points + block - 1) / block;

    prolong3_kernel<<<grid, block, 0, stream>>>(
        ni, nj, nk, i_start, j_start, k_start,
        lbc_idx[0], lbc_idx[1], lbc_idx[2], lbf_idx[0], lbf_idx[1], lbf_idx[2],
        llbc[0], llbc[1], llbc[2],
        uubc[0], uubc[1], uubc[2],
        extc[0], extc[1], extc[2],
        d_src_c,
        llbf[0], llbf[1], llbf[2],
        uubf[0], uubf[1], uubf[2],
        extf[0], extf[1], extf[2],
        d_dst_f,
        llbt[0], llbt[1], llbt[2],
        uubt[0], uubt[1], uubt[2],
        SoA[0], SoA[1], SoA[2],
        Symmetry
    );
}

void gpu_prolong3_batch_launch(
    cudaStream_t stream,
    double* const* d_src_c, double* const* d_dst_f, const double* d_soa,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    int Symmetry, int nfields
) {
    double CD[3], FD[3], base[3];
    for (int d = 0; d < 3; ++d) {
        CD[d] = (uubc[d] - llbc[d]) / (double)extc[d];
        FD[d] = (uubf[d] - llbf[d]) / (double)extf[d];
        if (llbc[d] <= llbf[d]) base[d] = llbc[d];
        else {
            const int j = (int)std::trunc((llbc[d] - llbf[d]) / FD[d] + 0.4);
            base[d] = ((j / 2) * 2 == j) ? llbf[d] : llbf[d] - CD[d] / 2.0;
        }
    }
    int lbc[3], lbf[3], start[3], end[3];
    for (int d = 0; d < 3; ++d) {
        const int lbp = (int)std::trunc((llbt[d] - base[d]) / FD[d] + 0.4) + 1;
        const int ubp = (int)std::trunc((uubt[d] - base[d]) / FD[d] + 0.4);
        lbc[d] = (int)std::trunc((llbc[d] - base[d]) / CD[d] + 0.4) + 1;
        lbf[d] = (int)std::trunc((llbf[d] - base[d]) / FD[d] + 0.4) + 1;
        start[d] = lbp - lbf[d];
        end[d] = ubp - lbf[d];
    }
    const int ni = end[0] - start[0] + 1;
    const int nj = end[1] - start[1] + 1;
    const int nk = end[2] - start[2] + 1;
    if (ni <= 0 || nj <= 0 || nk <= 0 || nfields <= 0) return;
    constexpr int block = 128;
    const dim3 grid((ni * nj * nk + block - 1) / block, nfields);
    prolong3_batch_kernel<<<grid, block, 0, stream>>>(
        ni, nj, nk, start[0], start[1], start[2],
        lbc[0], lbc[1], lbc[2], lbf[0], lbf[1], lbf[2],
        extc[0], extc[1], extc[2], d_src_c, d_dst_f, d_soa,
        extf[0], extf[1], extf[2], Symmetry, nfields
    );
}

void gpu_restrict3_launch(
    cudaStream_t stream,
    const double* d_src_f, double* d_dst_c,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry
) {
    double CD[3], FD[3], base[3];
    for(int d = 0; d < 3; d++) {
        CD[d] = (uubc[d] - llbc[d]) / (double)extc[d];
        FD[d] = (uubf[d] - llbf[d]) / (double)extf[d];
        if (llbc[d] <= llbf[d]) {
            base[d] = llbc[d];
        } else {
            int j_val = (int)std::trunc((llbc[d] - llbf[d]) / FD[d] + 0.4);
            if ((j_val / 2) * 2 == j_val) base[d] = llbf[d];
            else base[d] = llbf[d] - CD[d] / 2.0;
        }
    }

    int lbc_idx[3], lbf_idx[3];
    int i_start, i_end, j_start, j_end, k_start, k_end;
    for(int d = 0; d < 3; d++) {
        int lbr = (int)std::trunc((llbt[d] - base[d]) / CD[d] + 0.4) + 1;
        int ubr = (int)std::trunc((uubt[d] - base[d]) / CD[d] + 0.4);
        lbc_idx[d] = (int)std::trunc((llbc[d] - base[d]) / CD[d] + 0.4) + 1;
        lbf_idx[d] = (int)std::trunc((llbf[d] - base[d]) / FD[d] + 0.4) + 1;
        
        if (d == 0) { i_start = lbr - lbc_idx[d]; i_end = ubr - lbc_idx[d]; }
        if (d == 1) { j_start = lbr - lbc_idx[d]; j_end = ubr - lbc_idx[d]; }
        if (d == 2) { k_start = lbr - lbc_idx[d]; k_end = ubr - lbc_idx[d]; }
    }

    int ni = i_end - i_start + 1;
    int nj = j_end - j_start + 1;
    int nk = k_end - k_start + 1;

    if (ni <= 0 || nj <= 0 || nk <= 0) return;

    int total_points = ni * nj * nk;
    int block = 256;
    int grid = (total_points + block - 1) / block;

    restrict3_kernel<<<grid, block, 0, stream>>>(
        ni, nj, nk, i_start, j_start, k_start,
        lbc_idx[0], lbc_idx[1], lbc_idx[2], lbf_idx[0], lbf_idx[1], lbf_idx[2],
        llbc[0], llbc[1], llbc[2],
        uubc[0], uubc[1], uubc[2],
        extc[0], extc[1], extc[2],
        d_dst_c,
        llbf[0], llbf[1], llbf[2],
        uubf[0], uubf[1], uubf[2],
        extf[0], extf[1], extf[2],
        d_src_f,
        llbt[0], llbt[1], llbt[2],
        uubt[0], uubt[1], uubt[2],
        SoA[0], SoA[1], SoA[2],
        Symmetry
    );
}

void gpu_restrict3_batch_launch(
    cudaStream_t stream,
    double* const* d_src_f, double* const* d_dst_c, const double* d_soa,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    int Symmetry, int nfields
) {
    double CD[3], FD[3], base[3];
    for (int d = 0; d < 3; ++d) {
        CD[d] = (uubc[d] - llbc[d]) / (double)extc[d];
        FD[d] = (uubf[d] - llbf[d]) / (double)extf[d];
        if (llbc[d] <= llbf[d]) base[d] = llbc[d];
        else {
            const int j = (int)std::trunc((llbc[d] - llbf[d]) / FD[d] + 0.4);
            base[d] = ((j / 2) * 2 == j) ? llbf[d] : llbf[d] - CD[d] / 2.0;
        }
    }
    int lbc[3], lbf[3], start[3], end[3];
    for (int d = 0; d < 3; ++d) {
        const int lbr = (int)std::trunc((llbt[d] - base[d]) / CD[d] + 0.4) + 1;
        const int ubr = (int)std::trunc((uubt[d] - base[d]) / CD[d] + 0.4);
        lbc[d] = (int)std::trunc((llbc[d] - base[d]) / CD[d] + 0.4) + 1;
        lbf[d] = (int)std::trunc((llbf[d] - base[d]) / FD[d] + 0.4) + 1;
        start[d] = lbr - lbc[d];
        end[d] = ubr - lbc[d];
    }
    const int ni = end[0] - start[0] + 1;
    const int nj = end[1] - start[1] + 1;
    const int nk = end[2] - start[2] + 1;
    if (ni <= 0 || nj <= 0 || nk <= 0 || nfields <= 0) return;
    constexpr int block = 128;
    const dim3 grid((ni * nj * nk + block - 1) / block, nfields);
    restrict3_batch_kernel<<<grid, block, 0, stream>>>(
        ni, nj, nk, start[0], start[1], start[2],
        lbc[0], lbc[1], lbc[2], lbf[0], lbf[1], lbf[2],
        extc[0], extc[1], extc[2], d_src_f, d_dst_c, d_soa,
        extf[0], extf[1], extf[2], Symmetry, nfields
    );
}
