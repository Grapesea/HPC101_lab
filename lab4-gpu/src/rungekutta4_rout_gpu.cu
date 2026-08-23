#include "rungekutta4_rout.h"

#include <cuda_runtime.h>
#include <math.h>

#include "gpu_manager.h"

__global__ void rungekutta4_rout_kernel(
    int n,
    const double* f0, double* f1, double* f_rhs, double dT, int RK4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double f0_val = f0[idx];
    double f1_val = f1[idx];
    double f_rhs_val = f_rhs[idx];

    if (RK4 == 0) {
        f1[idx] = f0_val + 0.5 * dT * f_rhs_val;
    } else if (RK4 == 1) {
        f_rhs[idx] = f_rhs_val + 2.0 * f1_val;
        f1[idx]    = f0_val + 0.5 * dT * f1_val;
    } else if (RK4 == 2) {
        f_rhs[idx] = f_rhs_val + 2.0 * f1_val;
        f1[idx]    = f0_val + dT * f1_val;
    } else if (RK4 == 3) {
        f1[idx] = f0_val + (1.0 / 6.0) * dT * (f1_val + f_rhs_val);
    }
}

__device__ __forceinline__ bool rk_sommerfeld_boundary(
    int i, int j, int k,
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax, int Symmetry
) {
    const double dX = X[1] - X[0];
    const double dY = Y[1] - Y[0];
    const double dZ = Z[1] - Z[0];
    if (i == ex0 && fabs(X[ex0 - 1] - xmax) < dX) return true;
    if (j == ex1 && fabs(Y[ex1 - 1] - ymax) < dY) return true;
    if (k == ex2 && fabs(Z[ex2 - 1] - zmax) < dZ) return true;
    if (i == 1 && fabs(X[0] - xmin) < dX && !(Symmetry == 2 && fabs(xmin) < dX / 2.0)) return true;
    if (j == 1 && fabs(Y[0] - ymin) < dY && !(Symmetry == 2 && fabs(ymin) < dY / 2.0)) return true;
    if (k == 1 && fabs(Z[0] - zmin) < dZ && !(Symmetry > 0 && fabs(zmin) < dZ / 2.0)) return true;
    return false;
}

__global__ void rungekutta4_rout_batch_kernel(
    int n, int ex0, int ex1, int ex2,
    const double* const* f0s, double* const* f1s, double* const* rhss,
    double dT, int RK4, bool fix_boundary,
    const double* X, const double* Y, const double* Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax, int Symmetry
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int field = blockIdx.y;
    if (idx >= n || field >= 24) return;

    const double* f0 = f0s[field];
    double* f1 = f1s[field];
    double* rhs = rhss[field];
    const double f0_val = f0[idx];
    const double f1_val = f1[idx];
    const double rhs_val = rhs[idx];
    if (RK4 == 0) {
        f1[idx] = f0_val + 0.5 * dT * rhs_val;
    } else if (RK4 == 1) {
        rhs[idx] = rhs_val + 2.0 * f1_val;
        f1[idx] = f0_val + 0.5 * dT * f1_val;
    } else if (RK4 == 2) {
        rhs[idx] = rhs_val + 2.0 * f1_val;
        f1[idx] = f0_val + dT * f1_val;
    } else if (RK4 == 3) {
        f1[idx] = f0_val + (1.0 / 6.0) * dT * (f1_val + rhs_val);
    }

    if (fix_boundary) {
        const int i = idx % ex0 + 1;
        const int j = (idx / ex0) % ex1 + 1;
        const int k = idx / (ex0 * ex1) + 1;
        if (rk_sommerfeld_boundary(i, j, k, ex0, ex1, ex2,
                                   X, Y, Z, xmin, ymin, zmin,
                                   xmax, ymax, zmax, Symmetry)) {
            f1[idx] = f0_val;
        }
    }
}

void gpu_rungekutta4_rout_launch(
    cudaStream_t &stream,
    int ex[3], double dT,
    double* d_f0, double* d_f1, double* d_f_rhs, int RK4
) {
    int n = ex[0] * ex[1] * ex[2];

    int block = 256;
    int grid = (n + block - 1) / block;

    rungekutta4_rout_kernel<<<grid, block, 0, stream>>>(
        n,
        d_f0, d_f1, d_f_rhs, dT, RK4
    );
}

void gpu_rungekutta4_rout_batch_launch(
    cudaStream_t &stream,
    int ex[3], double dT,
    double** d_f0, double** d_f1, double** d_f_rhs, int RK4,
    bool fix_boundary,
    const double* d_X, const double* d_Y, const double* d_Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax, int Symmetry
) {
    const int n = ex[0] * ex[1] * ex[2];
    const int block = 256;
    const dim3 grid((n + block - 1) / block, 24);
    rungekutta4_rout_batch_kernel<<<grid, block, 0, stream>>>(
        n, ex[0], ex[1], ex[2], d_f0, d_f1, d_f_rhs, dT, RK4,
        fix_boundary, d_X, d_Y, d_Z, xmin, ymin, zmin,
        xmax, ymax, zmax, Symmetry
    );
}
