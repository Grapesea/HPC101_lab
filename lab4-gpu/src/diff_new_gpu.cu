#include "derivatives.h"

#include "fmisc.h"

#include "macrodef.fh"
#include <cmath>
#include <iostream>

// ==========================================
// Device Function: 一阶导数 (4th Order)
// ==========================================
#ifndef AMSS_DERIV_ATTR
#define AMSS_DERIV_ATTR __device__
#endif

AMSS_DERIV_ATTR void d_fderivs_point(
    const int ex[3], const double* f,
    double* fx, double* fy, double* fz,
    const double* X, const double* Y, const double* Z,
    double SYM1, double SYM2, double SYM3,
    int symmetry, int onoff,
    int i, int j, int k
) {
    const double ONE = 1.0;
    const double TWO = 2.0;
    const double EIT = 8.0;
    const double F12 = 12.0;
    const double ZEO = 0.0;
    const int NO_SYMM = 0, EQ_SYMM = 1;

    const double dX = X[1] - X[0];
    const double dY = Y[1] - Y[0];
    const double dZ = Z[1] - Z[0];

    const int imax = ex[0] - 1;
    const int jmax = ex[1] - 1;
    const int kmax = ex[2] - 1;

    *fx = ZEO;
    *fy = ZEO;
    *fz = ZEO;

    // Fortran 循环范围是 1 到 ex-1，对应 CUDA 0 到 ex-2。
    // 如果 i >= imax (即 i >= ex-1)，直接返回，保持输出为 0。
    if (i >= imax || j >= jmax || k >= kmax) return;

    // --- 修复开始 ---
    // Fortran 中 imin = -1 (1-based index)。
    // 在 CUDA (0-based) 中，为了使边界点 i=0 满足 (i-2 >= imin)，
    // 即 (0-2 >= imin) -> (-2 >= imin)，imin 必须设为 -2。
    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > NO_SYMM && fabs(Z[0]) < dZ) kmin = -2; // 原代码为 -1，修正为 -2
    if (symmetry > EQ_SYMM && fabs(X[0]) < dX) imin = -2; // 原代码为 -1，修正为 -2
    if (symmetry > EQ_SYMM && fabs(Y[0]) < dY) jmin = -2; // 原代码为 -1，修正为 -2

    double SoA[3] = {SYM1, SYM2, SYM3};

    const double d12dx = ONE / F12 / dX;
    const double d12dy = ONE / F12 / dY;
    const double d12dz = ONE / F12 / dZ;

    const double d2dx = ONE / TWO / dX;
    const double d2dy = ONE / TWO / dY;
    const double d2dz = ONE / TWO / dZ;

    // Most grid points are at least two cells away from a symmetry plane or
    // an outer boundary.  For these points d_symmetry_bd_1b() only performs
    // range checks before returning the same contiguous element.  Take that
    // common path directly; the original symmetry-aware path below remains
    // responsible for every boundary point.
    if (i >= 2 && i + 2 <= imax &&
        j >= 2 && j + 2 <= jmax &&
        k >= 2 && k + 2 <= kmax) {
        const int sy = ex[0];
        const int sz = ex[0] * ex[1];
        const int p = i + sy * j + sz * k;
        *fx = d12dx * (f[p - 2] - EIT * f[p - 1] + EIT * f[p + 1] - f[p + 2]);
        *fy = d12dy * (f[p - 2 * sy] - EIT * f[p - sy] + EIT * f[p + sy] - f[p + 2 * sy]);
        *fz = d12dz * (f[p - 2 * sz] - EIT * f[p - sz] + EIT * f[p + sz] - f[p + 2 * sz]);
        return;
    }

    // Helper lambda for symmetry boundary access
    const auto fh = [&](int ii, int jj, int kk) -> double {
        if (ii >= 0 && ii < ex[0] && jj >= 0 && jj < ex[1] && kk >= 0 && kk < ex[2]) {
            return f[ii + ex[0] * (jj + ex[1] * kk)];
        }
        return d_symmetry_bd_1b(2, ex, f, ii + 1, jj + 1, kk + 1, SoA);
    };
    
    if (i + 2 <= imax && i - 2 >= imin &&
        j + 2 <= jmax && j - 2 >= jmin &&
        k + 2 <= kmax && k - 2 >= kmin) {

        *fx = d12dx * (fh(i-2,j,k) - EIT*fh(i-1,j,k) + EIT*fh(i+1,j,k) - fh(i+2,j,k));
        *fy = d12dy * (fh(i,j-2,k) - EIT*fh(i,j-1,k) + EIT*fh(i,j+1,k) - fh(i,j+2,k));
        *fz = d12dz * (fh(i,j,k-2) - EIT*fh(i,j,k-1) + EIT*fh(i,j,k+1) - fh(i,j,k+2));

    }
    else if (i + 1 <= imax && i - 1 >= imin &&
               j + 1 <= jmax && j - 1 >= jmin &&
               k + 1 <= kmax && k - 1 >= kmin) {

        *fx = d2dx * (-fh(i-1,j,k) + fh(i+1,j,k));
        *fy = d2dy * (-fh(i,j-1,k) + fh(i,j+1,k));
        *fz = d2dz * (-fh(i,j,k-1) + fh(i,j,k+1));
    }

    (void)onoff;
}

// ==========================================
// Device Function: 二阶导数 (4th Order)
// ==========================================
AMSS_DERIV_ATTR void d_fdderivs_point(
    const int ex[3], const double* f,
    double* fxx, double* fxy, double* fxz,
    double* fyy, double* fyz, double* fzz,
    const double* X, const double* Y, const double* Z,
    double SYM1, double SYM2, double SYM3,
    int symmetry, int onoff,
    int i, int j, int k
) {
    const double ONE = 1.0;
    const double TWO = 2.0;
    const double F1o4 = 0.25;
    const double F1o12 = ONE / 12.0;
    const double F1o144 = ONE / 144.0;
    const double F8 = 8.0;
    const double F16 = 16.0;
    const double F30 = 30.0;
    const double ZEO = 0.0;
    const int NO_SYMM = 0, EQ_SYMM = 1;

    const double dX = X[1] - X[0];
    const double dY = Y[1] - Y[0];
    const double dZ = Z[1] - Z[0];

    const int imax = ex[0] - 1;
    const int jmax = ex[1] - 1;
    const int kmax = ex[2] - 1;

    *fxx = ZEO; *fyy = ZEO; *fzz = ZEO;
    *fxy = ZEO; *fxz = ZEO; *fyz = ZEO;

    if (i >= imax || j >= jmax || k >= kmax) return;

    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > NO_SYMM && fabs(Z[0]) < dZ) kmin = -2;
    if (symmetry > EQ_SYMM && fabs(X[0]) < dX) imin = -2;
    if (symmetry > EQ_SYMM && fabs(Y[0]) < dY) jmin = -2;

    double SoA[3] = {SYM1, SYM2, SYM3};

    const double Sdxdx = ONE / (dX * dX);
    const double Sdydy = ONE / (dY * dY);
    const double Sdzdz = ONE / (dZ * dZ);

    const double Fdxdx = F1o12 / (dX * dX);
    const double Fdydy = F1o12 / (dY * dY);
    const double Fdzdz = F1o12 / (dZ * dZ);

    const double Sdxdy = F1o4 / (dX * dY);
    const double Sdxdz = F1o4 / (dX * dZ);
    const double Sdydz = F1o4 / (dY * dZ);

    const double Fdxdy = F1o144 / (dX * dY);
    const double Fdxdz = F1o144 / (dX * dZ);
    const double Fdydz = F1o144 / (dY * dZ);

    // Fast interior stencil.  This is algebraically identical to the
    // symmetry helper path, but avoids 78 helper calls for each second
    // derivative evaluation at a point that cannot touch a boundary.
    if (i >= 2 && i + 2 <= imax &&
        j >= 2 && j + 2 <= jmax &&
        k >= 2 && k + 2 <= kmax) {
        const int sy = ex[0];
        const int sz = ex[0] * ex[1];
        const int p = i + sy * j + sz * k;

        *fxx = Fdxdx * (-f[p - 2] + F16 * f[p - 1] - F30 * f[p]
                         - f[p + 2] + F16 * f[p + 1]);
        *fyy = Fdydy * (-f[p - 2 * sy] + F16 * f[p - sy] - F30 * f[p]
                         - f[p + 2 * sy] + F16 * f[p + sy]);
        *fzz = Fdzdz * (-f[p - 2 * sz] + F16 * f[p - sz] - F30 * f[p]
                         - f[p + 2 * sz] + F16 * f[p + sz]);

        *fxy = Fdxdy * (
            (f[p - 2 * sy - 2] - F8 * f[p - 2 * sy - 1] + F8 * f[p - 2 * sy + 1] - f[p - 2 * sy + 2])
          - F8 * (f[p - sy - 2]     - F8 * f[p - sy - 1]     + F8 * f[p - sy + 1]     - f[p - sy + 2])
          + F8 * (f[p + sy - 2]     - F8 * f[p + sy - 1]     + F8 * f[p + sy + 1]     - f[p + sy + 2])
          -      (f[p + 2 * sy - 2] - F8 * f[p + 2 * sy - 1] + F8 * f[p + 2 * sy + 1] - f[p + 2 * sy + 2]));
        *fxz = Fdxdz * (
            (f[p - 2 * sz - 2] - F8 * f[p - 2 * sz - 1] + F8 * f[p - 2 * sz + 1] - f[p - 2 * sz + 2])
          - F8 * (f[p - sz - 2]     - F8 * f[p - sz - 1]     + F8 * f[p - sz + 1]     - f[p - sz + 2])
          + F8 * (f[p + sz - 2]     - F8 * f[p + sz - 1]     + F8 * f[p + sz + 1]     - f[p + sz + 2])
          -      (f[p + 2 * sz - 2] - F8 * f[p + 2 * sz - 1] + F8 * f[p + 2 * sz + 1] - f[p + 2 * sz + 2]));
        *fyz = Fdydz * (
            (f[p - 2 * sz - 2 * sy] - F8 * f[p - 2 * sz - sy] + F8 * f[p - 2 * sz + sy] - f[p - 2 * sz + 2 * sy])
          - F8 * (f[p - sz - 2 * sy]     - F8 * f[p - sz - sy]     + F8 * f[p - sz + sy]     - f[p - sz + 2 * sy])
          + F8 * (f[p + sz - 2 * sy]     - F8 * f[p + sz - sy]     + F8 * f[p + sz + sy]     - f[p + sz + 2 * sy])
          -      (f[p + 2 * sz - 2 * sy] - F8 * f[p + 2 * sz - sy] + F8 * f[p + 2 * sz + sy] - f[p + 2 * sz + 2 * sy]));
        return;
    }

    const auto fh = [&](int ii, int jj, int kk) -> double {
        if (ii >= 0 && ii < ex[0] && jj >= 0 && jj < ex[1] && kk >= 0 && kk < ex[2]) {
            return f[ii + ex[0] * (jj + ex[1] * kk)];
        }
        return d_symmetry_bd_1b(2, ex, f, ii + 1, jj + 1, kk + 1, SoA);
    };
    
    // --- 4th Order Accuracy ---
    if (i + 2 <= imax && i - 2 >= imin &&
        j + 2 <= jmax && j - 2 >= jmin &&
        k + 2 <= kmax && k - 2 >= kmin) {

        *fxx = Fdxdx * (-fh(i-2,j,k) + F16*fh(i-1,j,k) - F30*fh(i,j,k)
                        -fh(i+2,j,k) + F16*fh(i+1,j,k));
        *fyy = Fdydy * (-fh(i,j-2,k) + F16*fh(i,j-1,k) - F30*fh(i,j,k)
                        -fh(i,j+2,k) + F16*fh(i,j+1,k));
        *fzz = Fdzdz * (-fh(i,j,k-2) + F16*fh(i,j,k-1) - F30*fh(i,j,k)
                        -fh(i,j,k+2) + F16*fh(i,j,k+1));

        *fxy = Fdxdy * (    (fh(i-2,j-2,k) - F8*fh(i-1,j-2,k) + F8*fh(i+1,j-2,k) - fh(i+2,j-2,k))
                        -F8*(fh(i-2,j-1,k) - F8*fh(i-1,j-1,k) + F8*fh(i+1,j-1,k) - fh(i+2,j-1,k))
                        +F8*(fh(i-2,j+1,k) - F8*fh(i-1,j+1,k) + F8*fh(i+1,j+1,k) - fh(i+2,j+1,k))
                        -   (fh(i-2,j+2,k) - F8*fh(i-1,j+2,k) + F8*fh(i+1,j+2,k) - fh(i+2,j+2,k)) );
        *fxz = Fdxdz * (    (fh(i-2,j,k-2) - F8*fh(i-1,j,k-2) + F8*fh(i+1,j,k-2) - fh(i+2,j,k-2))
                        -F8*(fh(i-2,j,k-1) - F8*fh(i-1,j,k-1) + F8*fh(i+1,j,k-1) - fh(i+2,j,k-1))
                        +F8*(fh(i-2,j,k+1) - F8*fh(i-1,j,k+1) + F8*fh(i+1,j,k+1) - fh(i+2,j,k+1))
                        -   (fh(i-2,j,k+2) - F8*fh(i-1,j,k+2) + F8*fh(i+1,j,k+2) - fh(i+2,j,k+2)) );
        *fyz = Fdydz * (    (fh(i,j-2,k-2) - F8*fh(i,j-1,k-2) + F8*fh(i,j+1,k-2) - fh(i,j+2,k-2))
                        -F8*(fh(i,j-2,k-1) - F8*fh(i,j-1,k-1) + F8*fh(i,j+1,k-1) - fh(i,j+2,k-1))
                        +F8*(fh(i,j-2,k+1) - F8*fh(i,j-1,k+1) + F8*fh(i,j+1,k+1) - fh(i,j+2,k+1))
                        -   (fh(i,j-2,k+2) - F8*fh(i,j-1,k+2) + F8*fh(i,j+1,k+2) - fh(i,j+2,k+2)) );

    } 
    // --- 2nd Order Accuracy ---
    else if (i + 1 <= imax && i - 1 >= imin &&
               j + 1 <= jmax && j - 1 >= jmin &&
               k + 1 <= kmax && k - 1 >= kmin) {

        *fxx = Sdxdx * (fh(i-1,j,k) - TWO*fh(i,j,k) + fh(i+1,j,k));
        *fyy = Sdydy * (fh(i,j-1,k) - TWO*fh(i,j,k) + fh(i,j+1,k));
        *fzz = Sdzdz * (fh(i,j,k-1) - TWO*fh(i,j,k) + fh(i,j,k+1));

        *fxy = Sdxdy * (fh(i-1,j-1,k) - fh(i+1,j-1,k) - fh(i-1,j+1,k) + fh(i+1,j+1,k));
        *fxz = Sdxdz * (fh(i-1,j,k-1) - fh(i+1,j,k-1) - fh(i-1,j,k+1) + fh(i+1,j,k+1));
        *fyz = Sdydz * (fh(i,j-1,k-1) - fh(i,j+1,k-1) - fh(i,j-1,k+1) + fh(i,j+1,k+1));
    }

    (void)onoff;
}
