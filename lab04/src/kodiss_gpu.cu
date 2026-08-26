#include "kodiss.h"

#include "fmisc.h"

#include "macrodef.fh"
#include <cmath>

#ifndef AMSS_KODIS_ATTR
#define AMSS_KODIS_ATTR __device__
#endif

AMSS_KODIS_ATTR double d_kodis_point(
    const int ex[3], const double* f,
    const double* X, const double* Y, const double* Z,
    double SYM1, double SYM2, double SYM3,
    int symmetry, double eps,
    int i, int j, int k // 0-based
) {
    const double ONE = 1.0;
    const double SIX = 6.0;
    const double FIT = 15.0;
    const double TWT = 20.0;
    const double cof = 64.0;
    const int NO_SYMM = 0, OCTANT = 2;

    const double dX = X[1] - X[0];
    const double dY = Y[1] - Y[0];
    const double dZ = Z[1] - Z[0];

    const int imax = ex[0] - 1;
    const int jmax = ex[1] - 1;
    const int kmax = ex[2] - 1;

    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > NO_SYMM && fabs(Z[0]) < dZ) kmin = -3;
    if (symmetry == OCTANT && fabs(X[0]) < dX) imin = -3;
    if (symmetry == OCTANT && fabs(Y[0]) < dY) jmin = -3;

    // The original KO formula is defined only when its complete 7-point
    // box exists.  Most points in the compact outer shell therefore return
    // zero.  Test that condition before building SoA/lambda state or calling
    // any symmetry accessor; the numerical path for the remaining points is
    // unchanged below.
    if (i - 3 < imin || i + 3 > imax ||
        j - 3 < jmin || j + 3 > jmax ||
        k - 3 < kmin || k + 3 > kmax) {
        return 0.0;
    }

    double SoA[3] = {SYM1, SYM2, SYM3};

    const auto fh = [&](int ii, int jj, int kk) -> double {
        if (ii >= 0 && ii < ex[0] && jj >= 0 && jj < ex[1] && kk >= 0 && kk < ex[2]) {
            return f[ii + ex[0] * (jj + ex[1] * kk)];
        }
        return d_symmetry_bd_1b(3, ex, f, ii + 1, jj + 1, kk + 1, SoA);
    };

    double rhs_add = 0.0;

    // Same direct-access fast path as the first/second derivative stencils:
    // an interior point cannot invoke reflection or out-of-range handling.
    if (i >= 3 && i + 3 <= imax &&
        j >= 3 && j + 3 <= jmax &&
        k >= 3 && k + 3 <= kmax) {
        const int sy = ex[0];
        const int sz = ex[0] * ex[1];
        const int p = i + sy * j + sz * k;
        rhs_add = eps / cof * (
            ((f[p - 3] + f[p + 3]) - SIX * (f[p - 2] + f[p + 2]) +
             FIT * (f[p - 1] + f[p + 1]) - TWT * f[p]) / dX +
            ((f[p - 3 * sy] + f[p + 3 * sy]) - SIX * (f[p - 2 * sy] + f[p + 2 * sy]) +
             FIT * (f[p - sy] + f[p + sy]) - TWT * f[p]) / dY +
            ((f[p - 3 * sz] + f[p + 3 * sz]) - SIX * (f[p - 2 * sz] + f[p + 2 * sz]) +
             FIT * (f[p - sz] + f[p + sz]) - TWT * f[p]) / dZ
        );
        return rhs_add;
    }

    rhs_add = eps / cof * (
        ((fh(i-3,j,k) + fh(i+3,j,k)) - SIX*(fh(i-2,j,k) + fh(i+2,j,k)) +
         FIT*(fh(i-1,j,k) + fh(i+1,j,k)) - TWT*fh(i,j,k)) / dX +
        ((fh(i,j-3,k) + fh(i,j+3,k)) - SIX*(fh(i,j-2,k) + fh(i,j+2,k)) +
         FIT*(fh(i,j-1,k) + fh(i,j+1,k)) - TWT*fh(i,j,k)) / dY +
        ((fh(i,j,k-3) + fh(i,j,k+3)) - SIX*(fh(i,j,k-2) + fh(i,j,k+2)) +
         FIT*(fh(i,j,k-1) + fh(i,j,k+1)) - TWT*fh(i,j,k)) / dZ
    );

    return rhs_add;
}
