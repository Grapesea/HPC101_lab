
#ifndef RUNGEKUTTA4_H
#define RUNGEKUTTA4_H

#ifdef fortran1
#define f_euler_rout euler_rout
#define f_rungekutta4_rout rungekutta4_rout
#define f_rungekutta4_scalar rungekutta4_scalar
#define f_icn_rout icn_rout
#define f_icn_scalar icn_scalar
#endif
#ifdef fortran2
#define f_euler_rout EULER_ROUT
#define f_rungekutta4_rout RUNGEKUTTA4_ROUT
#define f_rungekutta4_scalar RUNGEKUTTA4_SCALAR
#define f_icn_rout ICN_ROUT
#define f_icn_scalar ICN_SCALAR
#endif
#ifdef fortran3
#define f_euler_rout euler_rout_
#define f_rungekutta4_rout rungekutta4_rout_
#define f_rungekutta4_scalar rungekutta4_scalar_
#define f_icn_rout icn_rout_
#define f_icn_scalar icn_scalar_
#endif

extern "C"
{
	void f_rungekutta4_scalar(double &, double &, double &, double &, int &);
}

extern "C"
{
	int f_rungekutta4_rout(int *, double &,
						   double *, double *, double *,
						   int &);
}

extern "C"
{
	void f_icn_scalar(double &, double &, double &, double &, int &);
}

extern "C"
{
	int f_icn_rout(int *, double &,
				   double *, double *, double *,
				   int &);
}

extern "C"
{
	int f_euler_rout(int *, double &,
					 double *, double *, double *);
}

#ifdef USE_GPU
#include <cuda_runtime.h>
void gpu_rungekutta4_rout_launch(
    cudaStream_t &stream,
    int ex[3], double dT,
    double* d_f0, double* d_f1, double* d_f_rhs, int RK4
);

// The BSSN state list has 24 independent fields.  On refined patches the
// correct-step Sommerfeld operation is only f1=f0 on the physical boundary,
// so fold that copy into the batched RK pass instead of launching another
// full-volume kernel for every field.
void gpu_rungekutta4_rout_batch_launch(
    cudaStream_t &stream,
    int ex[3], double dT,
    double** d_f0, double** d_f1, double** d_f_rhs, int RK4,
    bool fix_boundary,
    const double* d_X, const double* d_Y, const double* d_Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax, int Symmetry
);
#endif

#endif /* RUNGEKUTTA4_H */
