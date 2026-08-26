
#ifndef BLOCK_H
#define BLOCK_H

#include <mpi.h>
#include "macrodef.h" //need dim here; Vertex or Cell
#include "var.h"
#include "MyList.h"

#ifdef USE_GPU
#include "gpu_manager.h"   // brings in cudaStream_t / cuda_runtime.h
#endif

class Block
{

public:
   int shape[dim];
   double bbox[2 * dim];
   double *X[dim];
   int rank; // where the real data locate in
   int lev, cgpu;
   int ingfs, fngfs;
   int *(*igfs);
   double *(*fgfs); // fine grid functions

#ifdef USE_GPU
   // GPU Shadow pointers and valid flags
   double *d_X[dim];
   double *(*d_fgfs);
   bool *cpu_valid;
   bool *gpu_valid;
   cudaStream_t stream;
   // Reusable device-side tables used when a transfer applies the same AMR
   // geometry to the 24 independent BSSN state fields.  They belong to the
   // block's stream, so the asynchronous table update and batch kernel keep
   // their natural ordering without a device-wide synchronization.
   double **d_batch_src;
   double **d_batch_dst;
   double **d_batch_rhs;
   double *d_batch_soa;
#endif

public:
   Block() {};
   Block(int DIM, int *shapei, double *bboxi, int ranki, int ingfsi, int fngfs, int levi, const int cgpui = 0);

   ~Block();

   void checkBlock();

   double getdX(int dir);
   void swapList(MyList<var> *VarList1, MyList<var> *VarList2, int myrank);

#ifdef USE_GPU
   void require_on_gpu(int var_index);
   void require_on_cpu(int var_index);
   void mark_gpu_modified(int var_index);
   void mark_cpu_modified(int var_index);

   void move_to_gpu(MyList<var> *VarList);
   void move_to_cpu(MyList<var> *VarList);
#endif
};

#endif /* BLOCK_H */
