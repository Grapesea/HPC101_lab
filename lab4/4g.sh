#!/bin/bash
#HPC --partition=lab4g10
#HPC --cpu=16
#HPC --gpu=1
#HPC --mem=24Gi
#HPC --time=30m
set -euo pipefail

# hpc 默认在提交命令所在目录运行
echo "PWD=$(pwd)"
echo "Start: $(date)"

# One A100 MIG baseline: one MPI rank, no OpenMP parallelism.
export AMSS_GPU_CALCULATION=yes
export AMSS_MPI_PROCESSES=1
export AMSS_OMP_THREADS=1
export AMSS_ENABLE_OPENMP=OFF

# Keep the x86/CUDA CMake cache and outputs separate from the ARM CPU run.
export AMSS_ENABLE_GPU=ON
export AMSS_CUDA_ARCHITECTURES=80
export AMSS_BUILD_DIR="$PWD/build_gpu"
export AMSS_OUTPUT_ROOT="$PWD/gpu_output"
mkdir -p "$AMSS_OUTPUT_ROOT"

echo "GPU_Calculation=$AMSS_GPU_CALCULATION"
echo "MPI ranks=$AMSS_MPI_PROCESSES, OMP threads=$AMSS_OMP_THREADS"
echo "Build=$AMSS_BUILD_DIR"
echo "Output=$AMSS_OUTPUT_ROOT"
nvidia-smi -L

./compile.sh
./run.sh --twop-cache
./check.sh GW250118/AMSS_NCKU_output golden
