#!/bin/bash
#HPC --partition=lab4g10
#HPC --cpu=16
#HPC --gpu=1
#HPC --mem=24Gi
#HPC --time=30m
set -euo pipefail

echo "PWD=$(pwd)"
echo "Start GPU host perf: $(date)"

JOB_TAG="${HPC_JOB_ID:-${SLURM_JOB_ID:-${JOB_ID:-%j}}}"

input_backup="AMSS_NCKU_Input.py.4gssh.$$"
cp AMSS_NCKU_Input.py "$input_backup"
trap 'cp "$input_backup" AMSS_NCKU_Input.py; rm -f "$input_backup"' EXIT
sed -i '/^[[:space:]]*GPU_Calculation[[:space:]]*=/c\GPU_Calculation  = "yes"' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*Final_Evolution_Time[[:space:]]*=/c\Final_Evolution_Time = 1.0' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*Evolution_Step_Number[[:space:]]*=/c\Evolution_Step_Number = 1' AMSS_NCKU_Input.py

# Force the one-GPU baseline. Linux perf observes host/MPI/CUDA-runtime work;
# use Nsight Systems/Compute separately for CUDA kernel profiling.
export AMSS_GPU_CALCULATION=yes
export AMSS_MPI_PROCESSES=1
export AMSS_OMP_THREADS=1
export AMSS_EVOLUTION_STEPS=1
export AMSS_FINAL_EVOLUTION_TIME=1.0
export AMSS_ENABLE_GPU=ON
export AMSS_ENABLE_OPENMP=OFF
export AMSS_CUDA_ARCHITECTURES=80
export AMSS_BUILD_DIR="$PWD/build_gpu_perf"
export AMSS_OUTPUT_ROOT="$PWD/gpu_perf_output"
export AMSS_CACHE_DIR="$PWD/twopuncture_cache"
export AMSS_OPT="-O3 -g"
export CUDACXX=/usr/local/cuda/bin/nvcc
[[ -x "$CUDACXX" ]] || { echo "CUDA compiler missing: $CUDACXX" >&2; exit 1; }

PERF_DIR="$PWD/results/perf_gpu-$JOB_TAG"
NSYS_PREFIX="$PERF_DIR/nsys_gpu-$JOB_TAG"
mkdir -p "$AMSS_OUTPUT_ROOT" "$PERF_DIR"
nvidia-smi -L

echo "Mode=GPU host perf (ABEGPU), MPI ranks=$AMSS_MPI_PROCESSES, OMP threads=$AMSS_OMP_THREADS"
echo "Evolution steps=$AMSS_EVOLUTION_STEPS"
echo "Final evolution time=$AMSS_FINAL_EVOLUTION_TIME"
echo "Build=$AMSS_BUILD_DIR"
echo "Perf output=$PERF_DIR"
echo "Nsight Systems output=${NSYS_PREFIX}.nsys-rep"

./compile.sh

# GPU perf is host-side profiling (MPI/CUDA runtime); keep one rank and a
# low-rate sample stream. Disable DWARF unwinding because the container's
# perf unwinder prints "get_proc_name unsupported" for every sample. CUDA
# kernels require Nsight Systems/Compute.
perf stat -d -o "$PERF_DIR/perf-stat.txt" -- \
  perf record -F 5 -m 8 \
    -o "$PERF_DIR/perf.data" -- ./run.sh --twop-cache
if perf report --stdio -i "$PERF_DIR/perf.data" > "$PERF_DIR/perf-report.txt"; then
  perf script -i "$PERF_DIR/perf.data" > "$PERF_DIR/perf-script.txt"
else
  echo "perf report produced no samples; perf-stat.txt remains valid." >&2
fi

# Nsight Systems is a separate one-evolve collection and captures CUDA API,
# kernel, MPI and CPU scheduling timelines. It is optional by image.
if command -v nsys >/dev/null 2>&1; then
  nsys profile --force-overwrite=true --trace=cuda,nvtx,mpi,osrt \
    --sample=process-tree --cpuctxsw=none \
    -o "$NSYS_PREFIX" -- ./run.sh --twop-cache
  nsys stats --force-export=true -o "$PERF_DIR/nsys-stats-$JOB_TAG" "$NSYS_PREFIX.nsys-rep" \
    > "$PERF_DIR/nsys-stats-$JOB_TAG.txt" 2>&1 || true
else
  echo "nsys not found; skipped Nsight Systems collection." >&2
fi

echo "GPU host perf analysis finished: $PERF_DIR"
