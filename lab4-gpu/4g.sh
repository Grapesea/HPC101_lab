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

input_backup="$(mktemp AMSS_NCKU_Input.py.4gsh.XXXXXX)"
cp AMSS_NCKU_Input.py "$input_backup"
trap 'cp "$input_backup" AMSS_NCKU_Input.py; rm -f "$input_backup"' EXIT
sed -i '/^[[:space:]]*Final_Evolution_Time[[:space:]]*=/c\Final_Evolution_Time = 100.0' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*Evolution_Step_Number[[:space:]]*=/c\Evolution_Step_Number = 10000000' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*MPI_processes[[:space:]]*=/c\MPI_processes = 1' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*OMP_threads[[:space:]]*=/c\OMP_threads = 16' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*GPU_Calculation[[:space:]]*=/c\GPU_Calculation = "yes"' AMSS_NCKU_Input.py

# 4g.sh always means the one-GPU baseline. Keep the source configuration and
# environment consistent, even if a previous CPU run left it set to "no".
export AMSS_GPU_CALCULATION=yes
unset AMSS_FINAL_EVOLUTION_TIME AMSS_EVOLUTION_STEPS
export AMSS_MPI_PROCESSES=1
export AMSS_OMP_THREADS=16
export AMSS_ENABLE_GPU=ON
export AMSS_ENABLE_OPENMP=OFF
export AMSS_CUDA_ARCHITECTURES=80
export AMSS_BUILD_DIR="$PWD/build_gpu_a100"
export AMSS_OUTPUT_ROOT="$PWD/gpu_output"
export AMSS_CACHE_DIR="$PWD/twopuncture_cache"
export CUDACXX=/usr/local/cuda/bin/nvcc
[[ -x "$CUDACXX" ]] || { echo "CUDA compiler missing: $CUDACXX" >&2; exit 1; }
mkdir -p "$AMSS_OUTPUT_ROOT"
nvidia-smi -L

echo "GPU_Calculation=$AMSS_GPU_CALCULATION"
echo "MPI ranks=$AMSS_MPI_PROCESSES, OMP threads=$AMSS_OMP_THREADS"
echo "Build=$AMSS_BUILD_DIR"
echo "Output=$AMSS_OUTPUT_ROOT"
echo "CUDA compiler=${CUDACXX:-disabled}"

./compile.sh
mkdir -p results
./run.sh --twop-cache > results/full-inner.log 2>&1
tail -120 results/full-inner.log
./check.sh "$AMSS_OUTPUT_ROOT/GW250118/AMSS_NCKU_output" golden
