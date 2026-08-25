#!/bin/bash
#HPC --partition=lab4g10
#HPC --cpu=16
#HPC --gpu=1
#HPC --mem=24Gi
#HPC --time=15m
set -euo pipefail

ranks="${PROBE_MPI:-1}"
threads="${PROBE_OMP:-1}"
steps="${PROBE_STEPS:-5}"

input_backup="$(mktemp AMSS_NCKU_Input.py.probe.XXXXXX)"
cp AMSS_NCKU_Input.py "$input_backup"
trap 'cp "$input_backup" AMSS_NCKU_Input.py; rm -f "$input_backup"' EXIT
sed -i "/^[[:space:]]*Final_Evolution_Time[[:space:]]*=/c\Final_Evolution_Time = ${steps}.0" AMSS_NCKU_Input.py
sed -i "/^[[:space:]]*Evolution_Step_Number[[:space:]]*=/c\Evolution_Step_Number = ${steps}" AMSS_NCKU_Input.py
sed -i "/^[[:space:]]*MPI_processes[[:space:]]*=/c\MPI_processes = ${ranks}" AMSS_NCKU_Input.py
sed -i "/^[[:space:]]*OMP_threads[[:space:]]*=/c\OMP_threads = ${threads}" AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*GPU_Calculation[[:space:]]*=/c\GPU_Calculation = "yes"' AMSS_NCKU_Input.py

export AMSS_BUILD_DIR="$PWD/build_gpu_a100"
export AMSS_OUTPUT_ROOT="$PWD/probe-output-m${ranks}-o${threads}"
export AMSS_CACHE_DIR="$PWD/twopuncture_cache"
export AMSS_MPIEXEC="mpiexec --allow-run-as-root --bind-to none"
export AMSS_ENABLE_GPU=ON AMSS_CUDA_ARCHITECTURES=80
export AMSS_ENABLE_OPENMP=OFF AMSS_MPI_CUDA_AWARE=0
export AMSS_OPT='-Ofast -ffast-math'
export CUDACXX=/usr/local/cuda/bin/nvcc

echo "PROBE MPI=${ranks} OMP=${threads} STEPS=${steps}"
./compile.sh
./run.sh --twop-cache
