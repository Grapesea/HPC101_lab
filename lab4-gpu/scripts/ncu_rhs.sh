#!/bin/bash
#HPC --partition=lab4g10
#HPC --cpu=16
#HPC --gpu=1
#HPC --mem=24Gi
#HPC --time=20m
set -euo pipefail

input_backup="$(mktemp "$PWD/AMSS_NCKU_Input.py.ncu.XXXXXX")"
cp AMSS_NCKU_Input.py "$input_backup"
trap 'cp "$input_backup" AMSS_NCKU_Input.py; rm -f "$input_backup"' EXIT
sed -i '/^[[:space:]]*Final_Evolution_Time[[:space:]]*=/c\Final_Evolution_Time = 1.0' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*Evolution_Step_Number[[:space:]]*=/c\Evolution_Step_Number = 1' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*MPI_processes[[:space:]]*=/c\MPI_processes = 1' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*OMP_threads[[:space:]]*=/c\OMP_threads = 16' AMSS_NCKU_Input.py

export AMSS_BUILD_DIR="$PWD/build_gpu_a100"
export AMSS_OUTPUT_ROOT="$PWD/ncu-output"
export AMSS_CACHE_DIR="$PWD/twopuncture_cache"
export AMSS_ENABLE_GPU=ON AMSS_CUDA_ARCHITECTURES=80 AMSS_ENABLE_OPENMP=OFF
export AMSS_OPT='-Ofast -ffast-math' CUDACXX=/usr/local/cuda/bin/nvcc
./compile.sh

mkdir -p "$AMSS_OUTPUT_ROOT/GW250118/AMSS_NCKU_output"
# Let the normal driver create the final parfile and executable copy, then
# profile the resulting ABEGPU command on a second cached invocation.
./run.sh --twop-cache >/dev/null
cd "$AMSS_OUTPUT_ROOT/GW250118/AMSS_NCKU_output"
ncu --target-processes all --clock-control none --kernel-name-base function \
    --kernel-name 'regex:rhs_kernel' --launch-skip 20 --launch-count 1 \
    --set full --csv --log-file "$OLDPWD/results/ncu-rhs.csv" \
    mpiexec --allow-run-as-root -n 1 ./ABEGPU </dev/null
