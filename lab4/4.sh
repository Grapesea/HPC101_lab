#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
set -euo pipefail

# hpc 默认在提交命令所在目录运行
echo "PWD=$(pwd)"
echo "Start: $(date)"

# Pin the CPU baseline even if the submit shell contains GPU overrides.
export AMSS_GPU_CALCULATION=no
export AMSS_MPI_PROCESSES=30
export AMSS_OMP_THREADS=1
export AMSS_ENABLE_GPU=OFF
export AMSS_ENABLE_OPENMP=OFF

./compile.sh
./run.sh --twop-cache

# The CPU run stops at t=40, while the shipped golden trajectory extends to
# t=100.  Build the matching golden prefix from the configured final time.
FINAL_TIME=$(python3 -c 'import AMSS_NCKU_Input as cfg; print(cfg.Final_Evolution_Time)')
mkdir -p golden_cpu
awk -v cutoff="$FINAL_TIME" '
  /^[[:space:]]*#/ || NF == 0 { print; next }
  ($1 + 0) < cutoff { print }
' golden/bssn_BH.dat > golden_cpu/bssn_BH.dat

./check.sh GW250118/AMSS_NCKU_output golden_cpu
