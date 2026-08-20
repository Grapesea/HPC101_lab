#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
set -euo pipefail

# End-to-end wall-clock guard.  The OJ budget is 30 minutes; keep a 5-minute
# margin so a run that exceeds 25 minutes is visible before the scheduler
# kills it.  Set AMSS_TIME_LIMIT_SECONDS to override the reporting threshold.
START_EPOCH=$(date +%s)
AMSS_TIME_LIMIT_SECONDS="${AMSS_TIME_LIMIT_SECONDS:-1500}"
report_total_time() {
  local rc="${1:-0}" end_epoch elapsed whole_minutes remainder
  end_epoch=$(date +%s)
  elapsed=$((end_epoch - START_EPOCH))
  whole_minutes=$((elapsed / 60))
  remainder=$((elapsed % 60))
  echo "Total wall time: ${elapsed}s (${whole_minutes}m ${remainder}s)"
  if (( elapsed >= AMSS_TIME_LIMIT_SECONDS )); then
    echo "WARNING: total wall time exceeded ${AMSS_TIME_LIMIT_SECONDS}s (${AMSS_TIME_LIMIT_SECONDS} seconds / 25-minute target)." >&2
  else
    echo "Time budget remaining: $((AMSS_TIME_LIMIT_SECONDS - elapsed))s"
  fi
  return "$rc"
}
echo "PWD=$(pwd)"
echo "Start: $(date)"

input_backup="AMSS_NCKU_Input.py.4sh.$$"
cp AMSS_NCKU_Input.py "$input_backup"
trap 'rc=$?; cp "$input_backup" AMSS_NCKU_Input.py; rm -f "$input_backup"; report_total_time "$rc"; exit "$rc"' EXIT
sed -i '/^[[:space:]]*GPU_Calculation[[:space:]]*=/c\GPU_Calculation  = "no"' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*Final_Evolution_Time[[:space:]]*=/c\Final_Evolution_Time = 40.0' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*Evolution_Step_Number[[:space:]]*=/c\Evolution_Step_Number = 10000000' AMSS_NCKU_Input.py
# The last complete, checker-passing lab4 runs used 30 MPI ranks and one
# OpenMP thread (about 1319--1321 s).  Keep that measured configuration as the
# submission default; a 5x12 experiment can be requested explicitly without
# editing this script.
SUBMIT_MPI_PROCESSES="${AMSS_SUBMIT_MPI_PROCESSES:-30}"
SUBMIT_OMP_THREADS="${AMSS_SUBMIT_OMP_THREADS:-1}"
sed -i "/^[[:space:]]*MPI_processes[[:space:]]*=/c\\MPI_processes    = ${SUBMIT_MPI_PROCESSES}" AMSS_NCKU_Input.py
sed -i "/^[[:space:]]*OMP_threads[[:space:]]*=/c\\OMP_threads      = ${SUBMIT_OMP_THREADS}" AMSS_NCKU_Input.py

# 4.sh always means the CPU baseline. Keep the source configuration and
# environment consistent, even if a previous GPU run left it set to "yes".
# Build-time settings (GPU off, OpenMP on, sanity check off, -Ofast,
# -mcpu=native) are CMakeLists.txt defaults; the hybrid toolchain is
# compile.sh's default, so only runtime knobs are exported here.
export AMSS_GPU_CALCULATION=no
unset AMSS_FINAL_EVOLUTION_TIME AMSS_EVOLUTION_STEPS
export AMSS_MPI_PROCESSES="$SUBMIT_MPI_PROCESSES"
export AMSS_OMP_THREADS="$SUBMIT_OMP_THREADS"
# lab4 sweep: TwoPuncture relaxation plateaus at 20 threads.
export AMSS_TWOP_THREADS=20
# Keep the scheduler/MPI default affinity for the measured 30x1 path.  Binding
# OpenMP places while every MPI rank is unbound can make all 30 one-thread
# ranks land on the same place, so only enable these variables for a true
# hybrid request.
export AMSS_MPIEXEC="${AMSS_SUBMIT_MPIEXEC:-mpiexec --allow-run-as-root}"
if (( SUBMIT_OMP_THREADS > 1 )); then
  export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
  export OMP_PLACES="${OMP_PLACES:-threads}"
  export OMP_WAIT_POLICY=ACTIVE
  export GOMP_SPINCOUNT=INFINITE
else
  unset OMP_PROC_BIND OMP_PLACES OMP_WAIT_POLICY GOMP_SPINCOUNT
fi
export OPENBLAS_NUM_THREADS=1
# Scoring only reads the numerical output files; figures are skipped so the
# wall budget is not spent on matplotlib (see AMSS_NCKU_Program.py).
export AMSS_SKIP_PLOTS=1
# The complete passing submissions were built with -mcpu=native.  This lets
# the compiler select the actual node CPU; override with
# AMSS_SUBMIT_ARCH_FLAGS='-mcpu=tsv110 -msve-vector-bits=256' only after a
# full checker-passing comparison on the target partition.
export AMSS_OPT="-Ofast -ffast-math"
export AMSS_ARCH_FLAGS="${AMSS_SUBMIT_ARCH_FLAGS:--mcpu=native}"
ARM_BUILD="$PWD/build_cpu_armclang"
export AMSS_BUILD_DIR="$ARM_BUILD"
# Isolate each submission's output directory by scheduler job id, so two
# overlapping/leftover runs never collide on the same $PWD/GW250118.
JOB_TAG="${HPC_JOB_ID:-${SLURM_JOB_ID:-${JOB_ID:-}}}"
if [[ -z "$JOB_TAG" ]]; then
  JOB_TAG="$(date +%Y%m%d-%H%M%S-%N)"
fi
export AMSS_OUTPUT_ROOT="$PWD/runs/${JOB_TAG}"
mkdir -p "$AMSS_OUTPUT_ROOT"
export AMSS_CACHE_DIR="$PWD/twopuncture_cache"
unset CUDACXX AMSS_CUDA_ARCHITECTURES

echo "Mode=CPU (ABE), MPI ranks=$AMSS_MPI_PROCESSES, OMP threads=$AMSS_OMP_THREADS"
echo "Build=$AMSS_BUILD_DIR"
echo "OutputRoot=$AMSS_OUTPUT_ROOT"

# The two stages prefer different compilers on Kunpeng 920B: GNU is much
# faster for TwoPuncture's line relaxation, while Arm Compiler wins on the
# Fortran-heavy ABE evolution. compile.sh defaults to that hybrid build:
# GNU compiles TwoPunctureABE into ${AMSS_BUILD_DIR}_gnu_twop, Arm Compiler
# compiles ABE into AMSS_BUILD_DIR, and it stages the GNU solver beside the
# Arm-built ABE. The GNU binary's runtime libs are baked in via RPATH
# (AMSS_EXTRA_RPATH), and run.sh re-exports LD_LIBRARY_PATH at run time.
./compile.sh

# Formal timing must include the uncached TwoPuncture stage.
./run.sh

# The CPU run stops at t=40, while the shipped golden trajectory extends to
# t=100.  Build the matching golden prefix from the configured final time.
FINAL_TIME=$(python3 -c 'import AMSS_NCKU_Input as cfg; print(cfg.Final_Evolution_Time)')
mkdir -p golden_cpu
awk -v cutoff="$FINAL_TIME" '
  /^[[:space:]]*#/ || NF == 0 { print; next }
  ($1 + 0) < cutoff { print }
' golden/bssn_BH.dat > golden_cpu/bssn_BH.dat

./check.sh "$AMSS_OUTPUT_ROOT/GW250118/AMSS_NCKU_output" golden_cpu
