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

# hpc 默认在提交命令所在目录运行
echo "PWD=$(pwd)"
echo "Start: $(date)"

input_backup="AMSS_NCKU_Input.py.4sh.$$"
cp AMSS_NCKU_Input.py "$input_backup"
trap 'rc=$?; cp "$input_backup" AMSS_NCKU_Input.py; rm -f "$input_backup"; report_total_time "$rc"; exit "$rc"' EXIT
sed -i '/^[[:space:]]*GPU_Calculation[[:space:]]*=/c\GPU_Calculation  = "no"' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*Final_Evolution_Time[[:space:]]*=/c\Final_Evolution_Time = 40.0' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*Evolution_Step_Number[[:space:]]*=/c\Evolution_Step_Number = 10000000' AMSS_NCKU_Input.py

# 4.sh always means the CPU baseline. Keep the source configuration and
# environment consistent, even if a previous GPU run left it set to "yes".
export AMSS_GPU_CALCULATION=no
unset AMSS_FINAL_EVOLUTION_TIME AMSS_EVOLUTION_STEPS
export AMSS_MPI_PROCESSES=30
export AMSS_OMP_THREADS=1
# The Python driver reads MPI_processes / OMP_threads from AMSS_NCKU_Input.py
# (not from these env vars), so sed them into the input file to keep the two
# in sync. The trap above restores the original input file on exit.
sed -i "/^[[:space:]]*MPI_processes[[:space:]]*=/c\\MPI_processes = $AMSS_MPI_PROCESSES" AMSS_NCKU_Input.py
sed -i "/^[[:space:]]*OMP_threads[[:space:]]*=/c\\OMP_threads = $AMSS_OMP_THREADS" AMSS_NCKU_Input.py
export AMSS_ENABLE_GPU=OFF
export AMSS_ENABLE_OPENMP=ON
export AMSS_ENABLE_SANITY_CHECK=OFF
# lab4 sweep: TwoPuncture relaxation plateaus at 24 threads (82s).
export AMSS_TWOP_THREADS=24
export AMSS_OPT="-Ofast -ffast-math"
export AMSS_ARCH_FLAGS="-mcpu=native"
GNU_BUILD="$PWD/build_cpu_gnu"
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

# The final executable directory contains Arm-compiled ABE together with the
# GNU-compiled TwoPunctureABE.  Both use OpenMP, but the GNU binary has no
# compiler RPATH; hpc jobs do not necessarily inherit the interactive
# container's LD_LIBRARY_PATH.  Export the compiler runtimes explicitly so
# the staged GNU solver can load libomp.so (and libstdc++) before run.sh
# forwards the environment to MPI ranks.
ARM_COMPILER_ROOT="/opt/arm/arm-linux-compiler-24.10.1_Ubuntu-22.04"
ARM_GCC_ROOT="/opt/arm/gcc-14.2.0_Ubuntu-22.04"
export LD_LIBRARY_PATH="$ARM_COMPILER_ROOT/lib:$ARM_GCC_ROOT/lib64:${LD_LIBRARY_PATH:-}"

echo "Mode=CPU (ABE), MPI ranks=$AMSS_MPI_PROCESSES, OMP threads=$AMSS_OMP_THREADS"
echo "Build=$AMSS_BUILD_DIR"
echo "OutputRoot=$AMSS_OUTPUT_ROOT"

# The two stages prefer different compilers on Kunpeng 920B: GNU is much
# faster for TwoPuncture's line relaxation, while Arm Compiler wins on the
# Fortran-heavy ABE evolution. Build both, then stage the GNU initial solver
# beside the Arm-compiled ABE for the normal driver.
env CXX=/usr/bin/c++ FC=/usr/bin/gfortran \
    AMSS_BUILD_DIR="$GNU_BUILD" ./compile.sh
env CXX=/opt/arm/arm-linux-compiler-24.10.1_Ubuntu-22.04/bin/armclang++ \
    FC=/opt/arm/arm-linux-compiler-24.10.1_Ubuntu-22.04/bin/armflang \
    AMSS_BUILD_DIR="$ARM_BUILD" ./compile.sh
cp "$GNU_BUILD/TwoPunctureABE" "$ARM_BUILD/TwoPunctureABE"

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
