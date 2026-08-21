#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m
# run.sh — run the AMSS-NCKU Lab 4 driver.
#
# Path layout (relative values resolve against the lab root):
#   AMSS_BUILD_DIR    build output          (default: <lab4>/build)
#   AMSS_OUTPUT_ROOT  run directory parent  (default: <lab4>)
#   AMSS_CACHE_DIR    TwoPuncture cache root (default: <lab4>/twopuncture_cache)
#   AMSS_MPIEXEC      MPI launcher          (default: mpiexec)
set -euo pipefail

# Ansorg-TwoPuncture allocates large Fortran automatic arrays.
ulimit -s unlimited

# compile.sh's default hybrid-toolchain build stages a GNU-compiled
# TwoPunctureABE (faster TwoPuncture relaxation) inside the Arm-compiled
# build tree (faster Fortran-heavy ABE evolution). That GNU binary's RPATH
# is already patched at link time (AMSS_EXTRA_RPATH in CMakeLists.txt), but
# hpc jobs do not reliably inherit the interactive container's
# LD_LIBRARY_PATH, so export the same paths here too as a second line of
# defense (see lab4.md "提交后的运行时链接修复"). Harmless no-op on any
# machine without the Arm Compiler installed.
ARM_COMPILER_ROOT="${AMSS_ARM_COMPILER_ROOT:-/opt/arm/arm-linux-compiler-24.10.1_Ubuntu-22.04}"
ARM_GCC_ROOT="${AMSS_ARM_GCC_ROOT:-/opt/arm/gcc-14.2.0_Ubuntu-22.04}"
extra_lib_path=""
for d in "$ARM_COMPILER_ROOT/lib" "$ARM_GCC_ROOT/lib64"; do
  [[ -d "$d" ]] || continue
  extra_lib_path="${extra_lib_path:+$extra_lib_path:}$d"
done
[[ -n "$extra_lib_path" ]] && export LD_LIBRARY_PATH="${extra_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# bssn_class::Step() parallelizes over grid blocks with OpenMP, and each
# worker thread runs compute_rhs_bssn which allocates ~35 MB of Fortran
# automatic arrays on the thread stack. `ulimit -s` only affects the main
# thread, so give OpenMP worker threads an explicit large stack too.
export OMP_STACKSIZE="${OMP_STACKSIZE:-1G}"

# HPC jobs run inside a root container even when submitted by a regular user.
# Open MPI 5.x (prterun) therefore needs its explicit container opt-in. These
# variables are ignored for non-root launches.
export OMPI_ALLOW_RUN_AS_ROOT="${OMPI_ALLOW_RUN_AS_ROOT:-1}"
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="${OMPI_ALLOW_RUN_AS_ROOT_CONFIRM:-1}"

ROOT_DIR="$(pwd)"
PYTHON="${PYTHON:-python3}"

resolve_under_root() {
  case "$1" in
    /*) printf '%s' "$1" ;;
     *) printf '%s/%s' "$ROOT_DIR" "$1" ;;
  esac
}

AMSS_BUILD_DIR="$(resolve_under_root "${AMSS_BUILD_DIR:-$ROOT_DIR/build}")"
AMSS_OUTPUT_ROOT="$(resolve_under_root "${AMSS_OUTPUT_ROOT:-$ROOT_DIR}")"
AMSS_CACHE_DIR="$(resolve_under_root "${AMSS_CACHE_DIR:-$ROOT_DIR/twopuncture_cache}")"
# The evaluation allocation exposes 30 physical cores as 60 hardware threads.
# A 12x5 run therefore has to bind five hardware threads per rank; CORE/PE=5
# asks for 60 physical cores and OpenMPI rejects the mapping with status 213.
AMSS_MPIEXEC="mpiexec --allow-run-as-root --use-hwthread-cpus --bind-to hwthread --map-by slot:PE=5"
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=threads
export AMSS_DISTRIBUTION_NODES="${AMSS_DISTRIBUTION_NODES:-30}"
# TwoPuncture's red/black plane relaxation is independent across planes.  The
# main ABE ranks keep their configured OMP_NUM_THREADS value; TwoPunctureABE
# reads this separate setting when compiled with OpenMP. The final twop20
# sweep selected 20 threads for the single-process initial-data solve.
export AMSS_TWOP_THREADS="${AMSS_TWOP_THREADS:-20}"
export AMSS_BUILD_DIR AMSS_OUTPUT_ROOT AMSS_CACHE_DIR AMSS_MPIEXEC AMSS_TWOP_THREADS
export AMSS_DISTRIBUTION_NODES

if [[ "${1:-}" == "--twop-cache" ]]; then
  export AMSS_NCKU_TWOP_CACHE=1
  shift
fi
if (( $# > 0 )); then
  echo "usage: ./run.sh [--twop-cache]" >&2
  exit 1
fi

echo "==> Build    : $AMSS_BUILD_DIR"
echo "==> Output   : $AMSS_OUTPUT_ROOT"
echo "==> Cache    : $AMSS_CACHE_DIR"
echo "==> MPI exec : $AMSS_MPIEXEC"

cd "$ROOT_DIR"
"$PYTHON" AMSS_NCKU_Program.py
