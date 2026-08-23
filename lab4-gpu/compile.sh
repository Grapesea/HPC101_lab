#!/bin/bash
# compile.sh — portable build driver for AMSS-NCKU Lab 4.
#
# GPU submission build for one A100 1g.10gb MIG instance.  The course target
# is fixed to CUDA sm_80; all three executables are rebuilt in one CMake tree.
set -euo pipefail

ROOT_DIR="$(pwd)"
CMAKE="${CMAKE:-cmake}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 2)}"

# Resolve build directory relative to the lab root.
BUILD_DIR="${AMSS_BUILD_DIR:-$ROOT_DIR/build}"
case "$BUILD_DIR" in
  /*) : ;;
   *) BUILD_DIR="$ROOT_DIR/$BUILD_DIR" ;;
esac

cmake_args=()
# MPI wrapper wins over plain CXX — the wrapper carries the right MPI flags
# and include paths. Setting both to different values is almost always a
# mistake, so warn instead of silently letting CXX override the wrapper.
if [[ -n "${MPI_CXX_COMPILER:-}" && -n "${CXX:-}" && "${MPI_CXX_COMPILER}" != "${CXX}" ]]; then
  echo "warning: both MPI_CXX_COMPILER ($MPI_CXX_COMPILER) and CXX ($CXX) are set;" >&2
  echo "         using MPI_CXX_COMPILER for the C++ compiler (MPI wrapper)." >&2
fi
if [[ -n "${MPI_CXX_COMPILER:-}" ]]; then
  cmake_args+=("-DCMAKE_CXX_COMPILER=$MPI_CXX_COMPILER")
elif [[ -n "${CXX:-}" ]]; then
  cmake_args+=("-DCMAKE_CXX_COMPILER=$CXX")
fi
[[ -n "${FC:-}" ]] && cmake_args+=("-DCMAKE_Fortran_COMPILER=$FC")
if [[ -n "${CUDACXX:-}" ]]; then
  cmake_args+=("-DCMAKE_CUDA_COMPILER=$CUDACXX")
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  cmake_args+=("-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc")
fi

# These options are deliberately explicit so an old CMake cache or inherited
# CPU settings cannot silently turn this submission into a CPU build.
cmake_args+=(
  "-DAMSS_ENABLE_GPU=ON"
  "-DCMAKE_CUDA_ARCHITECTURES=80"
  "-DAMSS_ENABLE_OPENMP=ON"
  "-DAMSS_MPI_CUDA_AWARE=0"
  "-DAMSS_OPT=${AMSS_OPT:--Ofast -ffast-math}"
  "-DAMSS_ARCH_FLAGS=${AMSS_ARCH_FLAGS:--march=native}"
)

echo "==> Configure: $CMAKE -S \"$ROOT_DIR\" -B \"$BUILD_DIR\" ${cmake_args[*]:-} $*"
"$CMAKE" -S "$ROOT_DIR" -B "$BUILD_DIR" "${cmake_args[@]}" "$@"

echo "==> Build: $CMAKE --build \"$BUILD_DIR\" -j $JOBS"
"$CMAKE" --build "$BUILD_DIR" -j "$JOBS"

echo "==> Built executables:"
ls -lh "$BUILD_DIR/TwoPunctureABE" "$BUILD_DIR/ABE"
if [[ -f "$BUILD_DIR/ABEGPU" ]]; then
  ls -lh "$BUILD_DIR/ABEGPU"
fi
