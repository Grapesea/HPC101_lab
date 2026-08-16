#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=5m
set -euo pipefail

ROOT="$(pwd)"
TEST_DIR="$(mktemp -d)"
cp "$ROOT/twop_sweep_results/TwoPunctureinput.par" "$TEST_DIR/TwoPunctureinput.par"

ARM_COMPILER_ROOT="/opt/arm/arm-linux-compiler-24.10.1_Ubuntu-22.04"
ARM_GCC_ROOT="/opt/arm/gcc-14.2.0_Ubuntu-22.04"
export LD_LIBRARY_PATH="$ARM_COMPILER_ROOT/lib:$ARM_GCC_ROOT/lib64"
export AMSS_TWOP_THREADS=1

set +e
(cd "$TEST_DIR"; timeout 8s "$ROOT/build_cpu_gnu/TwoPunctureABE" < /dev/null)
rc=$?
set -e

# timeout(1) returns 124 when the executable started normally.  A loader
# failure returns 127 and must fail this smoke test.
if [[ "$rc" != 124 && "$rc" != 0 ]]; then
  echo "RUNTIME_SMOKE_FAIL rc=$rc"
  exit "$rc"
fi
echo "RUNTIME_SMOKE_PASS rc=$rc"
