#!/bin/bash
#HPC --partition=lab4
# Profiling one MPI rank gives usable per-process hotspots and avoids
# multiplying perf's sample stream by 30 ranks.
#HPC --cpu=16
#HPC --mem=100Gi
#HPC --time=30m
set -euo pipefail

# End-to-end wall-clock guard for the profiling run.  Keep the report at the
# requested 25-minute threshold (1500 s), below the scheduler's 30-minute cap.
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
echo "Start CPU perf: $(date)"

# hpc expands %j in output paths; inside the container use the scheduler job
# variable when available, with a literal fallback for local execution.
JOB_TAG="${HPC_JOB_ID:-${SLURM_JOB_ID:-${JOB_ID:-%j}}}"

input_backup="AMSS_NCKU_Input.py.4ssh.$$"
cp AMSS_NCKU_Input.py "$input_backup"
trap 'rc=$?; cp "$input_backup" AMSS_NCKU_Input.py; rm -f "$input_backup"; report_total_time "$rc"; exit "$rc"' EXIT
sed -i '/^[[:space:]]*GPU_Calculation[[:space:]]*=/c\GPU_Calculation  = "no"' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*Final_Evolution_Time[[:space:]]*=/c\Final_Evolution_Time = 1.0' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*Evolution_Step_Number[[:space:]]*=/c\Evolution_Step_Number = 1' AMSS_NCKU_Input.py

# Force the CPU baseline and keep its profiled build/run separate.
export AMSS_GPU_CALCULATION=no
export AMSS_MPI_PROCESSES=1
export AMSS_OMP_THREADS=1
export AMSS_FINAL_EVOLUTION_TIME=1.0
export AMSS_EVOLUTION_STEPS=1
export AMSS_ENABLE_GPU=OFF
export AMSS_ENABLE_OPENMP=OFF
export AMSS_OPT="-Ofast -ffast-math"
export AMSS_ARCH_FLAGS="-mcpu=native"
export AMSS_BUILD_DIR="$PWD/build_cpu_perf"
export AMSS_OUTPUT_ROOT="$PWD/cpu_perf_output"
export AMSS_CACHE_DIR="$PWD/twopuncture_cache"
unset CUDACXX AMSS_CUDA_ARCHITECTURES

PERF_DIR="$PWD/results/perf_cpu-$JOB_TAG"
VTUNE_DIR="$PWD/results/vtune_cpu-$JOB_TAG"
mkdir -p "$AMSS_OUTPUT_ROOT" "$PERF_DIR"

echo "Mode=CPU perf (ABE), MPI ranks=$AMSS_MPI_PROCESSES, OMP threads=$AMSS_OMP_THREADS"
echo "Evolution steps=$AMSS_EVOLUTION_STEPS"
echo "Final evolution time=$AMSS_FINAL_EVOLUTION_TIME"
echo "Build=$AMSS_BUILD_DIR"
echo "Perf output=$PERF_DIR"
echo "VTune output=$VTUNE_DIR"

./compile.sh

# One simulation run produces both counter statistics and sampled call stacks.
# Low frequency and a bounded ring buffer avoid losing samples.  Disable
# DWARF unwinding: the cluster container's perf unwinder emits one warning per
# sample ("get_proc_name unsupported"). Flat samples still identify hotspots.
perf stat -d -o "$PERF_DIR/perf-stat.txt" -- \
  perf record -F 5 -m 8 \
    -o "$PERF_DIR/perf.data" -- ./run.sh --twop-cache
if perf report --stdio -i "$PERF_DIR/perf.data" > "$PERF_DIR/perf-report.txt"; then
  perf script -i "$PERF_DIR/perf.data" > "$PERF_DIR/perf-script.txt"
else
  echo "perf report produced no samples; perf-stat.txt remains valid." >&2
fi

# VTune is a separate one-evolve collection.  Keep it optional so the script
# remains usable on images where VTune is not installed.
if command -v vtune >/dev/null 2>&1; then
  rm -rf "$VTUNE_DIR"
  vtune -collect hotspots -result-dir "$VTUNE_DIR" -- ./run.sh --twop-cache
  vtune -report summary -result-dir "$VTUNE_DIR" -report-output "$VTUNE_DIR/summary.txt"
else
  echo "vtune not found; skipped VTune collection." >&2
fi

echo "CPU perf analysis finished: $PERF_DIR"
