#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=12m
set -euo pipefail

ROOT="$(pwd)"
START_NS="$(date +%s%N)"
PERF_DIR="$ROOT/results/perf-5x12-${START_NS}"
RUN_ROOT="$ROOT/runs/perf-5x12-${START_NS}"
mkdir -p "$PERF_DIR" "$RUN_ROOT"

input_backup="AMSS_NCKU_Input.py.perf5x12.$$"
cp AMSS_NCKU_Input.py "$input_backup"
trap 'rc=$?; cp "$input_backup" AMSS_NCKU_Input.py; rm -f "$input_backup"; exit "$rc"' EXIT
sed -i '/^[[:space:]]*GPU_Calculation[[:space:]]*=/c\GPU_Calculation = "no"' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*Final_Evolution_Time[[:space:]]*=/c\Final_Evolution_Time = 2.0' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*Evolution_Step_Number[[:space:]]*=/c\Evolution_Step_Number = 2' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*MPI_processes[[:space:]]*=/c\MPI_processes = 5' AMSS_NCKU_Input.py
sed -i '/^[[:space:]]*OMP_threads[[:space:]]*=/c\OMP_threads = 12' AMSS_NCKU_Input.py

export AMSS_GPU_CALCULATION=no
export AMSS_BUILD_DIR="$ROOT/build_cpu_armclang"
export AMSS_OUTPUT_ROOT="$RUN_ROOT"
export AMSS_CACHE_DIR="$ROOT/twopuncture_cache"
export AMSS_NCKU_TWOP_CACHE=1
export AMSS_TWOP_THREADS=20
export AMSS_SKIP_PLOTS=1
export AMSS_MPIEXEC="mpiexec --allow-run-as-root --use-hwthread-cpus --bind-to hwthread --map-by slot:PE=12"
export OMP_PROC_BIND=close
export OMP_PLACES=threads
export OMP_WAIT_POLICY=ACTIVE
export GOMP_SPINCOUNT=INFINITE
export OPENBLAS_NUM_THREADS=1
export AMSS_PERF_PREFIX="$PERF_DIR/rank0"
export AMSS_ABE_WRAPPER="$ROOT/perf_rank0.sh"

echo "Perf config: MPI=5 OMP=12 TwoPuncture=20 cache=on steps=2"
echo "Perf output: $PERF_DIR"
./run.sh --twop-cache > "$PERF_DIR/driver.log" 2>&1

perf report -i "$PERF_DIR/rank0.data" --stdio > "$PERF_DIR/rank0-report.txt"
echo "Perf completed: $PERF_DIR"
rg -n "This Program Cost|Total Evolve Time|Trajectory RMS|FINAL:|Timestep #" "$PERF_DIR/driver.log" | tail -20
