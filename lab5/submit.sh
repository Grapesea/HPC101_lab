#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR_ARG="${1:-${MODEL_DIR:-}}"
QUANT_DIR_ARG="${2:-${QUANT_DIR:-}}"

if [[ -z "$MODEL_DIR_ARG" || -z "$QUANT_DIR_ARG" ]]; then
    echo "Usage: $0 MODEL_DIR QUANT_DIR" >&2
    echo "Alternatively export MODEL_DIR and QUANT_DIR before running it." >&2
    exit 2
fi
if ! command -v hpc >/dev/null 2>&1; then
    echo "hpc command was not found; run this script on the cluster login node." >&2
    exit 127
fi

mkdir -p "$SCRIPT_DIR/results"
printf -v JOB_COMMAND 'bash %q %q %q' \
    "$SCRIPT_DIR/run_baseline.sh" "$MODEL_DIR_ARG" "$QUANT_DIR_ARG"

hpc submit \
    -p lab5 \
    -g 1 \
    -t 30m \
    -d \
    -o "results/lab5-baseline-%j.log" \
    "$JOB_COMMAND"
