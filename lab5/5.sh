#!/bin/sh
set -eu

# 非交互地提交完整 baseline。不要先申请 interactive shell；实际计算由
# submit.sh 提交的 run_baseline.sh 在 Lab5 GPU 节点中完成。
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL_DIR_ARG="${1:-${MODEL_DIR:-/checkpoints/gemma-4-12b}}"
QUANT_DIR_ARG="${2:-${QUANT_DIR:-/results/quantized-gemma-4}}"

exec bash "$SCRIPT_DIR/submit.sh" "$MODEL_DIR_ARG" "$QUANT_DIR_ARG"
