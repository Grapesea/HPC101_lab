#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 MODEL_DIR QUANT_DIR" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR_ARG="$1"
QUANT_DIR_ARG="$2"

if [[ ! -d "$MODEL_DIR_ARG" ]]; then
    echo "Model directory does not exist: $MODEL_DIR_ARG" >&2
    exit 2
fi

cd "$SCRIPT_DIR"
mkdir -p results

# PYTHONPATH makes the source tree immediately importable. The editable install
# also registers dependencies/metadata when the Lab5 image provides uv.
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
if command -v uv >/dev/null 2>&1; then
    uv pip install -e . --no-deps
else
    python3 -m pip install -e . --no-deps
fi

python3 scripts/check_int4_triton.py

if [[ -f "$QUANT_DIR_ARG/manifest.json" && \
      -f "$QUANT_DIR_ARG/quantization_config.json" ]]; then
    echo "Reusing complete quantized checkpoint: $QUANT_DIR_ARG"
else
    python3 scripts/quantize.py \
        --config config.yaml \
        --model "$MODEL_DIR_ARG" \
        --output "$QUANT_DIR_ARG" \
        --device cuda
fi

if [[ -f results/gptq-public-quality.json ]]; then
    echo "Reusing existing quality result: results/gptq-public-quality.json"
else
    python3 scripts/evaluate_quality.py \
        --model "$QUANT_DIR_ARG" \
        --dataset datasets/quality_public.jsonl \
        --output results/gptq-public-quality.json \
        --reference results/bf16-public-quality.json \
        --max-delta-nll 0.16 \
        --linear-backend int4_reference \
        --device cuda \
        --dtype bfloat16 \
        --max-sequence-length 2048 \
        --chunk-size 128
fi

python3 scripts/run_generation_queue.py \
    --config config.yaml \
    --model "$QUANT_DIR_ARG" \
    --input datasets/performance_public.jsonl \
    --output results/gptq-public-performance.jsonl \
    --summary-output results/gptq-public-summary.json \
    --device cuda \
    --linear-backend int4_triton \
    --batch-size 4 \
    --max-batch-size 4 \
    --max-sequence-length 2048
