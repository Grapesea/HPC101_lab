#!/bin/bash
#HPC --partition=lab4
#HPC --cpu=60
#HPC --mem=100Gi
#HPC --time=30m

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p results
exec > "results/${DATE}.log" 2>&1

./compile.sh
./run.sh --twop-cache
./check.sh