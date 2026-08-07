#!/bin/bash
#HPC --partition=lab3
#HPC --cpu=8
#HPC --mem=32Gi
#HPC --time=5m

NAME="${NAME:-default_name}"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p results
exec > "results/${DATE}.log" 2>&1

python run.py --reference-benchmarks

nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --stats=true --force-overwrite=true -o ${NAME} python run.py --case parallel_gva
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --report cuda_gpu_mem_time_sum --force-export=true --format table ${NAME}.nsys-rep