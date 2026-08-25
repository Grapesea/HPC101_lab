#!/bin/bash
#HPC --partition=lab4g10
#HPC --cpu=16
#HPC --gpu=1
#HPC --mem=24Gi
#HPC --time=15m
set -euo pipefail

source_dir="$PWD/probe-output-m4-o4/GW250118/AMSS_NCKU_output"
test -x "$source_dir/TwoPunctureABE"
test -f "$source_dir/TwoPunctureinput.par"

for spec in '16 cores close' '16 threads close' '16 cores spread' '12 cores close'; do
    read -r threads places binding <<<"$spec"
    probe_dir="$(mktemp -d /tmp/twop-omp-${threads}.XXXXXX)"
    cp "$source_dir/TwoPunctureABE" "$source_dir/TwoPunctureinput.par" "$probe_dir/"
    echo "TWOP_OMP=${threads} places=${places} bind=${binding}"
    (
        cd "$probe_dir"
        start_ns="$(date +%s%N)"
        OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE OMP_PROC_BIND="$binding" OMP_PLACES="$places" \
            ./TwoPunctureABE >/dev/null
        end_ns="$(date +%s%N)"
        awk -v t="$threads" -v p="$places" -v bind="$binding" -v a="$start_ns" -v b="$end_ns" \
            'BEGIN { printf "TWOP_TIME threads=%s places=%s bind=%s real=%.6f\n", t, p, bind, (b-a)/1e9 }'
    )
done
