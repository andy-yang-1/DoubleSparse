#!/bin/bash

bszs=(1 4 8 16 32)
ctxs=(2048 4096 8192 16384)

for bsz in "${bszs[@]}"; do
    for ctx in "${ctxs[@]}"; do
        python3 ../models/triton_kernels/attention.py $bsz $ctx
    done
done
