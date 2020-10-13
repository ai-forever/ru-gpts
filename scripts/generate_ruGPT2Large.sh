#! /bin/bash

NUM_GPUS_PER_WORKER=1

mpirun --np ${NUM_GPUS_PER_WORKER} python generate_transformers.py \
    --model_type=gpt2 \
    --model_name_or_path=/home/jovyan/gpt2_large_bbpe_v50 \
    --k=10 \
    --p=0.9 \
