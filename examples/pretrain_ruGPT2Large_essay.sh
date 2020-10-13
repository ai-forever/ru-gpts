#! /bin/bash

NUM_GPUS_PER_WORKER=1

now=$(date +"%Y_%m_%d_%H_%I_%S")
host=$(hostname)

mpirun --np ${NUM_GPUS_PER_WORKER} python pretrain_transformers.py \
    --output_dir=/home/jovyan/gpt2_large_bbpe_v50/essays/checkpoints_"${now}"_"${host}" \
    --model_type=gpt2 \
    --model_name_or_path=/home/jovyan/gpt2_large_bbpe_v50 \
    --do_train \
    --train_data_file=/home/jovyan/data/all_essays.txt \
    --do_eval \
    --eval_data_file=/home/jovyan/data/valid_essays.txt \
    --fp16 \
    --num_train_epochs 10 \
    --overwrite_cache \
    --block_size=1024 \
    --per_gpu_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
