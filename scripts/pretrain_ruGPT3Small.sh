#! /bin/bash

NUM_GPUS_PER_WORKER=1

now=$(date +"%Y_%m_%d_%H_%I_%S")
host=$(hostname)

mpirun --np ${NUM_GPUS_PER_WORKER} python pretrain_transformers.py \
    --output_dir=/home/jovyan/rugpt2large/checkpoints_"${now}"_"${host}" \
    --model_type=gpt2 \
    --model_name_or_path=sberbank-ai/rugpt3small_based_on_gpt2 \
    --do_train \
    --train_data_file=/home/jovyan/data/train.txt \
    --do_eval \
    --eval_data_file=/home/jovyan/data/valid.txt \
    --fp16 \
    --per_gpu_train_batch_size 1 \
    --gradient_accumulation_steps 1
