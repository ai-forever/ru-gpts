#!/bin/bash
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
python ${script_dir}/../../run_rugpt3_training.py \
    --output_dir=models/gpt3_small/tmp1 \
    --train_data_file=/home/jovyan/datasets/unpacked/gpt3_zmitrovich/train \
    --tokenizer_name=/home/jovyan/datasets/unpacked/gpt3_zmitrovich/_tokenizer/ \
    --model_type=gpt3 \
    --block_size=2048 \
    --per_device_train_batch_size=2 \
    --max_files_load=500 \
    --do_train \
    --num_train_epochs=3 \
    --warmup_steps=2000 \
    --logging_steps=10 \
    --fp16 \
    --fp16_opt_level O1 \
    --save_total_limit 5 \
    --num_train_epochs 3 \
    --max_grad_norm 0.5
