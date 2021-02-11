#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

python ${script_dir}/../../deepspeed_megatron/telegram_bot.py \
       --model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1536 \
       --tg-token-name token.txt \
       --num-attention-heads 16 \
       --load /home/jovyan/models/gpt3/deepspeed/large/pretrained3_2020_10_15_14_02_03_eval2-0/8000/mp_rank_00_model_states.pt \
       --no-load-optim \
       --max-position-embeddings 2048 \
       --tokenizer-path /home/jovyan/datasets/unpacked/gpt3_zmitrovich/_tokenizer/ \
       --fp16 \
       --cache-dir cache \
       --out-seq-length 512 \
       --seq-length 512 \
       --temperature 0.9 \
       --top_k 0 \
       --top_p 0.95
