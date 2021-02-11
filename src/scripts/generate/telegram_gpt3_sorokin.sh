#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

python ${script_dir}/../../deepspeed_megatron/telegram_bot.py \
       --model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1536 \
       --tg-token-name token_sorokin.txt \
       --num-attention-heads 16 \
       --load /home/jovyan/models/sorokin53200/mp_rank_00_model_states.pt \
       --no-load-optim \
       --max-position-embeddings 2048 \
       --tokenizer-path /home/jovyan/datasets/unpacked/gpt3_zmitrovich/_tokenizer/ \
       --make-vocab-size-divisible-by 1 \
       --fp16 \
       --cache-dir bookscache \
       --out-seq-length 2048 \
       --seq-length 2048 \
       --temperature 1.05 \
       --top_k 0 \
       --top_p 0.95 \
