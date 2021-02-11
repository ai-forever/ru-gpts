#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json="$script_dir/deepspeed_config/gpt3_xl_sparse_2048.json"

python ${script_dir}/../../deepspeed_megatron/telegram_bot.py \
       --model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --tg-token-name token_xl.txt \
       --make-vocab-size-divisible-by 8 \
       --load /home/jovyan/models/gpt3/deepspeed/xl/big2_alternating_b8_seq2048_256gpu_finetune_2/340000/mp_rank_00_model_states.pt \
       --no-load-optim \
       --max-position-embeddings 2048 \
       --tokenizer-path /home/jovyan/datasets/unpacked/big2/_tokenizer/ \
       --fp16 \
       --cache-dir cache_xl \
       --out-seq-length 512 \
       --seq-length 512 \
       --temperature 0.9 \
       --top_k 0 \
       --top_p 0.95 \
       --sparse-mode alternating \
       --deepspeed \
       --deepspeed_config ${config_json}
