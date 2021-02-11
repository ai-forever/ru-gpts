#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

python ${script_dir}/../../deepspeed_megatron/generate_samples.py \
       --model-parallel-size 1 \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --make-vocab-size-divisible-by 8 \
       --max-position-embeddings 2048 \
       --load /home/jovyan/models/gpt3/deepspeed/small/finetune_dense_seq2048_b256_exp2/1070000/mp_rank_00_model_states.pt \
       --no-load-optim \
       --tokenizer-path /home/jovyan/datasets/unpacked/gpt3_zmitrovich/_tokenizer/ \
       --fp16 \
       --cache-dir cache \
       --out-seq-length 256 \
       --temperature 0.9 \
       --top_k 0 \
       --top_p 0.95
