#! /bin/bash

# Model parallel size
MP_SIZE=1
# Change for multinode config
NUM_GPUS_PER_WORKER=1

gpt_options=" \
       --train-data-path /path/to/train.list \
       --test-data-path /path/to/test.list \
       --logging-dir=log/ \
       --save model \
       --save-interval 1000 \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --batch-size 1 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 200000 \
       --resume-dataloader \
       --distributed-backend nccl \
       --lr 0.0002 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --warmup .01 \
       --log-interval 100 \
       --fp16 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --sparse-mode alternating \
       --deepspeed \
       --deepspeed_config src/deepspeed_config/gpt3_xl_sparse_2048.json \
"

run_cmd="mpirun --np ${NUM_GPUS_PER_WORKER} python ../pretrain_gpt3.py $@ ${gpt_options}"
echo "${run_cmd}"
eval "${run_cmd}"

set +x
