#! /bin/bash

# Model parallel size
MP_SIZE=1
# Change for multinode config
NUM_GPUS_PER_WORKER=1

#batch per gpu 4, grad acc 4, whole batch 256 samples == 512k tokens
#1 epoch 160000 steps, train 3 epochs
gpt_options=" \
       --train-data-path /path/to/train.list \
       --test-data-path /path/to/test.list \
       --logging-dir=log/ \
       --save model \
       --save-interval 1000 \
       --model-parallel-size ${MP_SIZE} \
       --save-interval 100 \
       --log-interval 100 \
       --eval-interval 100 \
       --eval-iters 100 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 1 \
       --seq-length 1024 \
       --max-position-embeddings 2048 \
       --train-iters 500000 \
       --resume-dataloader \
       --distributed-backend nccl \
       --lr 0.0002 \
       --lr-decay-style cosine \
       --lr-decay-iters 200000 \
       --min-lr 0.000005 \
       --warmup .001 \
       --fp16 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --deepspeed \
       --deepspeed_config src/deepspeed_config/gpt3_medium_2048.json \
"

run_cmd="USE_DEEPSPEED=1 mpirun --np ${NUM_GPUS_PER_WORKER} python ../pretrain_gpt3.py $@ ${gpt_options}"
echo "${run_cmd}"
eval "${run_cmd}"

set +x
