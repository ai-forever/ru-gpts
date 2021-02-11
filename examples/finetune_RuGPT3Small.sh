#! /bin/bash

# Model parallel size
MP_SIZE=1
# Change for multinode config
NUM_GPUS_PER_WORKER=1


#batch per gpu 4, grad acc 4, whole batch 256 samples == 512k tokens
#1 epoch 160000 steps, train 3 epochs
config_json=""
gpt_options=" \
       --train-data-path /home/jovyan/data/gpt3test/essays/train.list \
       --test-data-path /home/jovyan/data/gpt3test/essays/valid.list \
       --max-files-per-process 100 \
       --logging-dir=/home/jovyan/models/essays/log2 \
       --save /home/jovyan/models/essays/model2 \
       --load-huggingface sberbank-ai/rugpt3small_based_on_gpt2
       --save-interval 1000 \
       --no-load-optim \
       --finetune \
       --log-interval 100 \
       --model-parallel-size 1 \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --batch-size 1 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 5000 \
       --distributed-backend nccl \
       --lr 0.000015 \
       --warmup 0.0 \
       --lr-decay-style constant \
       --weight-decay 1e-2 \
       --fp16
"

run_cmd="CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node $NUM_GPUS_PER_WORKER /home/jovyan/devices/ru-gpts/pretrain_gpt3.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
