#! /bin/bash

# Change for multinode config
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
now=$(date +"%Y_%m_%d_%H_%I_%S")
host=$(hostname)


#batch per gpu 4, grad acc 4, whole batch 256 samples == 512k tokens
#1 epoch 160000 steps, train 3 epochs
config_json=""
gpt_options=" \
       --train-data-path /home/jovyan/datasets/unpacked/big2/train.list \
       --max-files-per-process 10 \
       --logging-dir=/home/jovyan/runs/gpt3/deepspeed/small/test/big2_${now}_${host} \
       --save /home/jovyan/models/gpt3/deepspeed/small/test/big2__${now}_${host} \
       --save-interval 100 \
       --log-interval 10 \
       --model-parallel-size 1 \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --batch-size 16 \
       --seq-length 512 \
       --max-position-embeddings 2048 \
       --train-iters 500000 \
       --resume-dataloader \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --lr-decay-iters 320000 \
       --clip-grad 0.5 \
       --warmup .004 \
       --fp16 \
       --deepspeed \
       --deepspeed_config ${script_dir}/deepspeed_config/gpt3_small_sparse.json \
"

run_cmd="mpirun --np ${NUM_GPUS_PER_WORKER} python ${script_dir}/../../deepspeed_megatron/pretrain_gpt3.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
