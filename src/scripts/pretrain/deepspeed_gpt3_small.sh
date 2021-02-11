#! /bin/bash

# Model parallel size
MP_SIZE=1
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
       --train-data-path /home/jovyan/datasets/unpacked/gpt3_zmitrovich/train \
       --test-data-path /home/jovyan/datasets/unpacked/gpt3_zmitrovich/test \
       --max-files-per-process 100 \
       --logging-dir=/home/jovyan/runs/gpt3/deepspeed/small/test/s1024_${now}_${host} \
       --save /home/jovyan/models/gpt3/deepspeed/small/test/s1024_${now}_${host} \
       --save-interval 100 \
       --log-interval 100 \
       --eval-interval 100 \
       --eval-iters 100 \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --batch-size 8 \
       --seq-length 1024 \
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
       --deepspeed_config ${script_dir}/deepspeed_config/gpt3_small.json \
"

run_cmd="mpirun --np ${NUM_GPUS_PER_WORKER} python ${script_dir}/../../deepspeed_megatron/pretrain_gpt3.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
