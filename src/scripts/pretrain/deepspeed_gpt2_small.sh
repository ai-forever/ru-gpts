#! /bin/bash

# Model parallel size
MP_SIZE=1
# Change for multinode config
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=14

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
now=$(date +"%Y_%m_%d_%H_%I_%S")
host=$(hostname)

config_json="$script_dir/deepspeed_config/gpt3_small.json"
gpt_options=" \
       --train-data-path /home/jovyan/datasets/unpacked/gpt3_zmitrovich/train \
       --logging-dir=/home/jovyan/runs/gpt3/deepspeed/small/${now}_${host} \
       --save /home/jovyan/models/gpt3/deepspeed/small/${now}_${host} \
       --save-interval 1000 \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --batch-size 12 \
       --seq-length 1024 \
       --max-position-embeddings 2048 \
       --train-iters 2500000 \
       --resume-dataloader \
       --distributed-backend nccl \
       --lr 0.00005 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .001 \
       --log-interval 200 \
       --fp16 \
       --deepspeed \
       --deepspeed_config ${config_json} \
"

run_cmd="mpirun --np ${NUM_GPUS_PER_WORKER} python ${script_dir}/../../deepspeed_megatron/pretrain_gpt3.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
