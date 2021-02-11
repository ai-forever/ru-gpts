#! /bin/bash

# Model parallel size
MP_SIZE=1
# Change for multinode config
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=4

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
now=$(date +"%Y_%m_%d_%H_%I_%S")
host=$(hostname)

config_json="$script_dir/deepspeed_config/gpt3_large.json"
gpt_options=" \
       --train-data-path /home/jovyan/datasets/unpacked/gpt3_zmitrovich/train \
       --logging-dir=/home/jovyan/runs/gpt3/deepspeed/large/${now}_${host} \
       --save /home/jovyan/models/gpt3/deepspeed/large/${now}_${host} \
       --save-interval 1000 \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 1536 \
       --num-attention-heads 16 \
       --batch-size 1 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 200000 \
       --resume-dataloader \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --log-interval 100 \
       --fp16 \
       --deepspeed \
       --deepspeed_config ${config_json} \
"
#        --checkpoint-activations \
#        --deepspeed-activation-checkpointing \
run_cmd="mpirun --np ${NUM_GPUS_PER_WORKER} python ${script_dir}/../../deepspeed_megatron/pretrain_gpt3.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
