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

config_json="$script_dir/deepspeed_config/gpt3_small.json"
gpt_options=" \
       --test-data-path /home/jovyan/datasets/unpacked/gpt3_zmitrovich/test \
       --load-huggingface /home/jovyan/models/gpt3/huggingface/gpt3_large_bbpe_v50 \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 36 \
       --hidden-size 1280 \
       --num-attention-heads 20 \
       --batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --distributed-backend nccl \
       --log-interval 10 \
       --fp16 \
       --deepspeed \
       --deepspeed_config ${config_json} \
"

run_cmd="mpirun --np ${NUM_GPUS_PER_WORKER} python ${script_dir}/../../deepspeed_megatron/pretrain_gpt3.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
