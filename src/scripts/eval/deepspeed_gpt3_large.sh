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

config_json="$script_dir/deepspeed_config/gpt3_large.json"
gpt_options=" \
       --test-data-path /home/jovyan/datasets/unpacked/big2/test.list \
       --max-files-per-process 10 \
       --load-huggingface /home/jovyan/models/gpt3/huggingface/large/gpt3_large_hf \
       --tokenizer-path /home/jovyan/datasets/unpacked/gpt3_zmitrovich/_tokenizer \
       --cache-prefix _big1_tokenizer_ \
       --make-vocab-size-divisible-by 1 \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 1536 \
       --num-attention-heads 16 \
       --batch-size 4 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
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
