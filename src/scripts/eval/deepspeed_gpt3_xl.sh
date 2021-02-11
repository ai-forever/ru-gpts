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

config_json="$script_dir/deepspeed_config/gpt3_xl_sparse_2048.json"
gpt_options=" \
       --test-data-path /home/jovyan/datasets/unpacked/gpt3_zmitrovich/test/test.list \
       --eval-iters 100 \
       --tokenizer-path /home/jovyan/datasets/unpacked/big2/_tokenizer \
       --max-files-per-process 100 \
       --load /home/jovyan/models/gpt3/deepspeed/xl/big2_alternating_b8_seq2048_256gpu_finetune_2 \
       --make-vocab-size-divisible-by 8 \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --batch-size 4 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --distributed-backend nccl \
       --log-interval 10 \
       --fp16 \
       --sparse-mode alternating \
       --deepspeed \
       --deepspeed_config ${config_json} \
"

run_cmd="mpirun --np ${NUM_GPUS_PER_WORKER} python ${script_dir}/../../deepspeed_megatron/pretrain_gpt3.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
