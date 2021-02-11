#! /bin/bash

# Model parallel size
MP_SIZE=1
# Change for multinode config
NUM_GPUS_PER_WORKER=10

gpt_options=" \
       --train-data-path /home/jovyan/devices/mog/ruwiki.list \
       --max-files-per-process 20000 \
       --logging-dir=/home/jovyan/devices/mog/runs/ \
       --load /home/jovyan/devices/mog/base \
       --save /home/jovyan/devices/mog/base \
       --tokenizer-path /home/jovyan/devices/dgpt_transformers/gpt3_serving/xl_serving/gpt3_xl \
       --save-interval 1000 \
       --model-parallel-size ${MP_SIZE} \
       --batch-size 4 \
       --seq-length 512 \
       --mog-iterations 6 \
       --num-layers 4 \
       --hidden-dropout 0.2 \
       --max-position-embeddings 512 \
       --train-iters 2000000 \
       --resume-dataloader \
       --distributed-backend nccl \
       --lr 0.005 \
       --lr-decay-style cosine \
       --weight-decay 1e-3 \
       --clip-grad 10.0 \
       --warmup .01 \
       --log-interval 100 \
"
run_cmd="mpirun --np ${NUM_GPUS_PER_WORKER} python /home/jovyan/devices/mog/dgpt_transformers/deepspeed_megatron/pretrain_mog.py $@ ${gpt_options}"
echo "${run_cmd}"
eval "${run_cmd}"

set +x
