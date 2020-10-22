#! /bin/bash

# Model parallel size
MP_SIZE=1

NUM_GPUS_PER_WORKER=1

now=$(date +"%Y_%m_%d_%H_%I_%S")
host=$(hostname)

gpt_options=" \
       --train-data /home/jovyan/data/train.jsonl \
       --valid-data /home/jovyan/data/valid.jsonl \
       --test-data /home/jovyan/data/valid.jsonl \
       --save /home/jovyan/ruGPT3Large/checkpoints_${now}_${host} \
       --load /home/jovyan/ruGPT3Large \
       --tensorboard-dir /home/jovyan/ruGPT3Large/runs_${now}_${host} \
       --save-interval 500 \
       --eval-interval 500 \
       --log-interval 100 \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 1536 \
       --num-attention-heads 16 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --vocab-size 50257 \
       --batch-size 1 \
       --train-iters 200000 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       --lazy-loader \
       --checkpoint-activations \
       --loose-json \
       --text-key text \
       --tokenizer-path /home/jovyan/ruGPT3Large \
       --tokenizer-type GPT2BPETokenizer \
       --finetune \
"

run_cmd="mpirun --np ${NUM_GPUS_PER_WORKER} python pretrain_gpt2.py $@ ${gpt_options}"
echo "${run_cmd}"
eval "${run_cmd}"

set +x
