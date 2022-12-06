%%bash

# Model parallel size
MP_SIZE=1
# Change for multinode config
NUM_GPUS_PER_WORKER=1

gpt_options=" \
       --train-data-path examples/train.list \
       --test-data-path examples/valid.list \
       --load-huggingface sberbank-ai/rugpt3xl \
       --logging-dir=examples/log/ \
       --save examples/model \
       --save-interval 200 \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --batch-size 1 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 1000 \
       --distributed-backend nccl \
       --lr 0.0002 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --warmup .01 \
       --log-interval 50 \
       --fp16 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --sparse-mode alternating \
       --deepspeed \
       --deepspeed_config src/deepspeed_config/gpt3_xl_sparse_2048.json \
"

run_cmd="USE_DEEPSPEED=1 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS_PER_WORKER} pretrain_gpt3.py $@ ${gpt_options}"
echo "${run_cmd}"
eval "${run_cmd}"

set +x
