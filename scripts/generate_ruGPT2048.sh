#! /bin/bash

# Model parallel size
MP_SIZE=1

NUM_GPUS_PER_WORKER=1

gpt_options=" \
       --load /home/jovyan/ruGPT3Large \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 1536 \
       --num-attention-heads 16 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --vocab-size 50257 \
       --batch-size 1 \
       --distributed-backend nccl \
       --fp16 \
       --checkpoint-activations \
       --tokenizer-path /home/jovyan/ruGPT3Large \
       --tokenizer-type GPT2BPETokenizer \
       --finetune \
       --out-seq-length 200 \
       --top_p 0.9\
       --top_k 10\
"

run_cmd="mpirun --np ${NUM_GPUS_PER_WORKER} python generate_samples.py $@ ${gpt_options}"
echo "${run_cmd}"
eval "${run_cmd}"

set +x
