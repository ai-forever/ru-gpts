#! /bin/bash

python generate_transformers.py \
    --model_type=gpt2 \
    --model_name_or_path=sberbank-ai/rugpt3medium_based_on_gpt2 \
    --k=50 \
    --p=0.9 \
