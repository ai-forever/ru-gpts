#! /bin/bash


python generate_transformers.py \
    --model_type=gpt2 \
    --model_name_or_path=sberbank-ai/rugpt3small_based_on_gpt2 \
    --k=10 \
    --p=0.9 \
