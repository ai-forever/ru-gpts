# ruGPT2048, ruGPT3Large and ruGPT3Medium2048
Russian GPT trained with 2048 context length (ruGPT2048), Russian GPT3 large (ruGPT3Large) trained with 1024 context length and Russian GPT Medium trained with context 2048 (ruGPT3Medium2048).

We suggest you use ruGPT2Large because this model is more stable and tested.

Examples [here](examples/)

**Note: If you cannot download the checkpoint, try adding it to your google drive following this [issue](https://www.geekrar.com/fix-bypass-google-drive-download-limit-error/)**

Table of contents
* [Setup ruGPT2048](#Setup-ruGPT2048)
* [Setup ruGPT3Large](#Setup-ruGPT3Large)
* [Setup ruGPTMedium2048](#Setup-ruGPTMedium2048)
* [Details of pretraining ruGPT2048](#Details-of-pretraining-ruGPT2048)
* [Details of pretraining ruGPT3Large](#Details-of-pretraining-ruGPT3Large)
* [Details of pretraining ruGPT3Medium2048](#Details-of-pretraining-ruGPT3Medium2048)
* [Usage ruGPT2048](#Usage-ruGPT2048)
* [Usage ruGPT3Large](#Usage-ruGPT3Large)
* [Usage ruGPT3Medium2048](#Usage-ruGPT3Medium2048)


# Setup
## Setup ruGPT2048
Code reused from microsoft [implementation](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM) of Megatron-LM.
Supports only python3.6.

To use this repo please install the latest supported versions of PyTorch with GPU support. 

Additionally, part of this codebase leverages tensorflow-cpu to (optionally) perform dataloading of TFRecords for GPT training. We recommend creating a virtual environment (to avoid breaking existing tf installations) and install our `requirements.txt`. 

```
python -m pip install virtualenv
virtualenv gpt_env
source gpt_env/bin/activate
pip install -r requirements.txt
```

For using of sparse operations in attention additionally install [torch-blocksparse](https://github.com/ptillet/torch-blocksparse):

```
source gpt_env/bin/activate
pip install torch-blocksparse
```

Torch-Blocksparse depends on CUDA 10.1 and the [Triton](https://github.com/ptillet/triton) language and compiler, which requires llvm-9.

## Setup ruGPT3Large
This model is smaller and was trained with [transformers==v2.8.0](https://github.com/huggingface/transformers/tree/v2.8.0).
For installing use command:
```
pip install transformers
```

## Setup ruGPT3Medium2048
For this model you can use code from microsoft [implementation](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM) of Megatron-LM in our repo or use transformers interface. Therefore, you should follow the instructions for ruGPT3Large or ruGPT2048 for installation.

# Details of pretraining
All GPUs are  Tesla V100-SXM3 32 Gb.
## Details of pretraining ruGPT2048
Model was trained on 1024 context length with transformers by [SberDevices](https://sberdevices.ru/) team on 80B tokens around 3 epochs. After that we finetune this on 2048 context. For load transformers checkpoint use `--load-openai`.

The training process took around two weeks on 8 DGX2 (128 GPUs) for 1024 context and 1 day (still training) on 10 GPUs for 2048 context on [Christophari](https://sbercloud.ru/ru/christofari).

Perplexity is 16 on test set.

You can obtain this model here [GDrive](https://drive.google.com/file/d/12JkbnzSoQwJqanVP-zoLNnFX3e4HHyvY/view?usp=sharing) [Yandex.Disk](https://yadi.sk/d/kchlR0MWF8MqvQ). 

## Details of pretraining ruGPT3Large
Model was trained on 1024 context length with transformers by [SberDevices](https://sberdevices.ru/) team on 80B tokens around 3 epoch. After that model was finetuned on 2048 context.

Total training time took around 16 days on 64 GPUs.

You can obtain this model here [GDrive](https://drive.google.com/file/d/1r65MwU0arie8NggxpSmc_3Ja5ldRNS70/view?usp=sharing) [Yandex.Disk](https://yadi.sk/d/B-zj3eojA3KmUQ)

## Details of pretraining ruGPT3Medium2048
Model was trained on 1024 context length with transformers by [SberDevices](https://sberdevices.ru/) team on 170Gb data on 64 GPUs 3 weeks.

You can obtain this model here [GDrive](https://drive.google.com/file/d/1GsTOqAOPKFfL8fu5Beag6_u8NcdtI3AA/view?usp=sharing) [Yandex.Disk](https://yadi.sk/d/1TGEhhkwtpUl4Q)

# Usage
## Usage ruGPT2048
We've provided 2 scripts that pretrain and generate with ruGPT2048. Save and load model checkpoints with `--save` and `--load`.

### Finetuning
#### Data preparation
We support three file formats for training, but all require preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:

```
{"src": "KISH", "text": "Как же джокер ты хитер", "type": "Ru", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "Ты удачи приговор", "type": "Ru", "id": "42", "title": "Second Part"}
```

The name of the text field of the json can be changed by using the `--text-key` flag. The other metadata are optional and are not used in training.
#### Running script
`bash ./scripts/pretrain_ruGPT2048.sh.sh`

This script runs single gpu ruGPT2048 pretraining. This script contains command for running on [Christophari](https://sbercloud.ru/ru/christofari):

```
MP_SIZE=1
NUM_GPUS_PER_WORKER=1

mpirun --np ${NUM_GPUS_PER_WORKER} python pretrain_gpt2.py \
       --train-data /home/jovyan/data/train.jsonl \
       --valid-data /home/jovyan/data/valid.jsonl \
       --test-data /home/jovyan/data/valid.jsonl \
       --save /home/jovyan/rugpt2048/checkpoints_${now}_${host} \
       --load /home/jovyan/rugpt2048 \
       --tensorboard-dir /home/jovyan/rugpt2048/runs_${now}_${host} \
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
       --text-key \
       --tokenizer-path /home/jovyan/rugpt2048 \
       --tokenizer-type GPT2BPETokenizer \
       --finetune \
```

### Text Generation
`bash ./scripts/generate_ruGPT2048.sh`

Starts an interactive terminal session that generates text either conditionally or unconditionally depending on what the user enters into the prompt. 

The script is capable of top-k, or top-p sampling as specified by the appropriate variables within the script.

Example of generation:

```
Context: на словах ты лев толстой
ruGPT2048: а в сущности, - ты тоже не дурак, просто так же, как и твой человек, то есть твоя "жизнь", а также как и ты думаешь по-настоящему "ты" и есть твои "жизнь" или "выбор" в отношении твоего положения.

Context: как же джокер ты хитер
ruGPT2048: или автор книги по бизнесу!
```

## Usage ruGPT3Large
We've provided 2 scripts that pretrain and generate with ruGPT3Large from [transformers](https://github.com/huggingface/transformers/tree/v2.8.0) original code.

### Finetuning
#### Data preparation
We can pass to model raw text files.
#### Running script
`bash ./scripts/pretrain_ruGPT2Large.sh`

This script runs single gpu ruGPT2048 pretraining. This script contains command for running on [Christophari](https://sbercloud.ru/ru/christofari):

```
python pretrain_transformers.py \
    --output_dir=/home/jovyan/rugpt2large/checkpoints_"${now}"_"${host}" \
    --model_type=gpt2 \
    --model_name_or_path=/home/jovyan/gpt2_large_bbpe_v50 \
    --do_train \
    --train_data_file=/home/jovyan/data/train.txt \
    --do_eval \
    --eval_data_file=/home/jovyan/data/valid.txt \
    --fp16
```

### Text Generation
`bash ./scripts/generate_ruGPT2Large.sh`

Starts an interactive terminal session that generates text either conditionally or unconditionally depending on what the user enters into the prompt. 

The script is capable of top-k, or top-p sampling as specified by the appropriate variables within the script.

Example of generation:

```
Context: на словах ты лев толстой
ruGPT2Large: на словах ты лев толстой кожи, а в деле — просто тряпка!
```

## Usage ruGPTMedium2048
Choose one of [ruGPT2048](#Usage-ruGPT2048) or [ruGPT3Large](#Usage-ruGPT3Large) way. This is the same.
