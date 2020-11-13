# ruGPT3Large, ruGPT3Medium, ruGPT3Small and ruGPT2Large
Russian GPT trained with 2048 context length (ruGPT3Large), Russian GPT Medium trained with context 2048 (ruGPT3Medium), Russian GPT Small trained with context 2048 (ruGPT3Small) and Russian GPT2 large (ruGPT2Large) trained with 1024 context length.

We suggest you use ruGPT2Large because this model is more stable and tested.

Examples [here](examples/)

**Note: If you cannot download the checkpoint, try adding it to your google drive following this [issue](https://www.geekrar.com/fix-bypass-google-drive-download-limit-error/)**

Table of contents
* [Setup ruGPT3Large](#Setup-ruGPT3Large)
* [Setup ruGPT3Medium](#Setup-ruGPT3Medium)
* [Setup ruGPT3Small](#Setup-ruGPT3Small)
* [Setup ruGPT2Large](#Setup-ruGPT2Large)
* [Details of pretraining ruGPT3Large](#Details-of-pretraining-ruGPT3Large)
* [Details of pretraining ruGPT3Medium](#Details-of-pretraining-ruGPT3Medium)
* [Details of pretraining ruGPT3Small](#Details-of-pretraining-ruGPT3Small)
* [Details of pretraining ruGPT2Large](#Details-of-pretraining-ruGPT2Large)
* [Usage ruGPT3Large](#Usage-ruGPT3Large)
* [Usage ruGPT3Medium](#Usage-ruGPT3Medium)
* [Usage ruGPT3Small](#Usage-ruGPT3Small)
* [Usage ruGPT2Large](#Usage-ruGPT2Large)

# Christophari GPUs

The organizers gave participants the opportunity to get access to Cristofari by SberCloud. To get access, please send to AIJ_ruGPT-3@sberbank.ru your request with brief information about your project. We will review your request and get back to you. Please note that the number of such accesses is limited - 100 accounts available. If necessary, please leave your request as early as possible.

# Setup
## Setup ruGPT3Large
Code reused from microsoft [implementation](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM) of Megatron-LM.
Supports only python3.6.

To use this repo please install the latest supported versions of PyTorch with GPU support. 

Additionally, part of this codebase leverages tensorflow-cpu to (optionally) perform dataloading of TFRecords for GPT training. We recommend creating a virtual environment (to avoid breaking existing tf installations) and install our `requirements.txt`. 

```bash
python -m pip install virtualenv
virtualenv gpt_env
source gpt_env/bin/activate
pip install -r requirements.txt
```

For using of sparse operations in attention additionally install [torch-blocksparse](https://github.com/ptillet/torch-blocksparse):

```bash
source gpt_env/bin/activate
pip install torch-blocksparse
```

Torch-Blocksparse depends on CUDA 10.1 and the [Triton](https://github.com/ptillet/triton) language and compiler, which requires llvm-9.

## Setup ruGPT3Medium
For this model you can use code from microsoft [implementation](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM) of Megatron-LM in our repo or use transformers interface. Therefore, you should follow the instructions for ruGPT2Large or ruGPT3Large for installation.

## Setup ruGPT3Small
For this model you can use code from microsoft [implementation](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM) of Megatron-LM in our repo or use transformers interface. Therefore, you should follow the instructions for ruGPT2Large or ruGPT3Large for installation.

## Setup ruGPT2Large
This model is smaller and was trained with [transformers==v2.8.0](https://github.com/huggingface/transformers/tree/v2.8.0).
For installing use command:
```bash
pip install transformers
```

# Details of pretraining
All GPUs are  Tesla V100-SXM3 32 Gb.

## Details of pretraining ruGPT3Large
Model was trained on 1024 context length with transformers by [SberDevices](https://sberdevices.ru/) team on 80B tokens around 3 epochs. After that we finetune this on 2048 context. For load transformers checkpoint use `--load-openai`.

The training process took around two weeks on 8 DGX2 (128 GPUs) for 1024 context and few days on 16 GPUs for 2048 context on [Christophari](https://sbercloud.ru/ru/christofari).

Perplexity is 16 on test set.

You can obtain this model here [GDrive](https://drive.google.com/file/d/1t4xw-nvNLQ8kt9FrWW4bPEgCr45M98vu/view?usp=sharing) [Yandex.Disk](https://yadi.sk/d/X7v84O9jrQ8jJg) [GDrive option-2](https://drive.google.com/file/d/1wtc2iBNTcYrqwOzRyEWYWoVBc9xfsbPP/view?usp=sharing) or use in transformers with model name `sberbank-ai/rugpt3large_based_on_gpt2` (see [usage](#Usage-ruGPT3Large) for details).

🤗HuggingFace model card [link](https://huggingface.co/sberbank-ai/rugpt3large_based_on_gpt2)

## Details of pretraining ruGPT3Medium
Model was trained on 1024 context length with transformers by [SberDevices](https://sberdevices.ru/) team on 80B tokens around 3 epoch. After that model was finetuned on 2048 context.

Total training time took around 16 days on 64 GPUs.

You can obtain this model here [GDrive](https://drive.google.com/file/d/1Lb9ILKw0N2ZSEG80QyaCvkp1b2RAw1pC/view?usp=sharing) [Yandex.Disk](https://yadi.sk/d/yE0cw0QIikCPAg) [GDrive option-2](https://drive.google.com/file/d/1gADn4VxDBVrxZ9Wv4bISbDjwCm_3mrDH/view?usp=sharing) or use in transformers with model name `sberbank-ai/rugpt3medium_based_on_gpt2` (see [usage](#Usage-ruGPT3Medium) for details). 

🤗HuggingFace model card [link](https://huggingface.co/sberbank-ai/rugpt3medium_based_on_gpt2)

## Details of pretraining ruGPT3Small
Model was trained on 1024 context length with transformers by [SberDevices](https://sberdevices.ru/) team on 80B tokens around 3 epoch. After that model was finetuned on 2048 context.

Total training time took around one week on 32 GPUs.

You can obtain this model here [GDrive](https://drive.google.com/file/d/19dyhhayJSVJpVPwPzqLRIdCtOddvkzJ4/view?usp=sharing) or use in transformers with model name `sberbank-ai/rugpt3small_based_on_gpt2` (see [usage](#Usage-ruGPT3Small) for details). 

🤗HuggingFace model card [link](https://huggingface.co/sberbank-ai/rugpt3small_based_on_gpt2)

## Details of pretraining ruGPT2Large
Model was trained on 1024 context length with transformers by [SberDevices](https://sberdevices.ru/) team on 170Gb data on 64 GPUs 3 weeks.

You can obtain this model here [GDrive](https://drive.google.com/file/d/1r65MwU0arie8NggxpSmc_3Ja5ldRNS70/view?usp=sharing) [Yandex.Disk](https://yadi.sk/d/BRbn4fl9wqKy0w) [GDrive option-2](https://drive.google.com/file/d/17YuV-uuhSVvMD1cnTe7cR-qscb3BtTiG/view?usp=sharing) or use in transformers with model name `sberbank-ai/rugpt2large` (see [usage](#Usage-ruGPT2Large) for details).

🤗HuggingFace model card [link](https://huggingface.co/sberbank-ai/rugpt2large)

# Usage
## Usage ruGPT3Large
We've provided 2 scripts that pretrain and generate with ruGPT3Large. Save and load model checkpoints with `--save` and `--load`.

### Finetuning
#### Data preparation
We support three file formats for training, but all require preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:

```json
{"src": "KISH", "text": "Как же джокер ты хитер", "type": "Ru", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "Ты удачи приговор", "type": "Ru", "id": "42", "title": "Second Part"}
```

The name of the text field of the json can be changed by using the `--text-key` flag. The other metadata are optional and are not used in training.
#### Running script
`bash ./scripts/pretrain_ruGPT3Large.sh`

This script runs single gpu ruGPT3Large pretraining. This script contains command for running on [Christophari](https://sbercloud.ru/ru/christofari):

```bash
MP_SIZE=1
NUM_GPUS_PER_WORKER=1

mpirun --np ${NUM_GPUS_PER_WORKER} python pretrain_megatron.py \
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
       --text-key \
       --tokenizer-path /home/jovyan/ruGPT3Large \
       --tokenizer-type GPT2BPETokenizer \
       --finetune \
```

Or you can use use transformers interface:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")

model = AutoModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
```

### Text Generation
`bash ./scripts/generate_ruGPT3Large.sh`

Starts an interactive terminal session that generates text either conditionally or unconditionally depending on what the user enters into the prompt. 

The script is capable of top-k, or top-p sampling as specified by the appropriate variables within the script.

Example of generation:

```text
Context: на словах ты лев толстой
ruGPT3Large: а в сущности, - ты тоже не дурак, просто так же, как и твой человек, то есть твоя "жизнь", а также как и ты думаешь по-настоящему "ты" и есть твои "жизнь" или "выбор" в отношении твоего положения.

Context: как же джокер ты хитер
ruGPT3Large: или автор книги по бизнесу!
```

Example of generation in colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/ruGPT3_generation_example.ipynb)

## Usage ruGPT3Medium
You can run megatron script with option `--load-openai` or use transformers interface:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")

model = AutoModel.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
```

### Text Generation
`bash ./scripts/generate_ruGPT3Medium.sh`

Starts an interactive terminal session that generates text either conditionally or unconditionally depending on what the user enters into the prompt. 

The script is capable of top-k, or top-p sampling as specified by the appropriate variables within the script.

Example of generation:

```text
Context >>> На словах ты Лев Толстой, а на деле
ruGPT: На словах ты Лев Толстой, а на деле я — Лев Давидович Троцкий, — сказал я. — Так что мы еще посмотрим

Context: как же джокер ты хитер
ruGPT: как же джокер ты хитер, в этой игре
 - Я не злодей, просто хотел узнать, можно ли узнать о чём?
```

## Usage ruGPT3Small
You can run megatron script with option `--load-openai` or use transformers interface:

```python
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

model = AutoModelWithLMHead.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
```

### Text Generation
`bash ./scripts/generate_ruGPT3Small.sh`

Starts an interactive terminal session that generates text either conditionally or unconditionally depending on what the user enters into the prompt. 

The script is capable of top-k, or top-p sampling as specified by the appropriate variables within the script.

Example of generation:

```text
Context >>> На словах ты Лев Толстой, а на деле
ruGPT: На словах ты Лев Толстой, а на деле – Толстой, – с улыбкой заметил Николай, – я вижу, что ты прав.

– А вот это – другое дело, – сказал Лев Толстой, – это дело другое.

– Да, да, – согласился Николай, – я прав.

– А вот что, Лев Николаевич, – сказал Лев Толстой, – я думаю, что в этом отношении у меня нет оснований сомневаться в твоей правоте.
```

Example of finetune on essays and generation in colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/Finetune_ruGPT3Small.ipynb)

## Usage ruGPT2Large
We've provided 2 scripts that pretrain and generate with ruGPT2Large from [transformers](https://github.com/huggingface/transformers/tree/v2.8.0) original code.

### Finetuning
#### Data preparation
We can pass to model raw text files.
#### Running script
`bash ./scripts/pretrain_ruGPT2Large.sh`

This script runs single gpu ruGPT3Large pretraining. This script contains command for running on [Christophari](https://sbercloud.ru/ru/christofari):

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

Or use transformers interface:

```
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt2large")

model = AutoModel.from_pretrained("sberbank-ai/rugpt2large")
```

### Text Generation
`bash ./scripts/generate_ruGPT2Large.sh`

Starts an interactive terminal session that generates text either conditionally or unconditionally depending on what the user enters into the prompt. 

The script is capable of top-k, or top-p sampling as specified by the appropriate variables within the script.

Example of generation:

```
Context: На словах ты Лев Толстой, а на деле
ruGPT: На словах ты Лев Толстой, а на деле – козел!» – так я про себя подумал, но решил не отвечать. Я встал, поклонился
```
