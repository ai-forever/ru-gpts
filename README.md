# ruGPT3XL, ruGPT3Large, ruGPT3Medium, ruGPT3Small and ruGPT2Large
Russian GPT trained with 2048 context length (ruGPT3XL) with sparse attention, Russian GPT trained with 2048 context length (ruGPT3Large), Russian GPT Medium trained with context 2048 (ruGPT3Medium), Russian GPT Small trained with context 2048 (ruGPT3Small) and Russian GPT2 large (ruGPT2Large) trained with 1024 context length.

We suggest you use ruGPT2Large or ruGPT3XL because this model is more stable and tested.

Examples [here](examples/)

Table of contents

# Christofari GPUs

The organizers gave participants the opportunity to get access to Cristofari by SberCloud.

# Setup and usage
Models can be used for inference or finetuning with two ways: 🤗HuggingFace interface or our code based on this [implementation](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM).

For both ways install transformers:

```bash
pip install transformers==3.5.0
```

## HuggingFace interface
We support 🤗HuggingFace interface only for ruGPT3Large, ruGPT3Medium, ruGPT3Small and ruGPT2Large models. For RuGPT3XL please use code in this repo because RuGPT3XL model was trained with sparse attention.

Here we can obtain examples of [finetuning](examples/Finetune_RuGPTs_with_HF.ipynb) or [generation](examples/Generate_text_with_RuGPTs_HF.ipynb).

Also this examples is adapted for google colab:
* [![finetuning](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/Finetune_RuGPTs_with_HF.ipynb)
* [![generation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/Generate_text_with_RuGPTs_HF.ipynb)

Basic usage:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer


model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()
text = "Александр Сергеевич Пушкин родился в "
input_ids = tokenizer.encode(text, return_tensors="pt").cuda()
out = model.generate(input_ids.cuda())
generated_text = list(map(tokenizer.decode, out))[0]
print(generated_text)
# Output should be like this:
# Александр Сергеевич Пушкин родился в \n1799 году. Его отец был крепостным крестьянином, а мать – крепостной крестьянкой. Детство и юность Пушкина прошли в деревне Михайловское под Петербургом. В 1820-х годах семья переехала
```

For more information about 🤗HuggingFace interface please follow this [documentation](https://huggingface.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate).

#### Data issues
For training pass single txt file.

## Megatron interface
### Without deepspeed
For using our code for finetuning without deepspeed (not recommended) we should install apex:

```bash
%%writefile setup.sh

export CUDA_HOME=/usr/local/cuda-10.1
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

sh setup.sh
```

Example of finetuning, generating and loading/convert megatron checkpoints [here](examples/Finetune_and_generate_RuGPTs_only_with_megatron.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/Finetune_and_generate_RuGPTs_only_with_megatron.ipynb)

**Note!** This way is valid for all RuGPTs models except RuGPT3XL.

### Megatron with deepspeed
For using our code for finetuning with deepspeed (recommended) we should install apex (see previous section) and deepspeed:

```bash
pip install deepspeed==0.3.7
```

Example of finetuning, generating and loading/convert megatron checkpoints [here](examples/Finetune_and_generate_RuGPTs_deepspeed_megatron.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/Finetune_and_generate_RuGPTs_deepspeed_megatron.ipynb)

**Note!** For using deepspeed we should specify environ variable before all your python scripts and run with torch.distributed or mpi:

```
USE_DEEPSPEED=1 python -m torch.distributed.launch --nproc_per_node 1 ru-gpts/pretrain_gpt3.py \
  --train-data-path "train.list" \
  --test-data-path "valid.list" \
  --max-files-per-process 100 \
  --save model \
  --load-huggingface sberbank-ai/rugpt3small_based_on_gpt2 \
  --model-parallel-size 1 \
  --num-layers 12 \
  --hidden-size 768 \
  --num-attention-heads 12 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --fp16 \
  --checkpoint-activations \
  --deepspeed-activation-checkpointing \
  --deepspeed \
  --deepspeed_config ru-gpts/src/deepspeed_config/gpt3_small_2048.json
```

#### Data issues
We use custom implementation of distributed dataset. For training and evaluating we should specify file `file.list` with list of paths to txt files. All files from `file.list` will be splitted between aviable GPUs. The logic of splitting is described by the following code:

```python
shard_size = len(files) // world_size
shard_start = rank * shard_size
shard_end = (rank + 1) * shard_size
files = files[shard_start:shard_end]
```

For more details please see full code of dataset: `src.dataset_rugpt3.RuGpt3TextDataset` and example.

**Note!** This way is valid for all RuGPTs models except RuGPT3XL.






## Setup ruGPT3XL
See all details [here](gw/)

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

## Details of pretraining ruGPT3XL
Model was trained on 512 context length with [deepspeed](https://github.com/microsoft/DeepSpeed) and [megatron](https://github.com/NVIDIA/Megatron-LM) code by [SberDevices](https://sberdevices.ru/) team. After that model was finetuned on 2048 context. Note! Model has sparse attention modules.

Total training time took around 10 days on 256 GPUs. Final perplexity on test set is `11.4`.

🤗HuggingFace model card [link](https://huggingface.co/sberbank-ai/rugpt3xl).

See more details [here](gw/)

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
## Usage ruGPT3XL
See all details [here](gw/) or run example on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/ruGPT3XL_generation.ipynb)

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
