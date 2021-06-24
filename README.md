#### Russian GPT-3 models
# ruGPT3XL, ruGPT3Large, ruGPT3Medium, ruGPT3Small and ruGPT2Large
This repository contains bunch of autoregressive transformer language models trained on a huge dataset of russian language.

 * Russian GPT-3 models (ruGPT3XL, ruGPT3Large, ruGPT3Medium, ruGPT3Small) trained with 2048 sequence length with sparse and dense attention blocks. We also provide Russian GPT-2 large model (ruGPT2Large) trained with 1024 sequence length.

 * Try Model Generation In Colab! ruGPT-3 XL: [![Try Model Generation In Colab!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/ruGPT3XL_generation.ipynb) or ruGPT-3 smaller models: [![Try Model Generation In Colab!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/Generate_text_with_RuGPTs_HF.ipynb)
 
 * Usage examples are described in detail [here](examples/). See how fine-tuning works: [![Try Model Generation In Colab!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/ruGPT3XL_finetune_example.ipynb)

 * Old version of code you can find [here](https://github.com/sberbank-ai/ru-gpts/tree/old)



## Table of contents
* OpenSource Solutions with ruGPT-3
* Papers mentioning ruGPT-3
* Setup and usage
  * [HuggingFace interface](#HuggingFace-interface)
  * [Megatron interface](#Megatron-interface)
* Pretraining details
  * [Pretraining ruGPT3XL](#Pretraining-ruGPT3XL)
  * [Pretraining ruGPT3Large](#Pretraining-ruGPT3Large)
  * [Pretraining ruGPT3Medium](#Pretraining-ruGPT3Medium)
  * [Pretraining ruGPT3Small](#Pretraining-ruGPT3Small)
  * [Pretraining ruGPT2Large](#Pretraining-ruGPT2Large)
* Advanced
  * [Pretrained scripts](#Pretrained-scripts-(advanced))
  * [Convert checkpoint to HuggingFace](#Convert-checkpoint-to-HuggingFace)

## OpenSource Solutions with ruGPT-3

* ruCLIP [Github](https://github.com/sberbank-ai/ru-clip)
* Simplification with ruGPT-3 XL [Github](https://github.com/Alenush/rugpt3simplification_rsse )
* Word normalization (RuNormAS shared task) [Github](https://github.com/RussianNLP/RuNormAS-solution)
* AI CopyWriter [Github](https://github.com/dilyararimovna/text_expansion)
* ЕГЭ Generation [Github](https://github.com/orzhan/rugpt3-question-generation )
* NeuroZhirinovsky [Github](https://github.com/GraphGrailAi/ruGPT3-ZhirV)
* PseudoKant [Github](https://github.com/AsakoKabe/pseudo-kant )
* DostoevskyDoesntWriteIt [Github](https://github.com/K7chyp/DostoevskyDoesntWriteIt)


## Papers mentioning ruGPT-3
According to google scholar [search](https://scholar.google.com/scholar?hl=ru&as_sdt=0%2C5&q=rugpt3&btnG=) - feel free to add links to this list

### Text Simplification
```
@article{shatilovsentence,
  title={Sentence simplification with ruGPT3},
  author={Shatilov, AA and Rey, AI},
  url={http://www.dialog-21.ru/media/5281/shatilovaaplusreyai142.pdf}
}

@article{fenogenovatext,
  title={Text Simplification with Autoregressive Models},
  author={Fenogenova, Alena and Sberbank, SberDevices},
  url={http://www.dialog-21.ru/media/5250/fenogenovaa141.pdf}}
  ```

### Text Detoxification
```
@article{dementieva2021methods,
  title={Methods for Detoxification of Texts for the Russian Language},
  author={Dementieva, Daryna and Moskovskiy, Daniil and Logacheva, Varvara and Dale, David and Kozlova, Olga and Semenov, Nikita and Panchenko, Alexander},
  journal={arXiv preprint arXiv:2105.09052},
  year={2021},
  url={https://arxiv.org/abs/2105.09052}
}
```

### Paraphrasing and Data Augmentation
```
@inproceedings{fenogenova2021russian,
  title={Russian Paraphrasers: Paraphrase with Transformers},
  author={Fenogenova, Alena},
  booktitle={Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing},
  pages={11--19},
  year={2021},
  url={https://www.aclweb.org/anthology/2021.bsnlp-1.2.pdf}
}
``` 

### Model Evaluation
```
@article{malykh2021morocco,
  title={MOROCCO: Model Resource Comparison Framework},
  author={Malykh, Valentin and Kukushkin, Alexander and Artemova, Ekaterina and Mikhailov, Vladislav and Tikhonova, Maria and Shavrina, Tatiana},
  journal={arXiv preprint arXiv:2104.14314},
  year={2021},
  url={https://arxiv.org/abs/2104.14314}}
  ``` 
 

## Setup and usage
Models can be used for inference or finetuning with two ways: 🤗HuggingFace interface or our code based on this [implementation](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM).

For both ways install transformers:

```bash
pip install transformers==3.5.0
```

### HuggingFace interface
We support 🤗HuggingFace interface only for ruGPT3Large, ruGPT3Medium, ruGPT3Small and ruGPT2Large models. For RuGPT3XL please use code in this repo because RuGPT3XL model was trained with sparse attention.

Here we can obtain examples of [finetuning](examples/Finetune_RuGPTs_with_HF.ipynb) or [generation](examples/Generate_text_with_RuGPTs_HF.ipynb).

Also this examples is adapted for google colab:
* finetuning: [![finetuning](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/Finetune_RuGPTs_with_HF.ipynb)
* generation: [![generation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/Generate_text_with_RuGPTs_HF.ipynb)

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

For more information about 🤗HuggingFace interface please follow this [documentation](https://HuggingFace.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate).

##### Data issues
For training pass single txt file.

### Megatron interface
#### Without deepspeed
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

#### Megatron with deepspeed
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

##### Data issues
We use custom implementation of distributed dataset. For training and evaluating we should specify file `file.list` with list of paths to txt files. All files from `file.list` will be splitted between aviable GPUs. The logic of splitting is described by the following code:

```python
shard_size = len(files) // world_size
shard_start = rank * shard_size
shard_end = (rank + 1) * shard_size
files = files[shard_start:shard_end]
```

For more details please see full code of dataset: `src.dataset_rugpt3.RuGpt3TextDataset` and example.

**Note!** This way is valid for all RuGPTs models except RuGPT3XL.

#### Megatron with deepspeed and sparsity
This section is used mostly for usage of RuGPT3XL model and training models with sparse attention.

```bash
apt-get install llvm-9-dev
pip install cpufeature
pip install triton==0.2.3
DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install deepspeed==0.3.7
```

Test installation of deepspeed you can with the following command: `ds_report`.

Example of inference of RuGPT3XL [here](examples/ruGPT3XL_generation.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/ruGPT3XL_generation.ipynb)

Example of finetune, load finetuned model and generate is [here](examples/ruGPT3XL_finetune_example.ipynb).

For using sparse layers in model use ```--sparse-mode <mode>``` and specify key `"sparse_attention"` at deepspeed_config (RuGPT3XL config [example](src/deepspeed_config/gpt3_xl_sparse_2048.json)). Modes can be: `fixed`, `bigbird`, `bslongformer`, `variable`, `dense`.

More information about sparse attention [here](https://www.deepspeed.ai/tutorials/sparse-attention/).

## Pretraining details
All pretraining was done on Nvidia Tesla V100-SXM3 32 Gb GPUs on a [Christofari Cluster](https://sbercloud.ru/ru/christofari). Following are the details of pretraining for each model.


### Pretraining ruGPT3XL
Model was trained with 512 sequence length using [Deepspeed](https://github.com/microsoft/DeepSpeed) and [Megatron](https://github.com/NVIDIA/Megatron-LM) code by [SberDevices](https://sberdevices.ru/) team, on 80B tokens dataset for 4 epochs. After that model was finetuned 1 epoch with sequence length 2048.  
*Note! Model has sparse attention blocks.*

Total training time was around 10 days on 256 GPUs.  
Final perplexity on test set is `12.05`.

🤗HuggingFace model card [link](https://HuggingFace.co/sberbank-ai/rugpt3xl).

See more details for generation [here](examples/ruGPT3XL_generation.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/ruGPT3XL_generation.ipynb).

Example of finetune, load finetuned model and generate is [here](examples/ruGPT3XL_finetune_example.ipynb).

Our pretraining script [here](scripts/deepspeed_gpt3_xl.sh)

Example of finetuning script [here](scripts/deepspeed_gpt3_xl_finetune.sh)

### Pretraining ruGPT3Large
Model was trained with sequence length 1024 using transformers lib by [SberDevices](https://sberdevices.ru/) team on 80B tokens for 3 epochs. After that model was finetuned 1 epoch with sequence length 2048. 

Total training time was around 14 days on 128 GPUs for 1024 context and few days on 16 GPUs for 2048 context.  
Final perplexity on test set is `13.6`.

You can obtain this model by using transformers with model name `sberbank-ai/rugpt3large_based_on_gpt2`.

🤗HuggingFace model card [link](https://HuggingFace.co/sberbank-ai/rugpt3large_based_on_gpt2)

Our pretraining script [here](scripts/deepspeed_gpt3_large.sh)

### Pretraining ruGPT3Medium
Model was trained with sequence length 1024 using transformers lib by [SberDevices](https://sberdevices.ru/) team on 80B tokens for 3 epoch. After that model was finetuned on 2048 context.

Total training time was around 16 days on 64 GPUs.  
Final perplexity on test set is `17.4`.

You can obtain this model by using transformers with model name `sberbank-ai/rugpt3medium_based_on_gpt2`. 

🤗HuggingFace model card [link](https://HuggingFace.co/sberbank-ai/rugpt3medium_based_on_gpt2)

Our pretraining script [here](scripts/deepspeed_gpt3_medium.sh)

### Pretraining ruGPT3Small
Model was trained with sequence length 1024 using transformers by [SberDevices](https://sberdevices.ru/) team on 80B tokens around 3 epoch. After that model was finetuned on 2048 context.

Total training time took around one week on 32 GPUs.

You can obtain this model by using transformers with model name `sberbank-ai/rugpt3small_based_on_gpt2`. 

🤗HuggingFace model card [link](https://HuggingFace.co/sberbank-ai/rugpt3small_based_on_gpt2)

Our pretraining script [here](scripts/deepspeed_gpt3_small.sh)

### Pretraining ruGPT2Large
Model was trained with sequence length 1024 using transformers by [SberDevices](https://sberdevices.ru/) team on 170Gb data on 64 GPUs 3 weeks.

You can obtain this model by using transformers with model name `sberbank-ai/rugpt2large`.

🤗HuggingFace model card [link](https://HuggingFace.co/sberbank-ai/rugpt2large)

## Advanced
### Pretrained scripts (advanced)
Also we add pretraining scripts for all models (except RuGPT2Large). See [scripts](scripts/) dir.

**Note!** All training params (such as lr, wd, ...) may was different while real training. This is just for example.

### Convert checkpoint to HuggingFace
For converting megatron checkpoint to HuggingFace format use the following script (example for RuGPT3Small):

```bash
python convert2huggingface.py \
  --load /path/to/save/dir/ \
  --model-parallel-size 1 \
  --num-layers 12 \
  --hidden-size 768 \
  --num-attention-heads 12 \
  --max-position-embeddings 2048 \
  --tokenizer-path sberbank-ai/rugpt3small_based_on_gpt2 \
  --no-load-optim \
  --export-huggingface /path/to/converted/checkpoint
```

After converting we can use HuggingFace model:

```python
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("/path/to/converted/checkpoint")
```

**Note!** Conversion is worked for all models except RuGPT3XL. For using of RuGPT3XL see example of inference of RuGPT3XL [here](examples/ruGPT3XL_generation.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/ruGPT3XL_generation.ipynb).
