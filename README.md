#### Russian GPT-3 models
# ruGPT3XL, ruGPT3Large, ruGPT3Medium, ruGPT3Small and ruGPT2Large
This repository contains bunch of autoregressive transformer language models trained on a huge dataset of russian language.

 * Russian GPT-3 models (ruGPT3XL, ruGPT3Large, ruGPT3Medium, ruGPT3Small) trained with 2048 sequence length with sparse and dense attention blocks. We also provide Russian GPT-2 large model (ruGPT2Large) trained with 1024 sequence length.

 * Try Model Generation In Colab! ruGPT-3 XL: [![Try Model Generation In Colab!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/ru-gpts/blob/master/examples/ruGPT3XL_generation.ipynb) or ruGPT-3 smaller models: [![Try Model Generation In Colab!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/ru-gpts/blob/master/examples/Generate_text_with_RuGPTs_HF.ipynb)
 
 * Usage examples are described in detail [here](examples/). See how fine-tuning works: [![Try Model Generation In Colab!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/ru-gpts/blob/master/examples/ruGPT3XL_finetune_example.ipynb)

## Table of contents
* [ruGPT3XL](#ruGPT3XL)
  * Setup
  * Usage
  * Finetune
  * [Pretraining details ruGPT3XL](#Pretraining-details-ruGPT3XL)
* [ruGPT3Large, ruGPT3Medium, ruGPT3Small, ruGPT2Large](ruGPT3Large,-ruGPT3Medium,-ruGPT3Small,-ruGPT2Large)
  * Setup
  * Usage
  * Pretraining details
    * [Pretraining details ruGPT3Large](#Pretraining-details-ruGPT3Large)
    * [Pretraining details ruGPT3Medium](#Pretraining-details-ruGPT3Medium)
    * [Pretraining details ruGPT3Small](#Pretraining-details-ruGPT3Small)
    * [Pretraining details ruGPT2Large](#Pretraining-details-ruGPT2Large)
* [Papers mentioning ruGPT3](#Papers-mentioning-ruGPT3)
* [OpenSource Solutions with ruGPT3](#OpenSource-Solutions-with-ruGPT3)

# ruGPT3XL
## Setup
For colab we recommend use the following installation instructions:

```%%bash
export LD_LIBRARY_PATH=/usr/lib/
apt-get install clang-9 llvm-9 llvm-9-dev llvm-9-tools
git clone https://github.com/qywu/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install triton
DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install deepspeed
pip install transformers
pip install huggingface_hub
pip install timm==0.3.2
git clone  https://github.com/sberbank-ai/ru-gpts
cp ru-gpts/src_utils/trainer_pt_utils.py /usr/local/lib/python3.8/dist-packages/transformers/trainer_pt_utils.py
cp ru-gpts/src_utils/_amp_state.py /usr/local/lib/python3.8/dist-packages/apex/amp/_amp_state.py
```

After installation env please restart colab. For checking is all ok, run the following commands:

```
!ds_report
# Output:
...
sparse_attn ............ [YES] ...... [OKAY]
...
import deepspeed.ops.sparse_attention.sparse_attn_op
```

## Usage
Here is a simple example of usage. For more see this [example](examples/ruGPT3XL_generation.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/ru-gpts/blob/master/examples/ruGPT3XL_generation.ipynb).

```python
import sys
from src.xl_wrapper import RuGPT3XL
import os

# If run to from content root.
sys.path.append("ru-gpts/")
os.environ["USE_DEEPSPEED"] = "1"
# We can change address and port
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "5000"
gpt = RuGPT3XL.from_pretrained("sberbank-ai/rugpt3xl", seq_len=512)
gpt.generate(
    "Кто был президентом США в 2020? ",
    max_length=50,
    no_repeat_ngram_size=3,
    repetition_penalty=2.,
)
```
## Finetuning
Example of finetune, load finetuned model and generate is [here](examples/ruGPT3XL_finetune_example.ipynb).

Our example of finetuning script [here](scripts/deepspeed_gpt3_xl_finetune.sh)

## Pretraining details ruGPT3XL
Model was trained with 512 sequence length using [Deepspeed](https://github.com/microsoft/DeepSpeed) and [Megatron](https://github.com/NVIDIA/Megatron-LM) code by [Devices](https://sberdevices.ru/) team, on 80B tokens dataset for 4 epochs. After that model was finetuned 1 epoch with sequence length 2048.  
*Note! Model has sparse attention blocks.*

Total training time was around 10 days on 256 GPUs.  
Final perplexity on test set is `12.05`.

🤗HuggingFace model card [link](https://HuggingFace.co/ai-forever/rugpt3xl).

# ruGPT3Large, ruGPT3Medium, ruGPT3Small, ruGPT2Large
## Setup
For using ruGPT3Large, ruGPT3Medium, ruGPT3Small, ruGPT2Large just install 🤗HuggingFace transformers.

```bash
pip install transformers==4.24.0
```

## Usage
Here we can obtain examples of [finetuning](examples/RuGPT3FinetuneHF.ipynb) or [generation](examples/Generate_text_with_RuGPTs_HF.ipynb).

Also this examples is adapted for google colab:
* finetuning: [![finetuning](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/ru-gpts/blob/master/examples/RuGPT3FinetuneHF.ipynb).
* generation: [![generation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/ru-gpts/blob/master/examples/Generate_text_with_RuGPTs_HF.ipynb)

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

### Pretraining details
All pretraining was done on Nvidia Tesla V100-SXM3 32 Gb GPUs on a [Christofari Cluster](https://en.wikipedia.org/wiki/Christofari). Following are the details of pretraining for each model.

#### Pretraining details ruGPT3Large
Model was trained with sequence length 1024 using transformers lib by [Devices](https://sberdevices.ru/) team on 80B tokens for 3 epochs. After that model was finetuned 1 epoch with sequence length 2048. 

Total training time was around 14 days on 128 GPUs for 1024 context and few days on 16 GPUs for 2048 context.  
Final perplexity on test set is `13.6`.

You can obtain this model by using transformers with model name `sberbank-ai/rugpt3large_based_on_gpt2`.

🤗HuggingFace model card [link](https://HuggingFace.co/sberbank-ai/rugpt3large_based_on_gpt2)

Our pretraining script [here](scripts/deepspeed_gpt3_large.sh)

#### Pretraining details ruGPT3Medium
Model was trained with sequence length 1024 using transformers lib by [Devices](https://sberdevices.ru/) team on 80B tokens for 3 epoch. After that model was finetuned on 2048 context.

Total training time was around 16 days on 64 GPUs.  
Final perplexity on test set is `17.4`.

You can obtain this model by using transformers with model name `sberbank-ai/rugpt3medium_based_on_gpt2`. 

🤗HuggingFace model card [link](https://HuggingFace.co/sberbank-ai/rugpt3medium_based_on_gpt2)

Our pretraining script [here](scripts/deepspeed_gpt3_medium.sh)

#### Pretraining details ruGPT3Small
Model was trained with sequence length 1024 using transformers by [Devices](https://sberdevices.ru/) team on 80B tokens around 3 epoch. After that model was finetuned on 2048 context.

Total training time took around one week on 32 GPUs.

You can obtain this model by using transformers with model name `sberbank-ai/rugpt3small_based_on_gpt2`. 

🤗HuggingFace model card [link](https://HuggingFace.co/sberbank-ai/rugpt3small_based_on_gpt2)

Our pretraining script [here](scripts/deepspeed_gpt3_small.sh)

#### Pretraining details ruGPT2Large
Model was trained with sequence length 1024 using transformers by [Devices](https://sberdevices.ru/) team on 170Gb data on 64 GPUs 3 weeks.

You can obtain this model by using transformers with model name `sberbank-ai/rugpt2large`.

🤗HuggingFace model card [link](https://HuggingFace.co/sberbank-ai/rugpt2large)

## OpenSource Solutions with ruGPT3

* ruCLIP [Github](https://github.com/ai-forever/ru-clip)
* Simplification with ruGPT-3 XL [Github](https://github.com/Alenush/rugpt3simplification_rsse )
* Word normalization (RuNormAS shared task) [Github](https://github.com/RussianNLP/RuNormAS-solution)
* AI CopyWriter [Github](https://github.com/dilyararimovna/text_expansion)
* ЕГЭ Generation [Github](https://github.com/orzhan/rugpt3-question-generation )
* NeuroZhirinovsky [Github](https://github.com/GraphGrailAi/ruGPT3-ZhirV)
* PseudoKant [Github](https://github.com/AsakoKabe/pseudo-kant )
* DostoevskyDoesntWriteIt [Github](https://github.com/K7chyp/DostoevskyDoesntWriteIt)


## Papers mentioning ruGPT3
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
  author={Fenogenova, Alena},
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
