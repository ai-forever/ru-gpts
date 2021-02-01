# ruGPT3XL

ruGPT3XL model is GPT3 model with sparse multi-head attention layers which alternates with dense layers.
We use [deepspeed](https://github.com/microsoft/DeepSpeed) implementation of sparse attention layers instead of previous custom realisation.

## Details of pretraining ruGPT3XL
Model was trained on 512 context length with [deepspeed](https://github.com/microsoft/DeepSpeed) and [megatron](https://github.com/NVIDIA/Megatron-LM) code by [SberDevices](https://sberdevices.ru/) team. After that model was finetuned on 2048 context.

Total training time took around 10 days on 256 GPUs. Final perplexity on test set is `11.4`.

🤗HuggingFace model card [link](https://huggingface.co/sberbank-ai/rugpt3xl).

## Setup
Run the following command:

```bash
pip install -r gw/requirements.txt
```

If you have errors with deepspeed you can install this manually:

```bash
pip install transformers==3.5.1
pip install natsort
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension

pip install triton==0.2.3
DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install deepspeed==0.3.7
```

After that you should check deepspeed installation:

```bash
ds_report
```

We should something like this:
```text
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [OKAY]
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
cpu_adam ............... [YES] ...... [OKAY]
fused_adam ............. [NO] ....... [OKAY]
fused_lamb ............. [NO] ....... [OKAY]
sparse_attn ............ [YES] ...... [OKAY]
transformer ............ [NO] ....... [OKAY]
stochastic_transformer . [NO] ....... [OKAY]
utils .................. [NO] ....... [OKAY]
--------------------------------------------------
DeepSpeed general environment info:
torch install path ............... ['/home/user/conda/lib/python3.7/site-packages/torch']
torch version .................... 1.6.0+cu101
torch cuda version ............... 10.1
nvcc version ..................... 10.1
deepspeed install path ........... ['/home/user/conda/lib/python3.7/site-packages/deepspeed']
deepspeed info ................... 0.3.7+ff58fa7, ff58fa7, HEAD
deepspeed wheel compiled w. ...... torch 1.6, cuda 10.1
```

If you have error while generation (see next section) with triton lib try reinstall triton:

```bash
pip install triton==0.2.2
```

Note! All installation pipeline was tested on linux gpu server with cuda.

## Usage
Model has been added to huggingface and we can [download](https://huggingface.co/sberbank-ai/rugpt3xl) this by our huggingface wrapper for this model.

```python
import sys
sys.path.append("gw/")

from generation_wrapper import RuGPT3XL


gpt = RuGPT3XL.from_pretrained("sberbank-ai/rugpt3xl", seq_len=512)
res = gpt.generate(
    "Кто был президентом США в 2020? ",
    max_length=50,
    no_repeat_ngram_size=3,
    repetition_penalty=2.,
)
print(res)
# ['Кто был президентом США в 2020? \nВ этом году выборы президента Соединенных Штатов Америки пройдут уже через несколько дней. И, как и всегда на протяжении последних лет (а это более чем 20-ти), кандидаты будут бороться за право стать главой государств']
```

More examples see [here](examples/ruGPT3XL_generation.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/ruGPT3XL_generation.ipynb)
