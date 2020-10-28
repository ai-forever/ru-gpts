# Examples
All instructions was written for [Christophari](https://sbercloud.ru/ru/christofari).

For contest: you can obtain checkpoints from aws s3.

| Section                    | Description                                                                                                                                                |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------
| [ruGPT3Large finetune on essays](#ruGPT3Large-finetune-on-essays) | Examples of finetuning ruGPT3Large model for generating school essays. |
| [ruGPT2Large finetune on essays](#ruGPT2Large-finetune-on-essays) | Examples of finetuning ruGPT2Large model for generating school essays. |
| [ruGPT3Small finetune on essays](#ruGPT3Small-finetune on-essays) | Examples of finetuning ruGPT3Small model for generating school essays in colab. |
| [ruGPT3Large generate](#ruGPT3Large-generate) | Examples of generate with ruGPT3Large model in colab. |


## ruGPT3Large finetune on essays
Finetune ruGPT3Large for school essays generation.

We prepare data with the following format:

```
{"text": "Тема: С какой целью В.А. Жуковский вносит русские фольклорные мотивы в традиционный балладный сюжет? (по балладе «Светлана»)\nСочинение: ..."}
```

For run finetuning download ruGPT3Large [checkpoint](https://drive.google.com/file/d/12JkbnzSoQwJqanVP-zoLNnFX3e4HHyvY/view?usp=sharing) and unpack to `/home/jovyan/ruGPT3Large`:

```
tar -zxvf ruGPT3Large.tar.gz
```

Download data to `/home/jovyan/data`. Data you can obtain here: [train](https://drive.google.com/file/d/1XEJWoVsZhDwrKy801y9K6iljDJdM0_zs/view?usp=sharing) and [valid](https://drive.google.com/file/d/1s5b7WvyCBB9nPprEXPgs45ljqhb8dQsn/view?usp=sharing).

Run script for pretrain: `bash ./examples/pretrain_ruGPT3Large_essay.sh`.

We obtain around 8 perplexity on valid set. Sample of generation you can see [here](pretrain_ruGPT3Large_essay_sample.txt)

You can download pretrained checkpoint [here](https://drive.google.com/file/d/13ezv9NpquKCB5TAgKC0jRRfxUKjzc7Mp/view?usp=sharing).

## ruGPT2Large finetune on essays

Finetune ruGPT2Large for school essays generation.

We prepare data with the following format (raw text):

```
<s>Тема: С какой целью В.А. Жуковский вносит русские фольклорные мотивы в традиционный балладный сюжет? (по балладе «Светлана»)\nСочинение: ...</s>
<s>Тема: ...
```

For run finetuning download ruGPT2Large [checkpoint](https://drive.google.com/file/d/1r65MwU0arie8NggxpSmc_3Ja5ldRNS70/view?usp=sharing) and unpack to `/home/jovyan/gpt2_large_bbpe_v50`:

```
tar -zxvf gpt2_large_bbpe_v50.tar.gz
```

Download data to `/home/jovyan/data`. Data you can obtain here: [train](https://drive.google.com/file/d/1CBXZjcNcqGdiyChzSlffVqIaeCp-7486/view?usp=sharing) and [valid](https://drive.google.com/file/d/1MhmPhj-VKCTmCWXf6WfR3Czuw3V7QMB9/view?usp=sharing).

Run script for pretrain: `bash ./examples/pretrain_ruGPT2Large_essay.sh`.

We obtain around 3 perplexity on valid set. Sample of generation you can see [here](./pretrain_ruGPT2Large_essay_sample.txt)

You can download pretrained checkpoint [here](https://drive.google.com/file/d/1AtK_2a-gx7-BBy8oBDlDSbbc0Z8JFCoa/view?usp=sharing).

## ruGPT3Small finetune on essays
Try example of finetune on essays and generation in colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/Finetune_ruGPT3Small.ipynb)

## ruGPT3Large generate
Try example of generation in colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-gpts/blob/master/examples/ruGPT3_generation_example.ipynb)
