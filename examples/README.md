# Examples
All instructions was written for [Christophari](https://sbercloud.ru/ru/christofari).

For contest: you can obtain checkpoints from aws s3.

| Section                    | Description                                                                                                                                                |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------
| [ruGPT2048 fintune on essays](#ruGPT2048 finetune on essays) | Examples of finetuning ruGPT2048 model for generating school essays. |
| [ruGPT2Large fintune on essays](#ruGPT2Large finetune on essays) | Examples of finetuning ruGPT2Large model for generating school essays. |


## ruGPT2048 fintune on essays
Finetune ruGPT2048 for school essays generation.

We prepare data with the following format:

```
{"text": "Тема: С какой целью В.А. Жуковский вносит русские фольклорные мотивы в традиционный балладный сюжет? (по балладе «Светлана»)\nСочинение: ..."}
```

For run finetuning download ruGPT2048 [checkpoint](https://drive.google.com/file/d/12JkbnzSoQwJqanVP-zoLNnFX3e4HHyvY/view?usp=sharing) and unpack to `/home/jovyan/rugpt2048`:

```
tar -zxvf rugpt2048.tar.gz
```

Download data to `/home/jovyan/data`. Data you can obtain here: [train](https://drive.google.com/file/d/1XEJWoVsZhDwrKy801y9K6iljDJdM0_zs/view?usp=sharing) and [valid](https://drive.google.com/file/d/1s5b7WvyCBB9nPprEXPgs45ljqhb8dQsn/view?usp=sharing).

Run script for pretrain: `bash ./examples/pretrain_ruGPT2048_essay.sh`.

We obtain around 8 perplexity on valid set. Sample of generation you can see [here](./pretrain_ruGPT2048_essay_sample.txt)

You can download pretrained checkpoint [here](https://drive.google.com/file/d/13ezv9NpquKCB5TAgKC0jRRfxUKjzc7Mp/view?usp=sharing).

## ruGPT2Large fintune on essays

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

You can download pretrained checkpoint [here]().
