import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from transformers import GPT2LMHeadModel,GPT2Tokenizer
import regex as re
from os import environ

device = environ.get('DEVICE', 'cuda:2')
model_path = environ.get('MODEL', 'poetry')
flavor_id = model_path + device + environ.get('INSTANCE', ':0')

from tendo import singleton
me = singleton.SingleInstance(flavor_id=flavor_id)

import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=f"logs/{hash(flavor_id)}.log", level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.to(device)
model.eval()

from apex import amp
model = amp.initialize(model, opt_level='O2')

def get_sample(prompt, length:int, num_samples:int, allow_linebreak:bool):
    logger.info(prompt)
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    bad_words_ids = [tokenizer.encode('[')[0], tokenizer.encode('(')[0], tokenizer.encode('1\xa01')[1]]
    linebreak = tokenizer.encode("1\n1")[1]
    bad_words_ids += [] if allow_linebreak else [linebreak]
    bad_words_ids = [[b] for b in bad_words_ids] + [[linebreak,linebreak]]
    output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            temperature=1,
            top_k=0,
            top_p=0.9,
            do_sample=True,num_return_sequences=num_samples,
            bad_words_ids = bad_words_ids
        )
    
    if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()
    generated_sequences = []
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        total_sequence = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        generated_sequences.append(total_sequence)

    reg_text = [re.match(r'[\w\W]*[\.!?]\n', item) for item in generated_sequences]
    reg_text2 = [re.match(r'[\w\W]*[\.!?]', item) for item in generated_sequences]
    result = [reg_item[0] if reg_item else reg_item2[0] if reg_item2 else item for reg_item, reg_item2, item in zip(reg_text, reg_text2, generated_sequences)]
    return result

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import threading

app = FastAPI(title="Russian GPT-2", version="0.2",)
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

lock = threading.RLock()

class Prompt(BaseModel):
    prompt:str = Field(..., max_length=3000, title='Model prompt')
    length:int = Field(15, ge=1, le=60, title='Number of tokens generated in each sample')
    num_samples:int = Field(3, ge=1, le=5, title='Number of samples generated')
    allow_linebreak:bool = Field(False, title='Allow linebreak in a sample')

@app.post("/generate/")
def gen_sample(prompt: Prompt):
    with lock:
        return {"replies": get_sample(prompt.prompt, prompt.length, prompt.num_samples, prompt.allow_linebreak)}

@app.get("/health")
def healthcheck():
    return True