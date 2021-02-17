# coding=utf-8
# Copyright (c) 2020, Sber.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT3"""

import os
import time

import torch
from transformers.tokenization_gpt2 import GPT2Tokenizer

from src import mpu
from src.arguments import get_args
from src.fp16 import FP16_Module
from src.model import DistributedDataParallel as DDP
from src.model import GPT3Model
from pretrain_gpt3 import generate
from pretrain_gpt3 import initialize_distributed
from pretrain_gpt3 import set_random_seed
from src.utils import Timers
from src.utils import export_to_huggingface_model
from src.utils import print_rank_0, load_checkpoint, DEEPSPEED_WRAP


def get_model(args):
    """Build the model."""

    print_rank_0('building GPT3 model ...')
    model = GPT3Model(num_layers=args.num_layers,
                      vocab_size=args.vocab_size,
                      hidden_size=args.hidden_size,
                      num_attention_heads=args.num_attention_heads,
                      embedding_dropout_prob=args.hidden_dropout,
                      attention_dropout_prob=args.attention_dropout,
                      output_dropout_prob=args.hidden_dropout,
                      max_sequence_length=args.max_position_embeddings,
                      checkpoint_activations=args.checkpoint_activations,
                      checkpoint_num_layers=args.checkpoint_num_layers,
                      parallel_output=False)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    model = DDP(model)

    return model


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)
    if DEEPSPEED_WRAP and args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = DEEPSPEED_WRAP.deepspeed.initialize(
            model=model,
            optimizer=None,
            args=args,
            lr_scheduler=None,
            mpu=mpu,
            dist_init_required=False
        )

    print("Load checkpoint from " + args.load)
    _ = load_checkpoint(model, None, None, args, deepspeed=DEEPSPEED_WRAP and args.deepspeed)
    model.eval()
    print("Loaded")
    if args.export_huggingface is not None:
        export_to_huggingface_model(model, args.export_huggingface)
        print(f"Exported in huggingface format to {args.export_huggingface}")

    return model


def generate_samples(model, tokenizer, args):
    model.eval()
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs = 0

            if mpu.get_model_parallel_rank() == 0:
                raw_text = input("\nContext prompt (stop to exit) >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("\nContext prompt (stop to exit) >>> ")

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer(raw_text)['input_ids']
                    context_length = len(context_tokens)

                    if context_length >= args.seq_length // 2:
                        print("\nContext length", context_length,
                              "\nPlease give smaller context (half of the sequence length)!")
                        continue
            else:
                _ = tokenizer("EMPTY TEXT")['input_ids']

            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(),
                                        group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            start_time = time.time()
            generated = generate(
                model, tokenizer, raw_text,
                out_seq_length=args.out_seq_length,
                seq_length=args.seq_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )

            if mpu.get_model_parallel_rank() == 0:
                os.system('clear')
                print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                print("\nContext:", raw_text, flush=True)
                print("\nGPT:", generated, flush=True)
            raw_text = None

            torch.distributed.barrier(group=mpu.get_model_parallel_group())


def prepare_tokenizer(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    eod_token = tokenizer.encoder['<pad>']
    num_tokens = len(tokenizer)

    args.tokenizer_num_tokens = num_tokens
    args.eod_token = eod_token

    after = num_tokens
    while after % args.make_vocab_size_divisible_by != 0:
        after += 1

    args.vocab_size = after
    print(f"prepare tokenizer done, size {after}", flush=True)

    return tokenizer


def main():
    """Main training program."""

    print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    _ = Timers()

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    # setting default batch size to 1
    args.batch_size = 1

    # generate samples
    generate_samples(model, tokenizer, args)


if __name__ == "__main__":
    main()
