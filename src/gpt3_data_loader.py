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

import os

import torch
from torch.utils.data import BatchSampler, DataLoader

from src import mpu
from src.dataset_rugpt3 import RuGpt3TextDataset, RuGpt3DatasetArguments
from src.utils import print_rank_0
from transformers import GPT2Tokenizer


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


class ResumableBatchSampler(BatchSampler):
    start_iter = 0

    def __iter__(self):
        batch = []
        i = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if i >= self.start_iter:
                    yield batch
                batch = []
                i += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch


def make_gpt3_dataloaders(args):
    # Data parallel arguments
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    # global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    # data_dir = args.train_data_path if args.train_data_path else os.path.dirname(args.test_data_path)
    tokenizer_path = args.load_huggingface if args.load_huggingface else \
        (args.tokenizer_path if args.tokenizer_path else os.path.join(os.path.dirname(args.train_data_path),
                                                                      '_tokenizer/'))
    print_rank_0('Load tokenizer from ' + tokenizer_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    eod_token = tokenizer.encoder['<pad>']
    num_tokens = len(tokenizer)

    train_dataset_args = RuGpt3DatasetArguments(
        block_size=args.seq_length, max_files_load=args.max_files_per_process, overwrite_cache=args.overwrite_cache,
        tqdm=False)
    eval_dataset_args = RuGpt3DatasetArguments(
        block_size=args.seq_length, max_files_load=args.max_files_per_process, overwrite_cache=args.overwrite_cache,
        tqdm=True)

    def make_data_loader_(data_path, dataset_args):
        print_rank_0(f'Load RuGPT3 Dataset from {data_path}, {dataset_args.max_files_load} files per process')
        dataset = RuGpt3TextDataset(
            tokenizer=tokenizer,
            args=dataset_args,
            rank=rank,
            world_size=world_size,
            file_path=data_path,
            # cache_prefix=args.cache_prefix
        )
        # Use a simple sampler with distributed batch sampler.
        sampler = torch.utils.data.SequentialSampler(dataset)
        batch_sampler = ResumableBatchSampler(sampler=sampler,
                                              batch_size=args.batch_size,
                                              drop_last=True)

        return InfiniteDataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True)

    train = make_data_loader_(args.train_data_path, train_dataset_args) if args.train_data_path else None
    valid = make_data_loader_(args.val_data_path, eval_dataset_args) if args.val_data_path else None
    test = make_data_loader_(args.test_data_path, eval_dataset_args) if args.test_data_path else None

    args.do_train = train is not None
    args.do_valid = valid is not None
    args.do_test = test is not None

    return (train, valid, test), num_tokens, eod_token, tokenizer
