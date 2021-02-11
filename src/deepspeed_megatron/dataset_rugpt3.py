import logging
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class RuGpt3DatasetArguments:
    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs"
                    " (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    random_shift: bool = field(default=False, metadata={"help": "Make random shift from start of each file"})
    max_files_load: int = field(default=50000, metadata={"help": "Maximum number of files to load at one worker"})
    tqdm: bool = field(default=False, metadata={"help": "Show tqdm progress bar"})


class RuGpt3TextDataset(Dataset):
    def process_file(self, file_path, filename, tokenizer, args):
        cached_features_file = os.path.join(self._cache_dir, filename.replace('/', '_') + '.pkl')
        examples = []

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            with open(cached_features_file, "rb") as handle:
                try:
                    examples = pickle.load(handle)
                    examples = np.asarray(examples, dtype=np.int32)
                except Exception as e:
                    print('Failed to load cache file:', cached_features_file)
                    raise e
        else:
            examples = []
            with open(os.path.join(file_path, filename), encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            max_shift = max(min(args.block_size, len(tokenized_text) - args.block_size), 0)
            rnd_shift = random.randrange(max_shift) if max_shift and args.random_shift else 0

            for i in range(rnd_shift, len(tokenized_text) - args.block_size + 1, args.block_size):
                example = tokenized_text[i:i + args.block_size]
                if None in example:
                    raise Exception('None in tokens!: ' + filename)
                if len(example) == args.block_size:
                    examples.append(example)
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            with open(cached_features_file, "wb") as handle:
                pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            examples = np.asarray(examples, dtype=np.int32)

        return examples

    def __init__(self, tokenizer, args, rank, world_size, file_path, cache_prefix='_'):
        self.rank = rank
        self.world_size = world_size
        self.log(f"Loading dataset {file_path}")
        max_file_load = args.max_files_load

        file_with_list = file_path
        file_path = os.path.dirname(file_with_list)
        self.log(f"Check filelist {file_with_list} with root dir {file_path}")

        if not os.path.exists(file_with_list) and rank < 1:
            raise Exception('No file list!')

        with open(file_with_list, 'r') as fp:
            files = [line.strip() for line in fp.read().split('\n') if line]

        if rank == -1:
            self.log('Shuffle')
            random.shuffle(files)
            if len(files) > max_file_load:
                files = files[:max_file_load]
        else:
            shard_size = len(files) // world_size
            if shard_size > max_file_load:
                logger.warning(
                    f"Shard size {shard_size} > max_file_load {max_file_load},"
                    f" only first {(max_file_load * world_size)}"
                    f" files of dataset would be loaded!")
                shard_size = max_file_load
            shard_start = rank * shard_size
            shard_end = (rank + 1) * shard_size
            self.log(f"Shard [{shard_start}, {shard_end}]")
            files = files[shard_start:shard_end]

        self._cache_dir = os.path.join(file_path, f'{cache_prefix}cache_{args.block_size}_{len(tokenizer)}')
        os.makedirs(self._cache_dir, exist_ok=True)
        if args.overwrite_cache:
            self.log('Overwrite cache ' + self._cache_dir)

        examples = []
        iterator = tqdm(files) if args.tqdm else files
        for i, filename in enumerate(iterator):
            if i % 1000 == 0:
                self.log(f"Loaded {i}/{len(files)} files")
            example = self.process_file(file_path, filename, tokenizer, args=args)
            if example.size:
                examples.append(example)
        self.examples = np.vstack(examples)
        np.random.shuffle(self.examples)
        self.log(f"Loaded {len(self.examples)} examples, {self.examples.size} tokens")

    def log(self, msg):
        logger.warning(f"R{self.rank}/{self.world_size}: {msg}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        item = item % len(self.examples)  # infinite loop, modulo dataset size
        if len(self.examples[item]) == 0:
            item = random.randint(1, len(self.examples))
        return torch.tensor(self.examples[item], dtype=torch.long)
