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

"""Utilities for logging and serialization"""

import os
import random
import time

import numpy as np
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from src import mpu
from src.fp16 import FP16_Optimizer, FP16_Module
from src.model import DistributedDataParallel as DDP
import os
from src.download_utils import download_model_files


class DeepSpeedImportWrap(object):
    def __init__(self):
        self.use_ds = os.environ.get("USE_DEEPSPEED", False)
        self.deepspeed = None
        if self.use_ds:
            import deepspeed
            self.deepspeed = deepspeed

    def __bool__(self):
        return bool(self.use_ds)


DEEPSPEED_WRAP = DeepSpeedImportWrap()


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, model-parallel,min, max, norm\n'
    optimizer_ = optimizer
    if isinstance(optimizer, FP16_Optimizer):
        optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = param.data.norm()
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


class Timers:
    """Group of timers."""

    class Timer:
        """Timer."""

        def __init__(self, name):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            """Start the timer."""
            assert not self.started_, 'timer has already been started'
            torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            torch.cuda.synchronize()
            self.elapsed_ += (time.time() - self.start_time)
            self.started_ = False

        def reset(self):
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self.elapsed_
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        print_rank_0(string)


def report_memory(name):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | cached: {}'.format(torch.cuda.memory_cached() / mega_bytes)
    string += ' | max cached: {}'.format(
        torch.cuda.max_memory_cached() / mega_bytes)
    print_rank_0(string)


def get_checkpoint_name(checkpoints_path, iteration, release=False, zero=False):
    if release:
        d = 'release'
    else:
        d = 'iter_{:07d}'.format(iteration)
    if zero:
        dp_rank = mpu.get_data_parallel_rank()
        d += '_zero_dp_rank_{}'.format(dp_rank)
    return os.path.join(checkpoints_path, d,
                        'mp_rank_{:02d}'.format(mpu.get_model_parallel_rank()),
                        'model_optim_rng.pt')


def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def save_zero_checkpoint(args, iteration, optimizer):
    zero_sd = {'iteration': iteration,
               'optimizer_state_dict': optimizer.state_dict()}
    zero_checkpoint_name = get_checkpoint_name(args.save, iteration, zero=True)
    ensure_directory_exists(zero_checkpoint_name)
    torch.save(zero_sd, zero_checkpoint_name)
    print('  successfully saved {}'.format(zero_checkpoint_name))


def save_checkpoint(iteration, model, optimizer,
                    lr_scheduler, args, deepspeed=False):
    """Save a model checkpoint."""
    if deepspeed:
        save_ds_checkpoint(iteration, model, args)
    else:
        # Only rank zer0 of the data parallel writes to the disk.
        if isinstance(model, torchDDP):
            model = model.module

        if mpu.get_data_parallel_rank() == 0:
            checkpoint_name = get_checkpoint_name(args.save, iteration)
            print('global rank {} is saving checkpoint at iteration {:7d} to {}'.
                  format(torch.distributed.get_rank(), iteration, checkpoint_name))

            sd = {}
            sd['iteration'] = iteration
            sd['model'] = model.state_dict()

            # Optimizer stuff.
            if not args.no_save_optim:
                if optimizer is not None:
                    sd['optimizer'] = optimizer.state_dict()
                if lr_scheduler is not None:
                    sd['lr_scheduler'] = lr_scheduler.state_dict()

            # rng states.
            if not args.no_save_rng:
                sd['random_rng_state'] = random.getstate()
                sd['np_rng_state'] = np.random.get_state()
                sd['torch_rng_state'] = torch.get_rng_state()
                sd['cuda_rng_state'] = torch.cuda.get_rng_state()
                sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()

            ensure_directory_exists(checkpoint_name)
            torch.save(sd, checkpoint_name)
            print('  successfully saved {}'.format(checkpoint_name))

    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()


def save_ds_checkpoint(iteration, model, args):
    """Save a model checkpoint."""

    sd = {}
    sd['iteration'] = iteration
    # rng states.
    if not args.no_save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
        sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()

    model.save_checkpoint(args.save, str(iteration), client_state=sd)


def get_checkpoints(load_dir):
    return [f for f in os.listdir(load_dir) if '.' not in f]


def get_last_checkpoints(load_dir, n=1):
    checkpoints = get_checkpoints(load_dir)
    checkpoints = [(int(c), c) for c in checkpoints]
    by_iteration = sorted(checkpoints, key=lambda x: x[0], reverse=True)
    return [b[-1] for b in by_iteration[:n]]


def get_outdated_checkpoints(load_dir, retain_last_n=5):
    last = get_last_checkpoints(load_dir, retain_last_n)
    return [d for d in get_checkpoints(load_dir) if d not in last]


def get_checkpoint_iteration(args):
    # Read the tracker file and set the iteration.
    if args.load_tag:
        return args.load_tag, False, True
    tracker_filename = get_checkpoint_tracker_filename(args.load)
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        print_rank_0('    will not load any checkpoints and will start from '
                     'random')
        return 0, False, False
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                exit()

    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)

    return iteration, release, True


def load_checkpoint(model, optimizer, lr_scheduler, args, deepspeed=False):
    """Load a model checkpoint."""

    iteration, release, success = get_checkpoint_iteration(args)

    if not success:
        return 0

    if deepspeed:
        load_optim = not args.no_load_optim
        checkpoint_name, sd = model.load_checkpoint(args.load, iteration, load_optimizer_states=load_optim,
                                                    load_lr_scheduler_states=load_optim)

        if checkpoint_name is None:
            if mpu.get_data_parallel_rank() == 0:
                print("Unable to load checkpoint.")
            return iteration

    else:

        # Checkpoint.
        checkpoint_name = get_checkpoint_name(args.load, iteration, release)

        # Load the checkpoint.
        if os.path.isfile(checkpoint_name):
            sd = torch.load(checkpoint_name, map_location='cpu')
        else:
            # Try load deepspeed checkpoint with only megatron
            checkpoint_name = os.path.join(
                args.load, str(iteration),
                'mp_rank_{:02d}_model_states.pt'.format(mpu.get_model_parallel_rank())
            )
            sd = torch.load(checkpoint_name, map_location='cpu')

        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        if isinstance(model, torchDDP):
            model = model.module

        # Model.
        try:
            model.load_state_dict(sd['model'])
        except KeyError:
            try:
                model.load_state_dict(sd['module'])
            except KeyError:
                print_rank_0('A metadata file exists but unable to load model '
                             'from checkpoint {}, exiting'.format(checkpoint_name))
                exit()

        # Optimizer.
        if not release and not args.finetune and not args.no_load_optim:
            try:
                if optimizer is not None:
                    optimizer.load_state_dict(sd['optimizer'])
                if lr_scheduler is not None:
                    lr_scheduler.load_state_dict(sd['lr_scheduler'])
            except KeyError:
                print_rank_0('Unable to load optimizer from checkpoint {}, exiting. '
                             'Specify --no-load-optim or --finetune to prevent '
                             'attempting to load the optimizer '
                             'state.'.format(checkpoint_name))
                exit()

    # Iterations.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = sd['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = sd['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but Unable to load iteration '
                             ' from checkpoint {}, exiting'.format(checkpoint_name))
                exit()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.cuda.set_rng_state(sd['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}, exiting. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer '
                         'state.'.format(checkpoint_name))
            exit()

    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration


def load_weights(src, dst, dst2src=False, double_pos_embeddings=False):
    """
    Loads weights from src to dst via in place copy.
    src is a huggingface gpt3model, while dst is one of our models.
    dst2src=True loads parameters from our models into huggingface's.
    ^dst2src is still untested
    """
    conv_layer = 'Conv1D' in str(type(src))
    for n, p in src.named_parameters():
        if dst2src:
            data = dst._parameters[n].data
            load = p.data
        else:
            if double_pos_embeddings:
                print('Double pos embeddings')
                mid = p.size(0) // 2
                p[mid:, :] = p[:mid, :]  # copy first half of position embedings to last
            data = p.data
            load = dst._parameters[n].data
        if conv_layer and 'weight' in n:
            data = data.t().contiguous()
        load.copy_(data)


#        dst._parameters[n].data.copy_(data)

def load_mlp(our, oai, dst2src=False):
    load_weights(oai.c_fc, our.dense_h_to_4h, dst2src)
    load_weights(oai.c_proj, our.dense_4h_to_h, dst2src)


def load_attention(our, oai, dst2src=False):
    load_weights(oai.c_attn, our.query_key_value, dst2src)
    load_weights(oai.c_proj, our.dense, dst2src)


def load_transformer_layer(our, oai, dst2src=False):
    load_weights(oai.ln_1, our.input_layernorm, dst2src)
    load_weights(oai.ln_2, our.post_attention_layernorm, dst2src)
    load_mlp(our.mlp, oai.mlp, dst2src)
    load_attention(our.attention, oai.attn, dst2src)


def move_weights(our, oai, dst2src=False, double_pos_embeddings=False):
    """
    Loads weights from `oai` to `our` via in place copy.
    `oai` is a huggingface gpt3model, while `our` is one of our models.
    dst2src=True loads parameters from our models into huggingface's.
    ^dst2src=True is still untested
    """
    #    while isinstance(our, (torchDDP, model.distributed.DistributedDataParallel, FP16_Module)):
    #        our=our.module
    transformer_model = oai.transformer
    load_weights(transformer_model.ln_f, our.transformer.final_layernorm, dst2src)
    load_weights(transformer_model.wte, our.word_embeddings, dst2src)
    load_weights(transformer_model.wpe, our.position_embeddings, dst2src, double_pos_embeddings=double_pos_embeddings)

    for our_layer, oai_layer in zip(our.transformer.layers, oai.transformer.h):
        load_transformer_layer(our_layer, oai_layer, dst2src)


def load_huggingface_model(model, path, double_pos_embeddings):
    from transformers import GPT2LMHeadModel
    print('Load huggingface model from', path, ('with pos emb doubling' if double_pos_embeddings else ''))
    model2fill = model
    while isinstance(model2fill, (torchDDP, FP16_Module)):
        model2fill = model2fill.module
    
    if path == "sberbank-ai/rugpt3xl":
        weights_path, _ = download_model_files("sberbank-ai/rugpt3xl")
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)['module']
        model2fill.load_state_dict(checkpoint)
    else:
        h_model = GPT2LMHeadModel.from_pretrained(path)
        move_weights(model2fill, h_model, double_pos_embeddings)

    print('Loaded huggingface model', type(model))
    return model


def export_to_huggingface_model(model, path):
    from transformers import GPT2LMHeadModel, GPT2Config
    model_from = model
    while isinstance(model_from, (DDP, torchDDP, FP16_Module)):
        model_from = model_from.module
    conf_dict = model_from._conf_dict
    print('Export to huggingface model ', path, 'with config', conf_dict)
    config = GPT2Config(**conf_dict)
    hf_model = GPT2LMHeadModel(config=config)
    model_to = hf_model
    while isinstance(model_to, (DDP, torchDDP, FP16_Module)):
        model_to = model_to.module
    move_weights(model_from, model_to, dst2src=True)
    hf_model.save_pretrained(path)
    print('Saved huggingface model', type(model))


def get_deepspeed_config(args):
    if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
        from deepspeed import DeepSpeedConfig
        return DeepSpeedConfig(args.deepspeed_config)
    else:
        raise RuntimeError('deepspeed_config is not found in args.')


def get_sparse_attention_config(args, num_heads):
    ds_config = get_deepspeed_config(args)
    if hasattr(ds_config, 'sparse_attention') and ds_config.sparse_attention:
        sa_config = ds_config.sparse_attention
        sa_mode = sa_config.get('mode')
        if (sa_mode == 'dense'):
            from deepspeed.ops.sparse_attention import DenseSparsityConfig as STConfig
        elif (sa_mode == 'fixed'):
            from deepspeed.ops.sparse_attention import FixedSparsityConfig as STConfig
        elif (sa_mode == 'bigbird'):
            from deepspeed.ops.sparse_attention import BigBirdSparsityConfig as STConfig
        elif (sa_mode == 'bslongformer'):
            from deepspeed.ops.sparse_attention import BSLongformerSparsityConfig as STConfig
        elif (sa_mode == 'variable'):
            from deepspeed.ops.sparse_attention import VariableSparsityConfig as STConfig
        else:
            raise NotImplementedError(
                f'Given sparsity mode, {sa_mode}, has not been implemented yet!'
            )
        del sa_config['mode']
        return STConfig(num_heads=num_heads, **sa_config)
    else:
        return None


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        logits = logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        # going back to 2D
        logits = logits.view(1, -1).contiguous()

    return logits
