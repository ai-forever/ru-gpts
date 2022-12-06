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

"""Pretrain GPT3"""

import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from apex.optimizers import FusedAdam as Adam
from torch.utils.tensorboard import SummaryWriter

from src import mpu
from src.arguments import get_args
from src.fp16 import FP16_Module
from src.fp16 import FP16_Optimizer
from src.gpt3_data_loader import make_gpt3_dataloaders
from src.learning_rates import AnnealingLR
from src.model import GPT3Model
from src.model import gpt3_get_params_for_weight_decay_optimization
from src.utils import (
    Timers, report_memory,
    save_checkpoint, load_checkpoint, load_huggingface_model,
    print_args, print_rank_0,
    get_sparse_attention_config, top_k_logits, DEEPSPEED_WRAP
)
from huggingface_hub import hf_hub_download
from src.download_utils import WEIGHTS_NAME

# Flag to use Pytorch ddp which uses overlapping communication and computation.
USE_TORCH_DDP = False
if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
else:
    from src.model import DistributedDataParallel as DDP


def get_model(args):
    """Build the model."""

    print_rank_0('building GPT3 model ...')
    assert args.num_attention_heads % args.model_parallel_size == 0
    num_local_heads = args.num_attention_heads // args.model_parallel_size
    deepspeed_sparsity_config = None
    if DEEPSPEED_WRAP and args.deepspeed:
        deepspeed_sparsity_config = get_sparse_attention_config(args, num_local_heads)
    if deepspeed_sparsity_config is not None:
        print_rank_0(f"Use sparse attention with mode {args.sparse_mode}")
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
                      parallel_output=True,
                      deepspeed_sparsity_config=deepspeed_sparsity_config,
                      sparse_mode=args.sparse_mode)

    if args.load_huggingface is not None:
        if args.load_huggingface == "sberbank-ai/rugpt3xl":
            weights_path = hf_hub_download(args.load_huggingface, WEIGHTS_NAME)
            checkpoint = torch.load(weights_path, map_location="cpu")['module']
            model.load_state_dict(checkpoint, strict=False)
        else:
            model = load_huggingface_model(model, args.load_huggingface, args.huggingface_double_pos_embeddings)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if DEEPSPEED_WRAP and args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        model = DDP(model)

    return model


def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (DDP, FP16_Module)):
        model = model.module
    param_groups = gpt3_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    if args.cpu_optimizer:
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                                       lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Use FusedAdam.
        optimizer = Adam(param_groups,
                         lr=args.lr, weight_decay=args.weight_decay)

    print(f'Optimizer = {optimizer.__class__.__name__}')
    if DEEPSPEED_WRAP and args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               min_lr=args.min_lr)

    return lr_scheduler


def setup_model_and_optimizer(args):
    """Setup model and optimizer."""

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if DEEPSPEED_WRAP and args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = DEEPSPEED_WRAP.deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=False
        )

    if args.load is not None:
        print_rank_0("Load checkpoint from " + args.load)
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args, deepspeed=DEEPSPEED_WRAP and args.deepspeed)
        print_rank_0("Checkpoint loaded")
    else:
        args.iteration = 0

    return model, optimizer, lr_scheduler


def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indices where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indices:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids


def get_batch(data, args, timers):
    """ get_batch subdivides the source data into chunks of
    length args.seq_length. If source is equal to the example
    output of the data loading example, with a seq_length limit
    of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the data loader. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM. A Variable representing an appropriate
    shard reset mask of the same dimensions is also returned.
    """

    # Broadcast data.
    data_b = mpu.broadcast_data(['text'], {'text': data}, torch.int64)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_.contiguous()
    tokens = tokens_.contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        args.reset_position_ids,
        args.reset_attention_mask)
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, loss_mask, attention_mask, position_ids


def forward_step(sample, model, args, timers, tokenizer=None, iteration=None, tb_writer=None):
    """Forward step."""

    # Get the batch.
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(sample, args, timers)

    # Forward model.
    output = model(tokens, position_ids, attention_mask)

    labels = labels[:, 1:].contiguous()
    output = output[:, :-1].contiguous()
    loss_mask = loss_mask[:, :-1].contiguous()

    losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)

    #     if tokenizer is not None and tb_writer is not None and iteration % 1000 == 0:
    #         try:
    #             inf_indexes = np.where(torch.isinf(losses).cpu())[0]
    #             nan_indexes = np.where(torch.isnan(losses).cpu())[0]
    #             if len(nan_indexes):
    #                 batch_text = ''
    #                 for i in nan_indexes:
    #                     ids = tokens[i].tolist()
    #                     batch_text += f"\n\nSample {i}: {tokenizer.decode(ids)}"
    #                 tb_writer.add_text('nan_loss', batch_text, iteration)
    #             if len(inf_indexes):
    #                 batch_text = ''
    #                 for i in inf_indexes:
    #                     ids = tokens[i].tolist()
    #                     batch_text += f"\n\nSample {i}: {tokenizer.decode(ids)}"
    #                 tb_writer.add_text('inf_loss', batch_text, iteration)
    #         except Exception as e:
    #             print(f"Exception during nan/inf logging: {e}")

    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    return loss


def backward_step(optimizer, model, lm_loss, args, timers):
    """Backward step."""

    # Total loss.
    loss = lm_loss

    # Backward pass.
    if DEEPSPEED_WRAP and args.deepspeed:
        model.backward(loss)
    else:
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

    # Reduce across processes.
    # lm_loss_reduced = lm_loss

    reduced_losses = lm_loss.view(1)

    if DEEPSPEED_WRAP and args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers('allreduce').reset()
    else:
        torch.distributed.all_reduce(reduced_losses.data)
        reduced_losses.data = reduced_losses.data / args.world_size
        if not USE_TORCH_DDP:
            timers('allreduce').start()
            model.allreduce_params(reduce_after=False,
                                   fp32_allreduce=args.fp32_allreduce)
            timers('allreduce').stop()

    lm_loss_reduced = reduced_losses

    # Update master gradients.
    if not (DEEPSPEED_WRAP and args.deepspeed):
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)

    return lm_loss_reduced


def log_memory_usage(tb_writer, iteration):
    dist.barrier()
    if dist.get_rank() == 0:
        alloc = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
        cache = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        max_cache = torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024)
        print("Memory Allocated ", alloc, "GigaBytes")
        print("Max Memory Allocated ", max_alloc, "GigaBytes")
        print("Cache Allocated ", cache, "GigaBytes")
        print("Max cache Allocated ", max_cache, "GigaBytes")
        if tb_writer is not None:
            tb_writer.add_scalar('mem/alloc', alloc, iteration)
            tb_writer.add_scalar('mem/max_alloc', max_alloc, iteration)
            tb_writer.add_scalar('mem/cache', cache, iteration)
            tb_writer.add_scalar('mem/max_cache', max_cache, iteration)


def train_step(sample, model, optimizer, lr_scheduler,
               args, timers, tokenizer, iteration, tb_writer):
    """Single training step."""

    # Forward model for one step.
    timers('forward').start()
    lm_loss = forward_step(sample, model, args, timers, tokenizer, iteration, tb_writer)
    timers('forward').stop()

    # print_rank_0("loss is {}".format(lm_loss))

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    lm_loss_reduced = backward_step(optimizer, model, lm_loss, args, timers)
    timers('backward').stop()

    # Update parameters.
    skipped_iter = 0
    timers('optimizer').start()
    if DEEPSPEED_WRAP and args.deepspeed:
        model.step()
    else:
        optimizer.step()

        # Update learning rate.
        if not (args.fp16 and optimizer.overflow):
            lr_scheduler.step()
        else:
            skipped_iter = 1
    timers('optimizer').stop()

    return lm_loss_reduced, skipped_iter


def train(model, optimizer, lr_scheduler,
          train_data_iterator, val_data, timers, args, tokenizer):
    """Train the model."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0

    # Iterations.
    iteration = args.iteration
    skipped_iters = 0
    tb_writer = None
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        tb_writer = SummaryWriter(log_dir=args.logging_dir)

    timers('interval time').start()
    report_memory_flag = True
    is_master = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    print('--Start training loop--')
    train_start = True
    # avg_lm_loss = 1e6
    while iteration < args.train_iters:
        timers('data loader').start()
        sample = next(train_data_iterator) if (train_data_iterator is not None) else None
        timers('data loader').stop()

        if train_start and is_master:
            batch_text = f"\n\Iteration {iteration} start sample: {tokenizer.decode(sample[0, :200])}"
            tb_writer.add_text('train_start', batch_text, iteration)

        lm_loss, skipped_iter = train_step(sample,
                                           model,
                                           optimizer,
                                           lr_scheduler,
                                           args, timers, tokenizer, iteration, tb_writer)
        skipped_iters += skipped_iter
        iteration += 1
        train_start = False

        # Update losses.
        total_lm_loss += lm_loss.data.detach().float()

        # Logging.
        if is_master and iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            ppl = math.exp(avg_lm_loss)
            elapsed_time = timers('interval time').elapsed()
            samples = args.log_interval * mpu.get_data_parallel_world_size() * args.batch_size
            tokens = samples * args.seq_length
            log_string = ' iteration {:8d}/{:8d} |'.format(iteration, args.train_iters)
            log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time * 1000.0 / args.log_interval)
            log_string += ' learning rate {:.3E} |'.format(learning_rate)
            log_string += ' lm loss {:.4f} |'.format(avg_lm_loss)
            log_string += ' perplexity {:.4f} |'.format(ppl)
            scalars = {
                'Loss/loss': avg_lm_loss,
                'Loss/perplexity': ppl,
                'learning_rate': learning_rate,
                'Speed/iteration_time_ms': (elapsed_time * 1000.0 / args.log_interval),
                'Speed/samples_per_sec': (samples / elapsed_time),
                'Speed/tokens_per_sec': (tokens / elapsed_time),
                'Speed/tokens_per_step': (tokens / args.log_interval),
                'Speed/seen_tokens': iteration * (tokens / args.log_interval)
            }
            if args.fp16:
                lscale = optimizer.cur_scale if DEEPSPEED_WRAP and args.deepspeed else optimizer.loss_scale
                log_string += ' loss scale {:.1f} |'.format(lscale)
                scalars['lscale'] = lscale
            print_rank_0(log_string)
            for k, v in scalars.items():
                tb_writer.add_scalar(k, v, iteration)

            if ppl < 3:
                # generate only when model is relatively good
                prefix = 'Бразильские ученые открыли редкий вид карликовых единорогов, обитающих на западе Ютландии'
                model.eval()
                with torch.no_grad():
                    text = generate(model, tokenizer, prefix, 128)
                model.train()
                tb_writer.add_text('sample', text, iteration)

            if args.log_memory:
                log_memory_usage(tb_writer, iteration)
            total_lm_loss = 0.0
            if report_memory_flag:
                report_memory('after {} iterations'.format(iteration))
                report_memory_flag = False
            if USE_TORCH_DDP:
                timers.log(['forward', 'backward', 'optimizer', 'data loader'], normalizer=args.log_interval)
            else:
                timers.log(['forward', 'backward', 'allreduce', 'optimizer', 'data loader'],
                           normalizer=args.log_interval)
        # Checkpointing
        if args.save and args.save_interval and iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, lr_scheduler, args, deepspeed=DEEPSPEED_WRAP and args.deepspeed)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            val_loss, val_ppl = evaluate_and_print_results(
                prefix, iter(val_data) if val_data else None, model, args, timers, False)
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                scalars = {'val_loss': val_loss, 'val_perplexity': val_ppl}
                for k, v in scalars.items():
                    tb_writer.add_scalar(k, v, iteration)

        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print('rank: {} | time: {} | exiting the program at iteration {}'.
                  format(rank, time_str, iteration), flush=True)
            exit()

    return iteration, skipped_iters


def evaluate(data_iterator, model, args, timers, verbose=False):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_lm_loss = 0
    eval_len = args.eval_iters or len(data_iterator)

    with torch.no_grad():
        # stop = False
        iteration = 0
        while iteration < eval_len:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration, eval_len))
            # Forward evaluation.
            sample = next(data_iterator) if (data_iterator is not None) else None
            lm_loss = forward_step(sample, model, args, timers)

            '''when contiguous memory optimizations are enabled, the buffers
            allocated by the optimizations are deallocated during backward pass
            in the absence of backward pass the buffers should be reset after each
            forward pass'''
            if DEEPSPEED_WRAP and args.deepspeed and args.deepspeed_activation_checkpointing:
                DEEPSPEED_WRAP.deepspeed.checkpointing.reset()

            # Reduce across processes.
            if isinstance(model, DDP):
                torch.distributed.all_reduce(lm_loss.data)
                lm_loss.data = lm_loss.data / args.world_size

            total_lm_loss += lm_loss.data.detach().float().item()

    # Move model back to the train mode.
    model.train()

    total_lm_loss /= eval_len
    return total_lm_loss


def evaluate_and_print_results(prefix, data_iterator, model,
                               args, timers, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    if args.load_tag:
        prefix = 'checkpoint {}'.format(args.load_tag)
    lm_loss = evaluate(data_iterator, model, args, timers, verbose)
    lm_ppl = math.exp(min(20, lm_loss))
    string = ' validation loss at {} | '.format(prefix)
    string += 'LM loss: {:.4f} | '.format(lm_loss)
    string += 'LM PPL: {:.3f}'.format(lm_ppl)
    length = len(string) + 1
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)

    return lm_loss, lm_ppl


'''
    Optional DeepSpeed Activation Checkpointing features
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.

    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.

    This must be done before all the calls to mpu.model_parallel_cuda_manual_seed
    '''


def set_deepspeed_activation_checkpointing(args):
    DEEPSPEED_WRAP.deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config,
                                                     num_checkpoints=args.num_layers)
    mpu.checkpoint = DEEPSPEED_WRAP.deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = DEEPSPEED_WRAP.deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = DEEPSPEED_WRAP.deepspeed.checkpointing.model_parallel_cuda_manual_seed


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if DEEPSPEED_WRAP and args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def get_train_val_test_data(args):
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""

    (train_data, val_data, test_data) = (None, None, None)

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        (train_data, val_data, test_data), num_tokens, eod_token, tokenizer = make_gpt3_dataloaders(args)
        before = num_tokens
        after = before
        multiple = args.make_vocab_size_divisible_by * mpu.get_model_parallel_world_size()
        while (after % multiple) != 0:
            after += 1
        print_rank_0(
            '> padded vocab (size: {}) with {} dummy tokens (new size: {})'.format(before, after - before, after))
        print_rank_0('> end-of-document token: {}'.format(eod_token))
        token_counts = torch.cuda.LongTensor(
            [after, eod_token, int(args.do_train), int(args.do_valid), int(args.do_test)])
    else:
        tokenizer = None
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    num_tokens = token_counts[0].item()
    eod_token = token_counts[1].item()
    args.do_train = token_counts[2].item()
    args.do_valid = token_counts[3].item()
    args.do_test = token_counts[4].item()

    return train_data, val_data, test_data, num_tokens, eod_token, tokenizer


def generate(model, tokenizer, raw_text, out_seq_length=256, seq_length=512, temperature=1.0, top_k=0, top_p=0.9):
    context_tokens = tokenizer(raw_text)['input_ids']
    context_length = len(context_tokens)
    pad_id = tokenizer.encoder['<pad>']
    if context_length < seq_length:
        context_tokens.extend([pad_id] * (seq_length - context_length))

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor([context_length])

    torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())

    context_length = context_length_tensor[0].item()

    tokens = context_tokens_tensor
    tokens = tokens.view(1, -1).contiguous()
    tokens = tokens.to(torch.cuda.current_device())
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(tokens, pad_id, False, False)

    counter = 0
    start_context_length = context_length

    while counter < (start_context_length + out_seq_length):
        logits = model(tokens, position_ids, attention_mask)
        logits = logits[:, context_length - 1, :] / temperature
        logits = top_k_logits(logits, top_k=top_k, top_p=top_p)
        log_probs = torch.nn.functional.softmax(logits, dim=-1)
        prev = torch.multinomial(log_probs, num_samples=1)
        tokens[0, context_length] = prev[0]
        context_length += 1
        if context_length >= seq_length:
            break
        counter += 1

        output_tokens_list = tokens.view(-1).tolist()
        decode_tokens = tokenizer.decode(output_tokens_list)
        decode_tokens = decode_tokens[:decode_tokens.find("<|endoftext|>")]
        token_end = decode_tokens.find("<|endoftext|>")
        if token_end != -1:
            break

    output_tokens_list = tokens.view(-1).tolist()
    decode_tokens = tokenizer.decode(output_tokens_list)
    return decode_tokens[:decode_tokens.find("<|endoftext|>")]


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    #     if args.load_huggingface:
    #         args.make_vocab_size_divisible_by = 1

    # Pytorch distributed.
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        print('Pretrain GPT3 model')
        print_args(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    train_data, val_data, test_data, args.vocab_size, args.eod_token, tokenizer = get_train_val_test_data(args)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)

    # Resume data loader if necessary.
    if args.resume_dataloader:
        if train_data is not None:
            train_data.batch_sampler.start_iter = args.iteration % len(train_data)
            print_rank_0(f"Resume train set from iteration {train_data.batch_sampler.start_iter}")
        if val_data is not None:
            start_iter_val = (args.train_iters // args.save_interval) * args.eval_interval
            val_data.batch_sampler.start_iter = start_iter_val % len(val_data)
    if train_data is not None:
        train_data_iterator = iter(train_data)
    else:
        train_data_iterator = None

    iteration = 0
    if args.train_iters > 0:
        if args.do_train:
            iteration, skipped = train(model, optimizer,
                                       lr_scheduler,
                                       train_data_iterator,
                                       val_data,
                                       timers,
                                       args,
                                       tokenizer)

        if args.do_valid:
            prefix = 'the end of training for val data'
            # val_loss, val_ppl
            _ = evaluate_and_print_results(prefix, iter(val_data) if val_data else None,
                                           model, args, timers, False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args, deepspeed=DEEPSPEED_WRAP and args.deepspeed)

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, iter(test_data) if test_data else None,
                                   model, args, timers, True)


# test
if __name__ == "__main__":
    main()
