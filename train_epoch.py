import argparse
import os


def main(args):
    import random
    import torch
    import torch.distributed as dist
    import transformers

    from copy import deepcopy
    from data import FiDDataset, tensorize
    from datetime import datetime
    from file_handling import mkdir_optional
    from model import FiDT5, get_mean_em
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoTokenizer, set_seed, \
        get_linear_schedule_with_warmup
    from tqdm import tqdm
    from util import Logger, check_distributed, strtime, count_parameters

    transformers.logging.set_verbosity_error()

    set_seed(args.seed)
    rank, local_rank, world_size = check_distributed()
    is_main_process = local_rank in [-1, 0]
    is_distributed = world_size != -1

    mkdir_optional(os.path.dirname(args.model))
    logger = Logger(log_path=args.model + '.log', on=is_main_process)
    logger.log(str(args))
    logger.log(f'rank {rank} local_rank {local_rank} world_size {world_size}',
               force=True)

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group('nccl')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.log(f'Using device: {str(device)}', force=True)

    tokenizer = AutoTokenizer.from_pretrained(args.t5_name)

    dataset_train = FiDDataset(args.data_train)
    sampler_train = DistributedSampler(dataset_train, num_replicas=world_size,
                                       rank=rank, shuffle=True,
                                       seed=args.seed) \
                                       if is_distributed else None
    collate_fn = lambda samples: tensorize(samples, args.num_contexts,
                                           tokenizer, args.max_length,
                                           shuffle=not args.no_shuffle)
    loader_train = DataLoader(dataset_train, args.batch_size,
                              shuffle=(sampler_train is None) and \
                              (not args.no_shuffle),
                              sampler=sampler_train,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    num_training_steps = len(loader_train) * args.epochs

    dataset_val = FiDDataset(args.data_val)
    sampler_val = DistributedSampler(dataset_val, num_replicas=world_size,
                                     rank=rank, shuffle=False) \
                                     if is_distributed else None
    loader_val = DataLoader(dataset_val, args.batch_size_val, shuffle=False,
                            sampler=sampler_val, num_workers=args.num_workers,
                            collate_fn=collate_fn)

    model = FiDT5(args.t5_name, dropout=args.dropout).to(device)
    model.set_checkpoint(args.use_checkpoint)
    logger.log(f'{count_parameters(model)} parameters')
    logger.log(f'Checkpointing: {args.use_checkpoint}')

    if is_distributed:
        logger.log('DDP wrapping')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    else:
        logger.log('Single-process single-device, no model wrapping')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps)

    # Training
    num_batches_processed = 0
    step = 0
    best_dev_em = 0.
    sd_best = None
    start_time = datetime.now()
    for epoch in range(args.epochs):
        model.train()
        start_time_epoch = datetime.now()
        loss_sum = 0.
        if is_distributed:
            loader_train.sampler.set_epoch(epoch)

        for i, batch in enumerate(tqdm(loader_train, disable=not args.tqdm)):
            I, T, P, P_mask = [tensor.to(device) for tensor in batch]

            # loss, logits, past_key_values, encoder_last_hidden_state
            model_out = model(P, P_mask, labels=T)

            loss = model_out.loss  # default CrossEntropyLoss (mean reduction)
            loss.backward()

            num_batches_processed += 1

            if num_batches_processed % args.grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                step += 1

            if is_distributed:
                dist.reduce(loss, 0, op=dist.ReduceOp.SUM)
                if is_main_process:
                    loss /= world_size
            loss_sum += loss.item()

        loss_avg = loss_sum / len(loader_train)

        log = f'Epoch {epoch:3d} | '
        log += f'step {step:5d} / {num_training_steps:5d} | '
        log += f'time {strtime(start_time_epoch)} | '
        log += f'lr: {scheduler.get_last_lr()[0]:.5f} | '
        log += f'loss: {loss_sum / len(loader_train)} | '

        start_time_val = datetime.now()
        dev_em, answers = get_mean_em(model, loader_val, tokenizer,
                                      rank, world_size, device,
                                      disable_tqdm=True)
        num_correct = int(sum([score for _, _, score in answers.values()]))
        log += f'val EM: {dev_em:10.2f} '
        log += f'({num_correct} / {len(dataset_val)}, strtime(start_time_val)})'
        if dev_em > best_dev_em:
            sd = model.module.state_dict() if is_distributed else \
                 model.state_dict()
            sd_best = deepcopy(sd)
            best_dev_em = dev_em
            log += f' <-------------'

        logger.log(log)

    if is_main_process and sd_best is not None:
        logger.log(f'\nDone training | total time {strtime(start_time)} | '
                   f'saving best model to {args.model}')
        torch.save({'sd': sd_best, 'args': args}, args.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('data_train', type=str)
    parser.add_argument('data_val', type=str)
    parser.add_argument('--num_contexts', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batch_size_val', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_warmup_steps', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip', type=float, default=1.)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--t5_name', type=str, default='t5-base')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--tqdm', action='store_true')
    parser.add_argument('--gpus', default='', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
