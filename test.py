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
    from model import load_model, get_mean_em
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoTokenizer
    from util import Logger, check_distributed, strtime, count_parameters

    transformers.logging.set_verbosity_error()

    rank, local_rank, world_size = check_distributed()
    is_main_process = local_rank in [-1, 0]
    is_distributed = world_size != -1

    logger = Logger(on=is_main_process)
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

    tokenizer = AutoTokenizer.from_pretrained('t5-base') # TODO: change

    collate_fn = lambda samples: tensorize(samples, args.num_contexts,
                                           tokenizer, args.max_length,
                                           shuffle=False)

    dataset = FiDDataset(args.data)
    logger.log(f'Evaluating on {len(dataset)} examples')
    sampler = DistributedSampler(dataset, num_replicas=world_size,
                                 rank=rank, shuffle=False) \
                                 if is_distributed else None
    loader = DataLoader(dataset, args.batch_size, shuffle=False,
                        sampler=sampler, num_workers=args.num_workers,
                        collate_fn=collate_fn)

    model, saved_args = load_model(args.model, device)
    logger.log(f'{count_parameters(model)} parameters')
    if saved_args is not None:
        if saved_args.max_length != args.max_length:
            logger.log(f'Warning: diff max length, {saved_args.max_length}'
                       f' in training, {args.max_length} given')
        if saved_args.num_contexts != args.num_contexts:
            logger.log(f'Warning: diff num contexts, {saved_args.num_contexts}'
                       f' in training, {args.num_contexts} given')

    if is_distributed:
        logger.log('DDP wrapping')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    else:
        logger.log('Single-process single-device, no model wrapping')

    start_time = datetime.now()
    mean_em, answers = get_mean_em(model, loader, tokenizer, rank, world_size,
                                   device)
    num_correct = int(sum([score for _, _, score in answers.values()]))
    logger.log(f'\nDone | EM: {mean_em:3.2f} ({num_correct} / {len(dataset)}) |'
               f' total time {strtime(start_time)}')

    if args.pred and is_main_process:
        with open(args.pred, 'w') as f:
            answers = sorted(answers.items(), key=lambda x: x[0])
            for i, (pred, golds, score) in answers:
                f.write(f'{i}\t{pred}\t{golds}\t{float(score)}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument('--num_contexts', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpus', default='', type=str)
    parser.add_argument('--pred', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
