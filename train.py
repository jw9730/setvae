import os
import random
import datetime
from pathlib import Path
from copy import deepcopy

import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import deepspeed

from args import get_args
from models.networks import SetVAE
from utils import AverageValueMeter, set_random_seed, save, resume, validate_sample, validate_reconstruct, \
    validate_reconstruct_l2
from datasets import get_datasets
from engine import train_one_epoch, validate, visualize

import requests
import json

# torch.autograd.set_detect_anomaly(True)

username = "VLLAB" if not "NAME" in os.environ.keys() else os.environ["NAME"]


def send_slack(msg):
    if 'SLACK' in os.environ.keys():
        webhook_url = os.environ['SLACK']
    else:
        return
    dump = {
        "username": username,
        "channel": "cvpr2021",
        "icon_emoji": ":clapper:",
        "text": msg
    }
    requests.post(webhook_url, json.dumps(dump))


def main_worker(save_dir, args):
    # basic setup
    cudnn.benchmark = True

    if args.log_name is not None:
        log_dir = "runs/%s" % args.log_name
    else:
        log_dir = f"runs/{datetime.datetime.now().strftime('%m-%d-%H-%M-%S')}"

    if args.local_rank == 0:
        logger = SummaryWriter(log_dir)
    else:
        logger = None

    deepspeed.init_distributed(dist_backend='nccl')
    torch.cuda.set_device(args.local_rank)

    model = SetVAE(args)
    parameters = model.parameters()

    n_parameters = sum(p.numel() for p in parameters if p.requires_grad)
    print(f'number of params: {n_parameters}')
    try:
        n_gen_parameters = sum(p.numel() for p in model.init_set.parameters() if p.requires_grad) + \
                           sum(p.numel() for p in model.pre_decoder.parameters() if p.requires_grad) + \
                           sum(p.numel() for p in model.decoder.parameters() if p.requires_grad) + \
                           sum(p.numel() for p in model.post_decoder.parameters() if p.requires_grad) + \
                           sum(p.numel() for p in model.output.parameters() if p.requires_grad)
        print(f'number of generator params: {n_gen_parameters}')
    except AttributeError:
        pass

    optimizer, criterion = model.make_optimizer(args)

    # initialize datasets and loaders
    train_dataset, val_dataset, train_loader, val_loader = get_datasets(args)

    # initialize the learning rate scheduler
    if args.scheduler == 'exponential':
        assert not (args.warmup_epochs > 0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.exp_decay)
    elif args.scheduler == 'step':
        assert not (args.warmup_epochs > 0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 2, gamma=0.1)
    elif args.scheduler == 'linear':
        def lambda_rule(ep):
            lr_w = min(1., ep / args.warmup_epochs) if (args.warmup_epochs > 0) else 1.
            lr_l = 1.0 - max(0, ep - 0.5 * args.epochs) / float(0.5 * args.epochs)
            return lr_l * lr_w

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.scheduler == 'cosine':
        assert not (args.warmup_epochs > 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        # Fake SCHEDULER
        def lambda_rule(ep):
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    # extract collate_fn
    if args.distributed:
        collate_fn = deepcopy(train_loader.collate_fn)
        model, optimizer, train_loader, scheduler = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=parameters,
            training_data=train_dataset,
            collate_fn=collate_fn,
            lr_scheduler=scheduler
        )

    # resume checkpoints
    start_epoch = 0
    if args.resume_checkpoint is None and Path(Path(save_dir) / 'checkpoint-latest.pt').exists():
        args.resume_checkpoint = os.path.join(save_dir, 'checkpoint-latest.pt')  # use the latest checkpoint
        print('Resumed from: ' + args.resume_checkpoint)
    if args.resume_checkpoint is not None:
        if args.distributed:
            if args.resume_optimizer:
                model.module, model.optimizer, model.lr_scheduler, start_epoch = resume(
                    args.resume_checkpoint, model.module, model.optimizer, scheduler=model.lr_scheduler,
                    strict=(not args.resume_non_strict))
            else:
                model.module, _, _, start_epoch = resume(
                    args.resume_checkpoint, model.module, optimizer=None, strict=(not args.resume_non_strict))
        else:
            if args.resume_optimizer:
                model, optimizer, scheduler, start_epoch = resume(
                    args.resume_checkpoint, model, optimizer, scheduler=scheduler, strict=(not args.resume_non_strict))
            else:
                model, _, _, start_epoch = resume(
                    args.resume_checkpoint, model, optimizer=None, strict=(not args.resume_non_strict))

    # save dataset statistics
    if args.local_rank == 0:
        train_dataset.save_statistics(save_dir)
        val_dataset.save_statistics(save_dir)

    # main training loop
    avg_meters = {
        'kl_avg_meter': AverageValueMeter(),
        'l2_avg_meter': AverageValueMeter()
    }

    assert args.distributed

    epoch = start_epoch
    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs):
        if args.local_rank == 0:
            # evaluate on the validation set
            if epoch % args.val_freq == 0 and epoch != 0:
                model.eval()
                with torch.no_grad():
                    val_res = validate(model.module, args, val_loader, epoch, logger, save_dir)
                    for k, v in val_res.items():
                        v = v.cpu().detach().item()
                        send_slack(f'{k}:{v}, Epoch {epoch - 1}')
                        if logger is not None and v is not None:
                            logger.add_scalar(f'val_sample/{k}', v, epoch - 1)

        # train for one epoch
        train_one_epoch(epoch, model, criterion, optimizer, args, train_loader, avg_meters, logger)

        # Only on HEAD process
        if args.local_rank == 0:
            # save checkpoints
            if (epoch + 1) % args.save_freq == 0:
                if args.eval:
                    validate_reconstruct_l2(epoch, val_loader, model, criterion, args, logger)
                save(model.module, model.optimizer, model.lr_scheduler, epoch + 1,
                     Path(save_dir) / f'checkpoint-{epoch}.pt')
                save(model.module, model.optimizer, model.lr_scheduler, epoch + 1,
                     Path(save_dir) / 'checkpoint-latest.pt')

            # save visualizations
            if (epoch + 1) % args.viz_freq == 0:
                with torch.no_grad():
                    visualize(model.module, args, val_loader, epoch, logger)

        # adjust the learning rate
        model.lr_scheduler.step()
        if logger is not None and args.local_rank == 0:
            logger.add_scalar('train lr', model.lr_scheduler.get_last_lr()[0], epoch)

    model.eval()
    if args.local_rank == 0:
        with torch.no_grad():
            val_res = validate(model.module, args, val_loader, epoch, logger, save_dir)
            for k, v in val_res.items():
                v = v.cpu().detach().item()
                send_slack(f'{k}:{v}, Epoch {epoch}')
                if logger is not None and v is not None:
                    logger.add_scalar(f'val_sample/{k}', v, epoch)

    if logger is not None:
        logger.flush()
        logger.close()


def main():
    args = get_args()
    save_dir = Path("checkpoints") / args.log_name
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(exist_ok=True, parents=True)

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    if args.local_rank == 0:
        send_slack(f'{args.log_name} started')
        print("Arguments:")
        print(args)

    main_worker(save_dir, args)


if __name__ == '__main__':
    main()
