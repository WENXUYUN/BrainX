# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.backends.cudnn as cudnn
import json

import random
from pathlib import Path
import torch.nn as nn
from timm.models import create_model
from optim_factory import create_optimizer

from datasets import build_pretraining_dataset,build_Val_dataset
from engine_for_pretraining import train_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import modeling_pretrain



def get_args():
    parser = argparse.ArgumentParser('MAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--patience', default=50, type=int)
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch7_642', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--vertex_size', default=1284, type=int,
                        help='surface central vertex size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
                        
    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "lamb"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Dataset parameters
    parser.add_argument('--data_path', default='/data1/dataset/NSD_surface', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')

    parser.add_argument('--output_dir', default='./results/NSD_finetune',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()

def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model

def percentile_cliping(data, lower_percentile = 1,upper_percentile=99):
    lower_cutoff = torch.kthvalue(data.flatten(),int(lower_percentile * len(data.flatten())/100)).values
    upper_cutoff = torch.kthvalue(data.flatten(),int(upper_percentile * len(data.flatten())/100)).values
    clipped_data = torch.clamp(data, lower_cutoff,upper_cutoff)
    return clipped_data


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.vertex_size // patch_size, 1)
    args.patch_size = patch_size

    # get dataset
    dataset_train = build_pretraining_dataset(args)
    #dataset_val = build_Val_dataset(args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        #sampler_val = torch.utils.data.DistributedSampler(
            #dataset_val, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        #)
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        #sampler_val = torch.utils.data.RandomSampler(dataset_val)


    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker
    )

    #data_loader_val = torch.utils.data.DataLoader(
        #dataset_val, sampler=sampler_val,
        #batch_size=args.batch_size,
        #num_workers=args.num_workers,
        #pin_memory=args.pin_mem,
        #drop_last=True,
        #worker_init_fn=utils.seed_worker
    #)


    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = args.batch_size * utils.get_world_size()
    #需要修改 256？
    #args.lr = args.lr * total_batch_size / 7
    #args.lr = 0.002
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp)

    loss_scaler = NativeScaler()
    #args.lr = 0.005
    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    #lr_schedule_values = lr_schedule_values.astype('float32')
    #wd_schedule_values = wd_schedule_values.astype('float32')
    best_loss = np.inf
    epochs_without_improvement = 100
    best_model_weights = None
    #print('Start Training')
    #print(args.start_epoch)
    #print(args.epochs)
    args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        #print('11111111111starttraining')
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size,
            normlize_target=args.normlize_target,
      
        )
        #print('11111111111starttraining')
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        #加入早停
        #model.eval()
        #val_loss = 0.0
        #num_batches = 0
        #loss_func = nn.MSELoss(reduction='mean')
        #loss_func = nn.L1Loss()
        #with torch.no_grad():
           # for step, batch in enumerate(data_loader_val):
                #surfaces, bool_masked_pos = batch
                #surfaces = surfaces.to(device, non_blocking=True)
                #surfaces = surfaces.reshape(-1, 163842, 1)
                #MIN-MAX
                #input_min = surfaces.min(dim=1,keepdim=True)[0]
                #input_max = surfaces.max(dim=1,keepdim=True)[0]
                #surfaces = (surfaces - input_min) / (input_max - input_min)
                
                #surfaces = percentile_cliping(surfaces,1,99)
                #input_min = surfaces.min(dim=1,keepdim=True)[0]
                #input_max = surfaces.max(dim=1,keepdim=True)[0]
               # surfaces = (surfaces - input_min)/(input_max - input_min)
                
                #bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
                #labels = surfaces
                #model = model.to(torch.float32)
               # surfaces = surfaces.to(torch.float32)
                #outputs = model(surfaces, bool_masked_pos)
                #loss = loss_func(input=outputs, target=labels)
                #val_loss += loss.item()
                #num_batches +=1

        # 计算平均损失
        #val_loss /= num_batches
       # print('Epoch: {}, Validation Loss: {:.4f}'.format(epoch + 1, val_loss))

        # 如果验证集上的损失更好，则更新最佳模型参数
        #if val_loss < best_loss:
           # best_loss = val_loss
           # epochs_without_improvement = 0
           # best_model_weights = model.state_dict()
        #else:
           # epochs_without_improvement += 1

        # 如果验证集上的损失连续patience个epoch没有提高，则停止训练
        #if epochs_without_improvement == args.patience:
            #print('Early stopping at epoch {}...'.format(epoch + 1))
            #break

    #model.load_state_dict(best_model_weights)
    #utils.save_model(
        #args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #loss_scaler=loss_scaler, epoch=epoch)
    #log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                 #'epoch': epoch, 'n_parameters': n_parameters}

    if args.output_dir and utils.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
