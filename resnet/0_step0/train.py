import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import glob
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("../../")
from utils.utils import *
from utils import KD_loss
from utils import log
from utils.log import lr_cosine_policy
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from birealnet import get_model, init_model_from, init_ea_model_from
import torchvision.models as models
from dataloaders import get_dataloaders
from mixup import *

parser = argparse.ArgumentParser("birealnet18")
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=256, help='num of training epochs')
parser.add_argument('--warmup', type=int, default=4, help='num of warmup epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
parser.add_argument('--resume', type=str, default='checkpoint.pth.tar', help='path checkpoint')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset. Cifar10, Cifar100, or ImageNet')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--teacher', type=str, default='none', help='Teacher Architecture')
parser.add_argument('--arch', type=str, default='resnet18', help='Student Architecture')
parser.add_argument('--channel', type=int, default=64, help='Number of channels')
parser.add_argument('--mixup', type=float, default=0.0, help='Mixup')
parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument("--amp", action="store_true", help="Run model AMP (automatic mixed precision) mode.",)
parser.add_argument("--qa", type=str, default='fp', help="fp, b, q1, q2, q3, m1, m2, m3, l1, l2, l3...",)
parser.add_argument("--qw", type=str, default='fp', help="fp, b, l1, l2, l3...",)
parser.add_argument("--project", type=str, default='bnn-imagenet', help="wandb project",)

args = parser.parse_args()

CLASSES = 1000


def cleanup():
    dist.destroy_process_group()


def main():
    if not torch.cuda.is_available():
        sys.exit(1)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled = True

    # Build dataloaders
    num_classes, train_loader, train_iters, val_loader, val_iters = \
        get_dataloaders(args.dataset, args.data, args.batch_size, args.mixup, workers=args.workers)

    # Setup logger
    logger = get_logger(args, train_iters, val_iters)

    # Build model
    model_student = get_model(args.arch, qa=args.qa, qw=args.qw,
                                         num_classes=num_classes, num_channels=args.channel)

    model_student = model_student.cuda()
    if args.distributed:
        model_student = DDP(model_student, device_ids=[args.gpu], find_unused_parameters=False)

    print(model_student)
    # TODO hack
    # for name, param in model_student.named_parameters():
    #     if name.find('step_size') == -1:
    #         param.requires_grad = False

    # load model
    criterion_train, criterion_val, model_teacher = \
        get_criterion(args.dataset, args.teacher, num_classes, args.gpu, args.mixup, args.label_smoothing)

    # Build optimizer
    all_parameters = model_student.parameters()
    weight_parameters = []
    step_parameters = []
    other_parameters = []
    for pname, p in model_student.named_parameters():
        if p.ndimension() == 4 or 'conv' in pname:
            weight_parameters.append(p)
        elif 'step_size' in pname:
            step_parameters.append(p)
        else:
            other_parameters.append(p)

    # optimizer = torch.optim.SGD(
    #     [{'params': other_parameters},
    #      {'params': weight_parameters, 'weight_decay': args.weight_decay},
    #      {'params': step_parameters, 'momentum': 0.0}],
    #     lr=args.learning_rate, momentum=args.momentum)
    optimizer = torch.optim.Adam(
        [{'params': other_parameters},
         {'params': weight_parameters, 'weight_decay': args.weight_decay},
         {'params': step_parameters}],
        lr=args.learning_rate, betas=(args.momentum, 0.999))

    scheduler = lr_cosine_policy(args.learning_rate, args.warmup, args.epochs, logger=logger)

    # Resume
    start_epoch = 0
    best_top1_acc = 0

    checkpoint_tar = args.resume
    if os.path.exists(checkpoint_tar):
        print('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar, map_location=lambda storage, loc: storage.cuda(args.gpu))
        # start_epoch = checkpoint['epoch']
        best_top1_acc = checkpoint['best_top1_acc']
        state_dict = checkpoint['state_dict']
        if not args.distributed:
            state_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict}
        else:
            find_module = False
            for k in state_dict:
                if 'module' in k:
                    find_module = True
            if not find_module:
                state_dict = {'module.'+k: state_dict[k] for k in state_dict}
        for k in state_dict:
            print(k)

        # TODO hack
        # init_ea_model_from(model_student, state_dict, 2)
        if args.qw[0] != 'm' and args.qw[0] != 'a':
            model_student.load_state_dict(state_dict, strict=False)
        else:
            initialized = True
            if args.qw[0] == 'a':
                initialized = False
            init_model_from(model_student, state_dict, int(args.qw[1:]), initialized=initialized)

        print("loaded checkpoint {} epoch = {}".format(checkpoint_tar, checkpoint['epoch']))

    # train the model
    epoch_iter = range(start_epoch, args.epochs)
    if logger is not None:
        epoch_iter = logger.epoch_generator_wrapper(epoch_iter)

    valid_top1_acc = validate(0, val_loader, model_student, criterion_val, args, logger)
    print('Val acc', valid_top1_acc)
    for epoch in epoch_iter:
        train(epoch,  train_loader, model_student, model_teacher, criterion_train, optimizer, scheduler, logger)
        # print('--------- Step size -------------')
        # for name, param in model_student.named_parameters():
        #     if name.find('step_size') != -1:
        #         print(name, param.mean().detach().cpu().numpy(),
        #                     param.min().detach().cpu().numpy(),
        #                     param.max().detach().cpu().numpy(),
        #                     param.grad.mean().detach().cpu().numpy())
        valid_top1_acc = validate(epoch, val_loader, model_student, criterion_val, args, logger)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model_student.state_dict(),
                'best_top1_acc': best_top1_acc,
                'optimizer': optimizer.state_dict(),
                }, is_best, args.save)

    training_time = (time.time() - start_t) / 3600
    print('total training time = {} hours'.format(training_time))

    if args.distributed:
        dist.destroy_process_group()


def train(epoch, train_loader, model_student, model_teacher, criterion, optimizer, scheduler, logger):
    if logger is not None:
        logger.register_metric('train.top1', log.AverageMeter(), log_level=0)
        logger.register_metric('train.top5', log.AverageMeter(), log_level=0)
        logger.register_metric('train.loss', log.AverageMeter(), log_level=0)
        logger.register_metric('train.data_time', log.AverageMeter(), log_level=0)
        logger.register_metric('train.time', log.AverageMeter(), log_level=0)
        logger.register_metric('train.ips', log.AverageMeter(), log_level=0)
        for pname, p in model_student.named_parameters():
            if 'step_size' in pname:
                logger.register_metric('step_size/' + pname.replace('.step_size', ''),
                                       log.AverageMeter(), log_level=1)
                logger.register_metric('step_size/' + pname.replace('.step_size', '') + '.grad',
                                       log.AverageMeter(), log_level=1)

    model_student.train()
    if model_teacher:
        model_teacher.eval()
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter)

    scaler = torch.cuda.amp.GradScaler(
        init_scale=128,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=1000000000,
        enabled=args.amp,
    )

    for i, (images, target) in data_iter:
        # if i > 3000:
        #     break
        data_time = time.time() - end
        images = images.cuda()
        target = target.cuda()
        scheduler(optimizer, i, epoch)

        # compute outputy
        with autocast(enabled=args.amp):
            logits_student = model_student(images)
            if model_teacher:
                logits_teacher = model_teacher(images)
                loss = criterion(logits_student, logits_teacher)
            else:
                loss = criterion(logits_student, target)
            if torch.distributed.is_initialized():
                reduced_loss = reduce_tensor(loss.data)
            else:
                reduced_loss = loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        n = images.size(0)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        # measure elapsed time
        if logger is not None:
            batch_time = time.time() - end
            logger.log_metric('train.loss', reduced_loss.item(), n)
            logger.log_metric('train.top1', prec1.item(), n)
            logger.log_metric('train.top5', prec5.item(), n)
            logger.log_metric('train.time', batch_time)
            logger.log_metric('train.data_time', data_time)
            logger.log_metric('train.ips', calc_ips(n, batch_time))
            for pname, p in model_student.named_parameters():
                if 'step_size' in pname:
                    logger.log_metric('step_size/' + pname.replace('.step_size', ''),
                                           p.abs().mean().item())
                    logger.log_metric('step_size/' + pname.replace('.step_size', '') + '.grad',
                                      p.grad.abs().mean().item())

        end = time.time()


def validate(epoch, val_loader, model, criterion, args, logger):
    top1 = log.AverageMeter()

    if logger is not None:
        logger.register_metric('val.top1', log.AverageMeter(), log_level=0)
        logger.register_metric('val.top5', log.AverageMeter(), log_level=0)
        logger.register_metric('val.loss', log.AverageMeter(), log_level=0)
        logger.register_metric('val.time', log.AverageMeter(), log_level=1)
        logger.register_metric('val.ips', log.AverageMeter(), log_level=1)

    data_iter = enumerate(val_loader)
    if not logger is None:
        data_iter = logger.iteration_generator_wrapper(data_iter, val=True)

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in data_iter:
            images = images.cuda()
            target = target.cuda()

            # compute output
            with autocast(enabled=args.amp):
                logits = model(images)
                loss = criterion(logits, target)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(logits, target, topk=(1, 5))
                if torch.distributed.is_initialized():
                    reduced_loss = reduce_tensor(loss.data)
                    prec1 = reduce_tensor(prec1[0])
                    prec5 = reduce_tensor(prec5[0])
                else:
                    reduced_loss = loss
                    prec1 = prec1[0]
                    prec5 = prec5[0]

            n = images.size(0)

            # measure elapsed time
            if logger is not None:
                batch_time = time.time() - end
                logger.log_metric('val.loss', reduced_loss.item(), n)
                logger.log_metric('val.top1', prec1.item(), n)
                logger.log_metric('val.top5', prec5.item(), n)
                logger.log_metric('val.time', batch_time)
                logger.log_metric('val.ips', calc_ips(n, batch_time))
                top1.record(prec1.item(), n)

            end = time.time()
            torch.cuda.synchronize()

    return top1.get_val()


def get_logger(args, train_iters, val_iters):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger_backends = [
                    log.JsonBackend(os.path.join(args.save, 'raport.json'), log_level=1),
                    log.StdOut1LBackend(train_iters, val_iters, args.epochs, log_level=0),
                ]
        try:
            import wandb
            run = wandb.init(project=args.project, entity="jianfeic", config=args, name=args.save)
            code = wandb.Artifact('project-source', type='code')
            for path in glob.glob('*.py', recursive=True):
                code.add_file(path)
            run.log_artifact(code)
            logger_backends.append(log.WandbBackend(wandb))
            print('Logging to wandb...')
        except ImportError:
            print('Wandb not found, logging to stdout and json...')

        logger = log.Logger(args.print_freq, logger_backends)

        for k, v in args.__dict__.items():
            logger.log_run_tag(k, v)
    else:
        logger = None

    return logger


def get_criterion(dataset, teacher, num_classes, gpu, mixup, label_smoothing):
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    criterion_kd = None
    if mixup > 0.0:
        criterion_kd = NLLMultiLabelSmooth(label_smoothing)
    elif label_smoothing > 0.0:
        criterion_kd = LabelSmoothing(label_smoothing)
    if criterion_kd:
        criterion_kd = criterion_kd.cuda()
        return criterion_kd, criterion, None

    # criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    # criterion_smooth = criterion_smooth.cuda()
    if teacher == 'none':
        return criterion, criterion, None

    if dataset == 'imagenet':
        # load model
        model_teacher = models.__dict__[teacher](pretrained=True)
        model_teacher = model_teacher.cuda()
        model_teacher = DDP(model_teacher, device_ids=[gpu])
        for p in model_teacher.parameters():
            p.requires_grad = False
        model_teacher.eval()

        criterion_kd = KD_loss.DistributionLoss()
        return criterion_kd, criterion, model_teacher
    else:
        model_teacher = get_model(teacher,
                                  num_classes=num_classes,
                                  num_channels=96)
        model_teacher = model_teacher.cuda()
        if args.distributed:
            model_teacher = DDP(model_teacher, device_ids=[gpu])

        checkpoint_tar = 'teacher.pth.tar'
        print('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar, map_location=lambda storage, loc: storage.cuda(args.gpu))
        model_teacher.load_state_dict(checkpoint['state_dict'], strict=False)
        print("loaded checkpoint {}".format(checkpoint_tar))

        for p in model_teacher.parameters():
            p.requires_grad = False
        model_teacher.eval()

        criterion_kd = KD_loss.DistributionLoss()
        return criterion_kd, criterion, model_teacher


if __name__ == '__main__':
    main()
