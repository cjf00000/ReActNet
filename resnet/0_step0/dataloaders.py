# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import torch
import sys
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from functools import partial
sys.path.append("../../")
from utils.utils import *

# from image_classification.autoaugment import AutoaugmentImageNetPolicy

DATA_BACKEND_CHOICES = ["pytorch", "syntetic"]
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types

    DATA_BACKEND_CHOICES.append("dali-gpu")
    DATA_BACKEND_CHOICES.append("dali-cpu")
except ImportError:
    print(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )


def load_jpeg_from_file(path, cuda=True):
    img_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )

    img = img_transforms(Image.open(path))
    with torch.no_grad():
        # mean and std are not multiplied by 255 as they are in training script
        # torch dataloader reads data into bytes whereas loading directly
        # through PIL creates a tensor with floats in [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if cuda:
            mean = mean.cuda()
            std = std.cuda()
            img = img.cuda()
        img = img.float()

        input = img.unsqueeze(0).sub_(mean).div_(std)

    return input


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        # memory_format=memory_format
    )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array.copy())

    return tensor, targets


def expand(num_classes, dtype, tensor):
    e = torch.zeros(
        tensor.size(0), num_classes, dtype=dtype, device=torch.device("cuda")
    )
    e = e.scatter(1, tensor.unsqueeze(1), 1.0)
    return e


class PrefetchedWrapper(object):
    def prefetched_loader(loader, num_classes, one_hot):
        mean = (
            torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
            .cuda()
            .view(1, 3, 1, 1)
        )
        std = (
            torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])
            .cuda()
            .view(1, 3, 1, 1)
        )

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float()
                if one_hot:
                    next_target = expand(num_classes, torch.float, next_target)

                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, start_epoch, num_classes, one_hot):
        self.dataloader = dataloader
        self.epoch = start_epoch
        self.one_hot = one_hot
        self.num_classes = num_classes

    def __iter__(self):
        if self.dataloader.sampler is not None and isinstance(
            self.dataloader.sampler, torch.utils.data.distributed.DistributedSampler
        ):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(
            self.dataloader, self.num_classes, self.one_hot
        )

    def __len__(self):
        return len(self.dataloader)


def get_pytorch_train_loader(
    data_path,
    image_size,
    batch_size,
    num_classes,
    one_hot,
    interpolation="bilinear",
    augmentation=None,
    start_epoch=0,
    workers=5,
    _worker_init_fn=None,
    memory_format=torch.contiguous_format,
):
    interpolation = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}[
        interpolation
    ]
    traindir = os.path.join(data_path, "train")
    lighting_param = 0.1
    transforms_list = [
        transforms.RandomResizedCrop(image_size, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        Lighting(lighting_param),
    ]
    if augmentation == "autoaugment":
        transforms_list.append(AutoaugmentImageNetPolicy())
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(transforms_list))

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        collate_fn=fast_collate,
        drop_last=True,
        persistent_workers=True,
    )

    return (
        PrefetchedWrapper(train_loader, start_epoch, num_classes, one_hot),
        len(train_loader),
    )


def get_pytorch_val_loader(
    data_path,
    image_size,
    batch_size,
    num_classes,
    one_hot,
    interpolation="bilinear",
    workers=5,
    _worker_init_fn=None,
    crop_padding=32,
    memory_format=torch.contiguous_format,
):
    interpolation = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}[
        interpolation
    ]
    valdir = os.path.join(data_path, "val")
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(
                    image_size + crop_padding, interpolation=interpolation
                ),
                transforms.CenterCrop(image_size),
            ]
        ),
    )

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        collate_fn=fast_collate,
        drop_last=False,
        persistent_workers=True,
    )

    return PrefetchedWrapper(val_loader, 0, num_classes, one_hot), len(val_loader)


class SynteticDataLoader(object):
    def __init__(
        self,
        batch_size,
        num_classes,
        num_channels,
        height,
        width,
        one_hot,
        memory_format=torch.contiguous_format,
    ):
        input_data = (
            torch.randn(batch_size, num_channels, height, width)
            .contiguous(memory_format=memory_format)
            .cuda()
            .normal_(0, 1.0)
        )
        if one_hot:
            input_target = torch.empty(batch_size, num_classes).cuda()
            input_target[:, 0] = 1.0
        else:
            input_target = torch.randint(0, num_classes, (batch_size,))
        input_target = input_target.cuda()

        self.input_data = input_data
        self.input_target = input_target

    def __iter__(self):
        while True:
            yield self.input_data, self.input_target


def get_syntetic_loader(
    data_path,
    image_size,
    batch_size,
    num_classes,
    one_hot,
    interpolation=None,
    augmentation=None,
    start_epoch=0,
    workers=None,
    _worker_init_fn=None,
    memory_format=torch.contiguous_format,
):
    return (
        SynteticDataLoader(
            batch_size,
            num_classes,
            3,
            image_size,
            image_size,
            one_hot,
            memory_format=memory_format,
        ),
        -1,
    )


def get_imagenet_loaders(data_path, batch_size, workers):
    # load training data
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # data augmentation
    crop_scale = 0.08
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=train_transforms)

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True)

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    else:
        val_sampler = None

    # load validation data
    val_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_dataset, train_loader, val_dataset, val_loader


def get_pytorch_train_loader_cifar10(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
    ])
    if num_classes == 10:
        print('Loading CIFAR10')
        train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    else:
        print('Loading CIFAR100')
        train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler,
        collate_fn=fast_collate, drop_last=False)

    # return train_loader, len(train_loader)
    return PrefetchedWrapper(train_loader, num_classes, fp16, one_hot), len(train_loader)


def get_pytorch_val_loader_cifar10(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
    if num_classes == 10:
        val_dataset = datasets.CIFAR10(root=data_path, train=False, download=True)
    else:
        val_dataset = datasets.CIFAR100(root=data_path, train=False, download=True)

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
            collate_fn=fast_collate)

    return PrefetchedWrapper(val_loader, num_classes, fp16, one_hot), len(val_loader)


def get_dataloaders(dataset, data_path, batch_size, one_hot, workers=5, _worker_init_fn=None, fp16=False):
    if dataset == 'cifar10' or dataset == 'cifar100':
        num_classes = 10 if dataset == 'cifar10' else 100
        train_loader, train_iters = get_pytorch_train_loader_cifar10(data_path, batch_size, num_classes, one_hot, workers, _worker_init_fn, fp16)
        val_loader, val_iters = get_pytorch_val_loader_cifar10(data_path, batch_size, num_classes, one_hot, workers, _worker_init_fn, fp16)
        return num_classes, train_loader, train_iters, val_loader, val_iters
    elif dataset == 'imagenet':
        num_classes = 1000
        train_loader, train_iters = get_pytorch_train_loader(data_path, 224, batch_size, num_classes, one_hot, workers=workers)
        val_loader, val_iters = get_pytorch_val_loader(data_path, 224, batch_size, num_classes, one_hot, workers=workers)
        return num_classes, train_loader, train_iters, val_loader, val_iters
    else:
        raise RuntimeError('Unsupported Dataset')
