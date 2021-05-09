import sys
import torch
import time
from torchvision import datasets, transforms
from torch.cuda.amp import autocast

sys.path.append("../../")
from utils.utils import *
from birealnet import birealnet18
from utils import log
from dataloaders import get_pytorch_val_loader


ckpt_file = 'models/checkpoint-255.pth.tar'
valdir = '/home/LargeData/Large/ImageNet'
num_classes = 1000
data = torch.load(ckpt_file)
checkpoint = data['state_dict']
state_dict = {}

for k in checkpoint.keys():
    print(k, checkpoint[k].shape)
    state_dict[k.replace('module.', '')] = checkpoint[k]

model = birealnet18()
model = model.cuda()
model.load_state_dict(state_dict)
print('State dict loaded.')


val_loader, _ = get_pytorch_val_loader(valdir, 224, 512, 1000, False, workers=24)

# Validation loop
top1 = log.AverageMeter()
top5 = log.AverageMeter()

model.eval()
start_t = time.time()
num_images = 50000
with torch.no_grad():
    for i, (images, target) in enumerate(val_loader):
        images = images.cuda()
        target = target.cuda()
        n = images.size(0)

        with autocast(enabled=True):
            logits = model(images)
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))

        top1.record(prec1[0].item())
        top5.record(prec5[0].item())
        print(i, top1.get_val())

        torch.cuda.synchronize()

elapsed = time.time() - start_t
print('Top 1 = {}, Top 5 = {}. IPS = {}'.format(top1.get_val(), top5.get_val(),
                                                num_images / elapsed))
