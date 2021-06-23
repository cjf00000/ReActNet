import sys
import torch
import time
from torchvision import datasets, transforms
from torch.cuda.amp import autocast

sys.path.append("../../")
from utils.utils import *
import birealnet
from birealnet import birealnet18
from utils import log
from dataloaders import get_dataloaders
from birealnet import get_model, init_model_from, init_ea_model_from, init_binaryduo_from
from quantize import *

# ckpt_file = 'cifar100_fp18_c64_sgd_wd5e-4_distill_qa/model_best.pth.tar'
# ckpt_file = 'cifar100_64_sa2.pth.tar'
ckpt_file = 'cifar100_fp18_c64_a2_mixup_nowd/checkpoint-1.pth.tar'
valdir = '~/data/'
batch_size = 1000
data = torch.load(ckpt_file)
checkpoint = data['state_dict']
state_dict = {}

for k in checkpoint.keys():
    print(k, checkpoint[k].shape)
    state_dict[k.replace('module.', '')] = checkpoint[k]

num_classes, _, _, val_loader, val_iters = \
        get_dataloaders('cifar100', valdir, batch_size, False, workers=32)

# model = birealnet18()
model = get_model('resnet18', num_classes=num_classes, num_channels=64,
                  qa='fp', qw='a2')
model = model.cuda()
model.load_state_dict(state_dict, strict=False)

# init_binaryduo_from(model, state_dict)

# for name, layer in model.named_modules():
#     if isinstance(layer, BinaryDuo):
#         weight = layer.weight
#         C = weight.shape[1] // 3
#         print(weight.shape, weight[0, 0, 0, 0].item(),
#               weight[0, C, 0, 0].item(), weight[0, 2*C, 0, 0].item())
#
# print(state_dict.keys())
# exit(0)
# # init_model_from(model, state_dict, 2)
# #init_ea_model_from(model, state_dict, 2)
for name, layer in model.named_modules():
    if isinstance(layer, LSQ) or isinstance(layer, MultibitLSQNoScale):
        layer.initialized = True


# Validation loop
top1 = log.AverageMeter()
top5 = log.AverageMeter()

model.eval()
start_t = time.time()
num_images = 50000
with torch.no_grad():
    for i, (images, target) in enumerate(val_loader):
        if i == 0:
            birealnet.debug = True
        else:
            birealnet.debug = False

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
