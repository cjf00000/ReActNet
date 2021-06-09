import torch

data = torch.load('imagenet_r18_c64_adam_sa2/checkpoint-89.pth.tar')
print(data['epoch'])
state_dict = data['state_dict']
for k in state_dict:
    if 'step_size' in k:
        print(k, state_dict[k])