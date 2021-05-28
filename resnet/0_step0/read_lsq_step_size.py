import torch

for i in range(20):
    data = torch.load('cifar100_fp18_c64_sgd_wd5e-4_la2/checkpoint-{}.pth.tar'.format(i))
    print(data['epoch'])
    state_dict = data['state_dict']
    for k in state_dict:
        if k.find('activation') != -1:
            print(k, state_dict[k])