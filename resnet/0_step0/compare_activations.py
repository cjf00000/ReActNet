import torch

for i in range(1, 2):
    a = 'layers/layer_{}.pth.tar'.format(i)
    b = 'layers_expand/layer_{}.pth.tar'.format(i)

    act, ss = torch.load(a)
    act2, ss2 = torch.load(b)

    print(ss.item(), ss2.item(), act.norm().item(), (act-act2).norm().item())
