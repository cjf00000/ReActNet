import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

num_layers = 16
fig, axs = plt.subplots(num_layers, figsize=(5, 80))
for l in range(1, num_layers+1):
    print(l)
    ax = axs[l-1]
    act, weight, bias = torch.load('layers/layer_{}.pth.tar'.format(l))
    act = act.view(-1)
    mean = act.abs().mean().detach().cpu().numpy()
    act = act.detach().cpu().numpy()
    ax.hist(act, bins=100)
    ylim = ax.get_ylim()
    ax.plot([-mean, -mean], ylim, 'r')
    ax.plot([mean, mean], ylim, 'r')

    ax.plot([-1.5 * mean, -1.5 * mean], ylim, 'r:')
    ax.plot([-0.5 * mean, -0.5 * mean], ylim, 'r:')
    ax.plot([0.5 * mean, 0.5 * mean], ylim, 'r:')
    ax.plot([1.5 * mean, 1.5 * mean], ylim, 'r:')
    fig.savefig('act.pdf')
