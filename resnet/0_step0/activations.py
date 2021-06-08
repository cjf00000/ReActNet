import torch
import matplotlib
import numpy as np
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import quant.cpp_extension.calc_quant_bin as ext
from quantize import MultibitLSQNoExpand


num_bins = 256


def compute_histogram(x, num_bins=256):
    x = x.view(-1)
    range = x.abs().max()
    scale = (num_bins - 1) / range
    x = (x * scale).clamp(0, num_bins-1).round()
    hist = torch.zeros(num_bins, device=x.device)
    hist.scatter_add_(dim=0, index=x.to(torch.int64), src=torch.ones_like(x))
    return hist, scale


def get_lsq_step_size(x):
    hist, scale = compute_histogram(x)
    l, r = ext.calc_quant_bin(hist.cpu())
    return (l+r)/scale, (r-l)/scale


num_layers = 2
fig, axs = plt.subplots(num_layers, figsize=(5, 5*num_layers))
for l in range(1, num_layers+1):
    print(l)
    ax = axs[l-1]
    act, weight, bias = torch.load('layers/layer_{}.pth.tar'.format(l))
    act = act.detach()

    hist, scale = compute_histogram(act, num_bins=num_bins)
    ax.bar(np.arange(num_bins), hist.cpu().numpy(), width=1.0)
    l, r = ext.calc_quant_bin(hist.cpu())
    print('l = {}, r = {}, mid = {}'.format(l, r, (l+r)/2))

    ylim = ax.get_ylim()
    ax.plot([l, l], ylim, 'r:')
    ax.plot([r, r], ylim, 'r:')

    lsq = MultibitLSQNoExpand(2)
    quantized = lsq(act)
    ql = ((lsq.quantizers[0].step_size - lsq.quantizers[1].step_size) * scale / 2).item()
    qr = ((lsq.quantizers[0].step_size + lsq.quantizers[1].step_size) * scale / 2).item()
    ax.plot([ql, ql], ylim, 'b:')
    ax.plot([qr, qr], ylim, 'b:')
    print(act.shape, quantized.shape)
    print('Error = ', (act - quantized).norm() ** 2)

    optimizer = torch.optim.SGD(lsq.parameters(), lr=1e-4)
    for iter in range(100):
        quantized = lsq(act)
        loss = (quantized - act).norm() ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Iter {}, loss {}, grad {} {}'.format(iter, loss, lsq.quantizers[0].step_size, lsq.quantizers[1].step_size))

    ql = ((lsq.quantizers[0].step_size - lsq.quantizers[1].step_size) * scale / 2).item()
    qr = ((lsq.quantizers[0].step_size + lsq.quantizers[1].step_size) * scale / 2).item()
    ax.plot([ql, ql], ylim, 'y')
    ax.plot([qr, qr], ylim, 'y')

    with torch.no_grad():
        lsq.quantizers[0].step_size.copy_((l+r)/scale)
        lsq.quantizers[1].step_size.copy_((r-l)/scale)
    quantized = lsq(act)
    print('Error = ', (act - quantized).norm() ** 2)

    # x = torch.arange(num_bins, device=act.device) / scale
    # quantized = lsq(x)
    # for i in range(num_bins):
    #     print('{} --> {}'.format(i, quantized[i]))


    fig.savefig('act.pdf')
    exit(0)

    # act = act.view(-1)
    # mean = act.abs().mean().detach().cpu().numpy()
    # act = act.detach().cpu().numpy()
    # ax.hist(act, bins=100)
    # ylim = ax.get_ylim()
    # ax.plot([-mean, -mean], ylim, 'r')
    # ax.plot([mean, mean], ylim, 'r')
    #
    # ax.plot([-1.5 * mean, -1.5 * mean], ylim, 'r:')
    # ax.plot([-0.5 * mean, -0.5 * mean], ylim, 'r:')
    # ax.plot([0.5 * mean, 0.5 * mean], ylim, 'r:')
    # ax.plot([1.5 * mean, 1.5 * mean], ylim, 'r:')
    # fig.savefig('act.pdf')
