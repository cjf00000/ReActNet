import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from quantize import MultibitActivation, LSQ


if __name__ == '__main__':
    x = torch.linspace(-3, 3, 100)
    x.requires_grad_()

    q = MultibitActivation(2)
    lsq = LSQ(2)

    grad_weight = torch.ones_like(x)

    fig, ax = plt.subplots(3, 2, figsize=(10, 15))

    # Quantize
    y0 = q(x)
    loss = (y0 * grad_weight).sum()
    loss.backward()
    g0 = x.grad.clone()
    x.grad = None
    ax[0, 0].plot(x.detach(), y0.detach())
    ax[0, 1].plot(x.detach(), g0.detach())

    # LSQ
    y = lsq(x)
    loss = (y * grad_weight).sum()
    loss.backward()
    g = x.grad.clone()
    ax[1, 0].plot(x.detach(), y.detach())
    ax[1, 1].plot(x.detach(), g.detach())

    gs = torch.zeros_like(x)
    for i in range(100):
        x1 = x[i:i+1]
        x1.requires_grad_()
        x1.grad = None
        lsq.step_size.grad = None
        y1 = lsq(x1)
        y1.backward()
        gs[i] = lsq.step_size.grad

    ax[2, 0].plot(x.detach(), gs)

    # Step size gradient
    fig.savefig('lsq.pdf')
