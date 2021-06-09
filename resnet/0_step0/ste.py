import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from quantize import MultibitLSQNoExpand

if __name__ == '__main__':
    x = torch.linspace(-2, 2, 1000)
    x.requires_grad_()

    fig, ax = plt.subplots(3, 3)
    q = MultibitLSQNoExpand(2)
    with torch.no_grad():
        q.quantizers[0].step_size.copy_(torch.tensor(2.0))
        q.quantizers[1].step_size.copy_(torch.tensor(1.2))
        q.quantizers[0].initialized = True
        q.quantizers[1].initialized = True

    y = q(x)
    wt = torch.ones_like(y)
    loss = (y * wt).sum()
    loss.backward()

    dy_dx = x.grad
    ax[0, 0].plot(x.detach(), y.detach())
    ax[0, 1].plot(x.detach(), dy_dx.detach())

    q0 = []
    q1 = []
    for i in range(1000):
        x_batch = x[i:i+1]
        y = q(x_batch)
        q.quantizers[0].step_size.grad = None
        q.quantizers[1].step_size.grad = None
        y.sum().backward()
        q0.append(q.quantizers[0].step_size.grad.item())
        q1.append(q.quantizers[1].step_size.grad.item())

    ax[1, 0].plot(x.detach(), q0)
    ax[1, 1].plot(x.detach(), q1)

    fig.savefig('ste.pdf')
