import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from quantize import BinaryActivation


class MultiBitBinaryActivation(nn.Module):
    def __init__(self, bits):
        super(MultiBitBinaryActivation, self).__init__()
        self.bits = bits
        self.binact = BinaryActivation()

    def forward(self, x):
        result = []
        scale = 1.0
        for i in range(self.bits):
            q = self.binact(x)
            result.append(q)
            x = x - scale * q.detach()
            scale /= 2

        return torch.stack(result, 0)


if __name__ == '__main__':
    x = torch.linspace(-3, 3, 1000)
    x.requires_grad_()

    fig, ax = plt.subplots(3, 3)
    q = MultiBitBinaryActivation(3)
    result = q(x)
    total = result[0] + 0.5 * result[1] + 0.25 * result[2]
    # total = result[0]
    grad0 = torch.autograd.grad(result[0], x, torch.ones_like(result[0]), retain_graph=True)[0]
    grad1 = torch.autograd.grad(result[1], x, torch.ones_like(result[1]), retain_graph=True)[0]
    grad2 = torch.autograd.grad(result[2], x, torch.ones_like(result[2]), retain_graph=True)[0]
    total_grad = torch.autograd.grad(total, x, torch.ones_like(total))[0]
    x = x.detach()
    ax[0, 0].plot(x, result[0].detach())
    ax[0, 1].plot(x, result[1].detach())
    ax[0, 2].plot(x, result[2].detach())
    ax[1, 0].plot(x, grad0.detach())
    ax[1, 1].plot(x, grad1.detach())
    ax[1, 2].plot(x, grad2.detach())
    ax[2, 0].plot(x, total.detach())
    ax[2, 2].plot(x, total_grad.detach())

    interval = x[1] - x[0]
    smoothed = (total_grad * interval).cumsum(0) + total[0]
    ax[2, 1].plot(x, smoothed.detach())
    plt.show()
