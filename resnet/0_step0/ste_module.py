import torch
import torch.nn as nn
from matplotlib import pyplot as plt


def multibit_quantize(input, bits):
    # assume input in [0, B]
    bins = 2 ** bits - 1
    input = input.clamp(0, bins)
    input = input.round()
    outputs = []
    for i in range(bits-1, -1, -1):
        print(i)
        scale = 2 ** i
        outputs.append((input >= scale).float())
        input = input - scale * outputs[-1]

    return outputs


def smooth_gradient(y, ws=99):
    smoothed_value = torch.zeros_like(y)
    N = y.shape[0]
    for i in range(ws, N-ws-1):
        smoothed_value[i] = y[i-ws:i+ws+1].mean()

    smoothed_gradient = torch.zeros_like(y)
    for i in range(ws+1, N-ws-2):
        smoothed_gradient[i] = (smoothed_value[i+1] - smoothed_value[i-1]) / 0.02

    return smoothed_value, smoothed_gradient


if __name__ == '__main__':
    x = torch.arange(-100, 800) / 100
    y = x.clamp(0, 7).round()
    smooth_y, smooth_dy = smooth_gradient(y)

    fig, ax = plt.subplots(4, 3, figsize=(9, 12))
    outputs = multibit_quantize(x, 3)
    ax[0, 0].plot(x, outputs[0])
    ax[0, 1].plot(x, outputs[1])
    ax[0, 2].plot(x, outputs[2])
    for b in range(3):
        soutput, s_doutput = smooth_gradient(outputs[b])
        ax[1, b].plot(x, soutput)
        ax[2, b].plot(x, s_doutput)

    ax[3, 0].plot(x, smooth_y)
    ax[3, 1].plot(x, smooth_dy)
    for i in range(4):
        for j in range(3):
            ax[i, j].set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    plt.show()
