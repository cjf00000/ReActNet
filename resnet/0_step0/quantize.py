import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class binary_activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        clip_mask = torch.logical_and(input >= -1, input <= 1).float()
        grad = 2 * clip_mask * (1 - input.abs())
        return grad_output * grad


class multi_bit_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bits):
        interval = 0.5 ** (bits - 1)
        mask = torch.logical_and(input >= -2 + interval, input <= 2 - interval)
        input = torch.clamp(input, -2 + interval, 2 - interval)
        ctx.save_for_backward(mask)

        # scale input to [0, B-1]
        zero = 2 - interval
        scale = (4 - 2 * interval) / (2 ** bits - 1)
        input = (input - zero) / scale
        input = torch.round(input)
        return input * scale + zero

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask.float(), None


class lsq_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, bits):
        num_bins = 2 ** bits - 1
        bias = -num_bins / 2
        num_features = input.numel() / input.shape[0]
        grad_scale = 1.0 / np.sqrt(num_features * num_bins)

        # Forward
        eps = 1e-7
        scale = scale + eps
        transformed = input / scale - bias
        vbar = torch.clamp(transformed, 0.0, num_bins).round()
        quantized = (vbar + bias) * scale

        # Step size gradient
        error = vbar - transformed
        mask = torch.logical_and(transformed >= 0, transformed <= num_bins)
        case1 = (transformed < 0).float() * bias
        case2 = mask.float() * error
        case3 = (transformed > num_bins).float() * (bias + num_bins)
        ss_gradient = (case1 + case2 + case3) * grad_scale * 100 # TODO hack
        ctx.save_for_backward(mask, ss_gradient)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        mask, ss_gradient = ctx.saved_tensors
        return grad_output * mask.float(), (grad_output * ss_gradient).sum(), None


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        return binary_activation().apply(x)
        # out_forward = torch.sign(x)
        # #out_e1 = (x^2 + 2*x)
        # #out_e2 = (-x^2 + 2*x)
        # out_e_total = 0
        # mask1 = x < -1
        # mask2 = x < 0
        # mask3 = x < 1
        # out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        # out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        # out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        # out = out_forward.detach() - out3.detach() + out3
        #
        # return out


class MultibitActivation(nn.Module):
    def __init__(self, bits):
        super(MultibitActivation, self).__init__()
        self.bits = bits

    def forward(self, x):
        return multi_bit_quantize().apply(x, self.bits)


class LSQ(nn.Module):
    def __init__(self, bits):
        super(LSQ, self).__init__()
        self.bits = bits
        self.step_size = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                num_bins = 2 ** self.bits - 1
                self.step_size.copy_(2 * x.abs().mean() / np.sqrt(num_bins))
                self.initialized = True
                print('Initializing step size to ', self.step_size)

        return lsq_quantize().apply(x, self.step_size, self.bits)


class MultiBitBinaryActivation(nn.Module):  # Extension of binary activation for ablation
    def __init__(self, bits):
        super(MultiBitBinaryActivation, self).__init__()
        self.bits = bits
        self.binact = BinaryActivation()
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x):
        result = 0
        scale = 1.0
        for i in range(self.bits):
            q = self.binact(x)
            x = x - scale * q.detach()
            result = result + scale * q
            scale /= 2

        return result


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        #self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)

    def forward(self, x):
        #real_weights = self.weights.view(self.shape)
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y


class LSQConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, num_bits=1):
        super(LSQConv, self).__init__(in_chn, out_chn, kernel_size, stride, padding)
        self.quantizer = LSQ(num_bits)

    def forward(self, x):
        scaling_factor = self.weight.abs().mean((1, 2, 3)).view(-1, 1, 1, 1)
        quant_weights = self.quantizer(self.weight / scaling_factor) * scaling_factor

        y = F.conv2d(x, quant_weights, stride=self.stride, padding=self.padding)

        return y
