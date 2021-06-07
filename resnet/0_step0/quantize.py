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


class basic_binary_activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # clip_mask = 0.5 * torch.logical_and(input >= -2, input <= 2).float()
        clip_mask = torch.logical_and(input >= -1, input <= 1).float()
        return grad_output * clip_mask


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


class minmax_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bits):
        mx = input.abs().max()
        num_bins = 2 ** bits - 1
        scale = num_bins / 2 / mx
        input = (input - mx) * scale
        input = torch.round(input)
        return input / scale + mx

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class lsq_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, bits):
        num_bins = 2 ** bits - 1
        bias = -num_bins / 2
        num_features = input.numel()
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
        # TODO gradient scale might be too small, so optimizing without AdaGrad might be problematic...
        ss_gradient = (case1 + case2 + case3) * grad_scale #* 100 * scale
        ctx.save_for_backward(mask, ss_gradient)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        mask, ss_gradient = ctx.saved_tensors
        return grad_output * mask.float(), (grad_output * ss_gradient).sum(), None


# TODO asym
class lsq_quantize_perchannel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, bits):
        scale = scale.view(-1, 1, 1, 1)
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
        ss_gradient = (case1 + case2 + case3) * grad_scale #* 100 * scale
        ctx.save_for_backward(mask, ss_gradient)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        mask, ss_gradient = ctx.saved_tensors
        return grad_output * mask.float(), \
               (grad_output * ss_gradient).sum([1, 2, 3]), None


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        return binary_activation().apply(x)


class BasicBinaryActivation(nn.Module):
    def __init__(self):
        super(BasicBinaryActivation, self).__init__()

    def forward(self, x):
        return basic_binary_activation().apply(x)


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


class LSQPerChannel(nn.Module):
    def __init__(self, num_channels, bits):
        super(LSQPerChannel, self).__init__()
        self.bits = bits
        self.step_size = nn.Parameter(torch.ones(num_channels), requires_grad=True)
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                num_bins = 2 ** self.bits - 1
                self.step_size.copy_(2 * x.abs().mean([1,2,3]) / np.sqrt(num_bins))
                self.initialized = True
                print('Initializing step size to ', self.step_size.mean())

        return lsq_quantize_perchannel().apply(x, self.step_size.abs(), self.bits)


class MultibitLSQ(nn.Module):
    def __init__(self, bits):
        super(MultibitLSQ, self).__init__()
        self.bits = bits
        self.quantizers = nn.ModuleList([LSQ(1) for i in range(bits)])

    def forward(self, x):
        output = []
        scale = 2 ** (self.bits - 1)        # Scale: 4, 2, 1, ...
        for quantizer in self.quantizers:
            quantized = quantizer(x)
            output.append(quantized / scale)
            scale /= 2
            x = x - quantized

        return torch.cat(output, 1)       # N, C*b, H, W


class MultibitLSQNoScale(nn.Module):
    def __init__(self, bits):
        super(MultibitLSQNoScale, self).__init__()
        self.bits = bits
        self.quantizers = nn.ModuleList([LSQ(1) for i in range(bits)])

    def forward(self, x):
        output = []
        for quantizer in self.quantizers:
            quantized = quantizer(x)
            output.append(quantized)
            x = x - quantized

        return torch.cat(output, 1)       # N, C*b, H, W


class MultibitLSQPerChannelInit(nn.Module):     # Used for initializing MultibitLSQConv
    def __init__(self, num_channels, bits):
        super(MultibitLSQPerChannelInit, self).__init__()
        self.bits = bits
        self.quantizers = nn.ModuleList([LSQPerChannel(num_channels, 1) for i in range(bits)])

    def forward(self, x):
        output = []
        scale = 2 ** (self.bits - 1)  # Scale: 4, 2, 1, ...
        for quantizer in self.quantizers:
            quantized = quantizer(x)
            output.append(x / scale)
            scale /= 2
            x = x - quantized

        return output


class Quantize(nn.Module):
    def __init__(self, bits):
        super(Quantize, self).__init__()
        self.bits = bits

    def forward(self, x):
        return minmax_quantize().apply(x, self.bits)


class MultiBitBinaryActivation(nn.Module):  # Extension of binary activation for ablation
    def __init__(self, bits):
        super(MultiBitBinaryActivation, self).__init__()
        self.bits = bits
        self.binact = BasicBinaryActivation()
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
        # self.quantizer = LSQ(num_bits)
        self.quantizer = LSQPerChannel(out_chn, num_bits)

    def forward(self, x):
        # scaling_factor = self.weight.abs().mean((1, 2, 3)).view(-1, 1, 1, 1)
        # quant_weights = self.quantizer(self.weight / scaling_factor) * scaling_factor
        quant_weights = self.quantizer(self.weight)

        y = F.conv2d(x, quant_weights, stride=self.stride, padding=self.padding)

        return y


class QConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, num_bits=1):
        super(QConv, self).__init__(in_chn, out_chn, kernel_size, stride, padding)
        self.quantizer = Quantize(num_bits)

    def forward(self, x):
        scaling_factor = self.weight.abs().mean((1, 2, 3)).view(-1, 1, 1, 1)
        quant_weights = self.quantizer(self.weight / scaling_factor) * scaling_factor

        y = F.conv2d(x, quant_weights, stride=self.stride, padding=self.padding)

        return y


class MultibitLSQConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, num_bits=1, full_matrix=False):
        super(MultibitLSQConv, self).__init__(in_chn * num_bits,
                                              out_chn * (2 * num_bits - 1),
                                              kernel_size,
                                              stride,
                                              padding)
        self.act_quantizer = MultibitLSQ(num_bits)
        self.wt_quantizer = LSQPerChannel(out_chn * (2 * num_bits - 1), 1)
        self.bits = num_bits
        self.Cout = out_chn
        self.weight_mask = torch.zeros_like(self.weight).cuda() # TODO hack
        self.full_matrix = full_matrix

    def init_from(self, weight, step_size):
        with torch.no_grad():
            # weight: [Cout, C, kH, kW]
            # step_size: [Cout]
            Cout, C, _, _ = weight.shape

            # convert integer weight to binary string
            lq = LSQPerChannel(Cout, self.bits).cuda()
            lq.initialized = True
            lq.step_size.copy_(step_size)

            wq = MultibitLSQPerChannelInit(Cout, self.bits).cuda()
            for layer in wq.modules():
                if isinstance(layer, LSQPerChannel):
                    layer.initialized = True
            for b in range(self.bits):
                wq.quantizers[b].step_size.copy_(step_size * 2**(self.bits - 1 - b))
            for b in range(self.bits * 2 - 1):
                self.wt_quantizer.step_size[Cout*b:Cout*(b+1)] = step_size

            weight_groups = wq(weight)
            qweight = lq(weight)
            print('---------')
            print(self.weight_mask.requires_grad)
            # print(weight_groups[0][0,0,0], wq.quantizers[0].step_size[0])
            # print(weight_groups[1][0, 0, 0], wq.quantizers[1].step_size[0])
            # print(weight_groups[2][0, 0, 0], wq.quantizers[2].step_size[0])
            # print('quantized weight diff ', weight.norm(),
            #       (weight - weight_groups[0]*4 - weight_groups[1]*2 - weight_groups[2]).norm(),
            #       (qweight - weight_groups[0]*4 - weight_groups[1]*2 - weight_groups[2]).norm())

            for b_a in range(self.bits):
                for b_w in range(self.bits):
                    b_out = b_a + b_w
                    self.weight[Cout*b_out:Cout*(b_out+1), C*b_a:C*(b_a+1)] = weight_groups[b_w]
                    self.weight_mask[Cout*b_out:Cout*(b_out+1), C*b_a:C*(b_a+1)] = 1.0

            # Rationality check
            self.wt_quantizer.initialized = True
            myweight = self.wt_quantizer(self.weight) * self.weight_mask

            print(self.weight.norm(), (self.weight - myweight).norm())
            for b_a in range(self.bits):
                aw = 0
                for b_w in range(self.bits):
                    b_out = b_a + b_w
                    aw = aw + myweight[Cout*b_out:Cout*(b_out+1), C*b_a:C*(b_a+1)] * 2**(self.bits-1-b_w)

                print(weight.norm(), aw.norm(), (aw - weight).norm(), (aw - qweight).norm())

    def forward(self, x):       # x: [N, C, H, W]
        # quant_input: [N, Cb, H, W]
        quant_input = self.act_quantizer(x)

        # Rationality Check
        C = x.shape[1]
        qinput = 0
        for i in range(self.bits):
            qinput = qinput + quant_input[:, C*i:C*(i+1)] * 2**(self.bits - 1 - i)

        # quant_weight: [Cout*(2b-1), Cb, kH, kW]
        quant_weights = self.wt_quantizer(self.weight)
        if not self.full_matrix:
            quant_weights = quant_weights * self.weight_mask

        # yb: [N, Cout*(2b-1), H, W]
        yb = F.conv2d(quant_input, quant_weights, stride=self.stride, padding=self.padding)
        y = 0   # y: [N, C, H, W]
        for b in range(2 * self.bits - 1):
            scale = 2 ** (2 * self.bits - 2 - b)
            y = y + yb[:, b*self.Cout:(b+1)*self.Cout] * scale

        return y


class MultibitLSQConvAct(nn.Conv2d):        # Only quantize activation
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, num_bits=1):
        super(MultibitLSQConvAct, self).__init__(in_chn * num_bits,
                                              out_chn,
                                              kernel_size,
                                              stride,
                                              padding)
        self.act_quantizer = MultibitLSQNoScale(num_bits)
        self.bits = num_bits
        self.C = in_chn

    def init_from(self, weight, step_size):
        with torch.no_grad():
            for b in range(self.bits):
                self.weight[:, b*self.C:(b+1)*self.C] = weight

    def forward(self, x):       # x: [N, C, H, W]
        # quant_input: [N, Cb, H, W]
        quant_input = self.act_quantizer(x)
        y = F.conv2d(quant_input, self.weight, stride=self.stride, padding=self.padding)
        return y
