import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import quant.cpp_extension.calc_quant_bin as ext


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


class binary_lsq(torch.autograd.Function):
    multiplier = 1      # Larger multiplier -> smaller bias, larger variance

    @staticmethod
    def forward(ctx, input, scale):
        num_features = input.numel()
        grad_scale = 1.0 / np.sqrt(num_features)

        # Forward
        eps = 1e-7
        scale = scale + eps
        sgn = torch.sign(input)
        quantized = sgn * scale / 2

        # Step size gradient
        cutoff = scale / (2 * binary_lsq.multiplier)
        mask = torch.logical_and(input >= -cutoff, input <= cutoff)
        ss_gradient = sgn * (0.5 * grad_scale) * 10     # TODO Hack (larger learning rate)
        ctx.save_for_backward(mask, ss_gradient)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        mask, ss_gradient = ctx.saved_tensors
        return grad_output * mask.float() * binary_lsq.multiplier, \
               (grad_output * ss_gradient).sum()


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
        self.step_size = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                num_bins = 2 ** self.bits - 1
                self.step_size.copy_(2 * x.abs().mean() / np.sqrt(num_bins))
                self.initialized = True
                print('Initializing step size to ', self.step_size)

        return binary_lsq().apply(x, self.step_size)
        # return lsq_quantize().apply(x, self.step_size, self.bits)


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


# class MultibitLSQNoScale(nn.Module):
#     def __init__(self, bits):
#         super(MultibitLSQNoScale, self).__init__()
#         self.bits = bits
#         # self.quantizers = nn.ModuleList([LSQ(1) for i in range(bits)])
#         self.step_size = nn.Parameter(torch.tensor(1.0), requires_grad=True)
#         # self.step_size = torch.tensor(1.0)  # TODO hack
#         self.initialized = False
#
#     def forward(self, x):
#         if not self.initialized:
#             with torch.no_grad():
#                 num_bins = 2 ** self.bits - 1
#                 self.step_size.copy_(2 * x.abs().mean() / np.sqrt(num_bins))
#                 self.initialized = True
#                 print('Initializing step size to ', self.step_size)
#
#         output = []
#         #for quantizer in self.quantizers:
#         #quantized = quantizer(x)
#         for b in range(self.bits):
#             scale = 2 ** (self.bits - 1 - b)
#             quantized = binary_lsq().apply(x, self.step_size * scale)
#             print(quantized.view(-1)[:100])
#             output.append(quantized)
#             x = x - quantized
#
#         exit(0)
#         return torch.cat(output, 1)       # N, C*b, H, W


class MultibitLSQNoScale(nn.Module):
    def __init__(self, bits):
        super(MultibitLSQNoScale, self).__init__()
        self.bits = bits
        # self.quantizers = nn.ModuleList([LSQ(1) for i in range(bits)])
        self.step_size = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # self.step_size = torch.tensor(1.0)  # TODO hack
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                num_bins = 2 ** self.bits - 1
                self.step_size.copy_(2 * x.abs().mean() / np.sqrt(num_bins))
                self.initialized = True
                print('Initializing step size to ', self.step_size)

        output = 0
        pos = 0
        for b in range(self.bits):
            scale = 2 ** (self.bits - 1 - b)
            quantized = binary_lsq().apply(x, self.step_size * scale)
            output = output + quantized
            pos = pos + (quantized > 0).to(torch.int64)
            x = x - quantized

        # pos = 0, 1, 2, 3
        with torch.no_grad():
            pos_0 = (pos == 0).float()
            pos_1 = (pos == 1).float()
            pos_2 = (pos == 2).float()
            pos_3 = (pos == 3).float()
            pos_mask = torch.cat([pos_0, pos_1, pos_2, pos_3], 1)

        return output.tile(1, 4, 1, 1) * pos_mask


# class MultibitLSQNoScale(nn.Module):   # TODO better initialization?
#     def __init__(self, bits):
#         super(MultibitLSQNoScale, self).__init__()
#         self.bits = bits
#         self.quantizers = nn.ModuleList([LSQ(1) for i in range(bits)])
#
#     def forward(self, x):
#         if not self.quantizers[0].initialized:
#             with torch.no_grad():
#                 ss0, ss1 = get_lsq_step_size(x)
#                 ss = [ss0, ss1]
#                 for b in range(self.bits):
#                     self.quantizers[b].step_size.copy_(ss[b])
#                     self.quantizers[b].initialized = True
#                     print('Initializing step size of {} to {}'.format(b, self.quantizers[b].step_size))
#
#         output = []
#         for quantizer in self.quantizers:
#             quantized = quantizer(x)
#             output.append(quantized)
#             x = x - quantized
#
#         return torch.cat(output, 1)


class MultibitLSQShared(nn.Module):
    def __init__(self, bits):
        super(MultibitLSQShared, self).__init__()
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

        output = 0
        for b in range(self.bits):
            scale = 2 ** (self.bits - 1 - b)
            # quantized = lsq_quantize().apply(x, self.step_size * scale, 1)
            quantized = binary_lsq().apply(x, self.step_size * scale)
            output = output + quantized
            x = x - quantized

        return output


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


class MultibitLSQNoExpand(nn.Module):   # TODO better initialization?
    def __init__(self, bits):
        super(MultibitLSQNoExpand, self).__init__()
        self.bits = bits
        self.quantizers = nn.ModuleList([LSQ(1) for i in range(bits)])

    def forward(self, x):
        if not self.quantizers[0].initialized:
            with torch.no_grad():
                ss0, ss1 = get_lsq_step_size(x)
                ss = [ss0, ss1]
                for b in range(self.bits):
                    self.quantizers[b].step_size.copy_(ss[b])
                    self.quantizers[b].initialized = True
                    print('Initializing step size of {} to {}'.format(b, self.quantizers[b].step_size))

                # num_bins = 2 ** self.bits - 1
                # base_ss = 2 * x.abs().mean() / np.sqrt(num_bins)
                # for b in range(self.bits):
                #     self.quantizers[b].step_size.copy_(base_ss * 2 ** (self.bits - 1 - b))
                #     self.quantizers[b].initialized = True
                #     print('Initializing step size of {} to {}'.format(b, self.quantizers[b].step_size))

        output = 0
        for quantizer in self.quantizers:
            quantized = quantizer(x)
            output = output + quantized
            x = x - quantized

        return output


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
                                              padding, bias=False)
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
        super(MultibitLSQConvAct, self).__init__(in_chn * 4,
                                              out_chn,
                                              kernel_size,
                                              stride,
                                              padding, bias=False)
        self.act_quantizer = MultibitLSQNoScale(num_bits)
        # self.act_quantizer.step_size.requires_grad = False
        # self.act_quantizer = lambda x: x
        self.bits = num_bits
        self.C = in_chn

    def init_from(self, weight, step_size):
        with torch.no_grad():
            if step_size is not None:
                self.act_quantizer.step_size.copy_(step_size)
            for b in range(4):
                self.weight[:, b*self.C:(b+1)*self.C] = weight

    def forward(self, x):       # x: [N, C, H, W]
        # quant_input: [N, Cb, H, W]
        quant_input = self.act_quantizer(x)
        y = F.conv2d(quant_input, self.weight, stride=self.stride, padding=self.padding)
        # weight = self.weight[:, :self.C].tile(1, 2, 1, 1)
        # y = F.conv2d(quant_input, weight, stride=self.stride, padding=self.padding)
        return y


# class basic_binary_activation(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         out = torch.sign(input)
#         return out
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         # clip_mask = 0.5 * torch.logical_and(input >= -2, input <= 2).float()
#         clip_mask = torch.logical_and(input >= -1, input <= 1).float()
#         return grad_output * clip_mask


# class BinaryDuo(nn.Conv2d):
#     def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, num_bits=1):
#         super(BinaryDuo, self).__init__(in_chn*3, out_chn, kernel_size, stride, padding)
#         self.step_size = torch.tensor(1.0)
#         self.bits = num_bits
#         self.bins = 3
#         self.C = in_chn
#
#     def init_from(self, weight, step_size):
#         with torch.no_grad():
#             self.step_size.copy_(step_size)
#             # self.weight.copy_(weight)
#             for b in range(self.bins):
#                 self.weight[:, b * self.C:(b + 1) * self.C] = weight
#
#     def forward(self, x):
#         x = x / self.step_size
#         x0 = basic_binary_activation().apply(x)
#         x_m = basic_binary_activation().apply(x + 1)
#         x_p = basic_binary_activation().apply(x - 1)
#
#         #x0 = basic_binary_activation()
#         # x0 = torch.sign(x) * (self.step_size / 2)
#         # x_m = -(x < -self.step_size).float() * self.step_size
#         # x_p = (x > self.step_size).float() * self.step_size
#
#         quant_input = torch.cat([x0, x_m, x_p], 1) * self.step_size / 2
#
#         y = F.conv2d(quant_input, self.weight, stride=self.stride, padding=self.padding)
#         return y


class BinaryDuo(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, num_bits=1):
        super(BinaryDuo, self).__init__(in_chn*3, out_chn, kernel_size, stride, padding)
        self.step_size = nn.Parameter(torch.tensor(1.0))
        self.step_size.requires_grad = False
        self.bits = num_bits
        self.C = in_chn
        self.initialized = False

    def init_from(self, weight, step_size):
        with torch.no_grad():
            if step_size:
                self.step_size.copy_(step_size * 2)
                self.initialized = True

            # self.weight.copy_(weight)
            for b in range(3):
                self.weight[:, b * self.C:(b + 1) * self.C] = weight

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                num_bins = 2 ** self.bits - 1
                self.step_size.copy_(2 * x.abs().mean() / np.sqrt(num_bins))
                self.initialized = True
                print('Initializing step size to ', self.step_size)

        # x = x / self.step_size
        # x0 = basic_binary_activation().apply(x)
        # x_m = basic_binary_activation().apply(x + 1)
        # x_p = basic_binary_activation().apply(x - 1)
        # quant_input = torch.cat([x0, x_m, x_p], 1) * (self.step_size / 2)

        # TODO: use BinaryLSQ to learn the step size
        x0 = binary_lsq().apply(x, self.step_size)
        x_m = binary_lsq().apply(x + 0.5 * self.step_size.detach(), self.step_size)
        x_p = binary_lsq().apply(x - 0.5 * self.step_size.detach(), self.step_size)
        quant_input = torch.cat([x0, x_m, x_p], 1) / 2

        # x0 = basic_binary_activation().apply(x)
        # residual = x - x0
        # x1 = basic_binary_activation().apply(residual * 2) / 2
        # quant_input = torch.cat([x0, x1], 1) * self.step_size

        # x < -1: -3
        # -1 < x < 0: -1
        # 0 < x < 1: 1
        # x > 1: 3

        # quant_input = torch.cat([x0, x_m, x_p], 1) * (self.step_size / 2)

        # Possibility 1: inapproprite scale
        # Possibility 2:


        y = F.conv2d(quant_input, self.weight, stride=self.stride, padding=self.padding)
        return y


class BinaryDuo2(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, num_bits=1):
        super(BinaryDuo2, self).__init__(in_chn*2, out_chn, kernel_size, stride, padding)
        self.step_size = nn.Parameter(torch.tensor(1.0))
        self.step_size.requires_grad = False
        self.bits = num_bits
        self.C = in_chn

    def init_from(self, weight, step_size):
        with torch.no_grad():
            self.step_size.copy_(step_size)
            for b in range(2):
                self.weight[:, b * self.C:(b + 1) * self.C] = weight
            # w1 = weight[:, :self.C]
            # w2 = weight[:, self.C:]
            # self.weight[:, :self.C] = (w1 + w2) / 2
            # self.weight[:, self.C:] = w2

    def forward(self, x):
        x = x / self.step_size
        x0 = basic_binary_activation().apply(x)
        residual = x - x0
        x1 = basic_binary_activation().apply(residual * 2) / 2

        quant_input = torch.cat([x0, x1], 1) * self.step_size
        y = F.conv2d(quant_input, self.weight, stride=self.stride, padding=self.padding)
        return y
