import numpy as np
import torch
import torch.nn.functional as F

x, weight, _ = torch.load('layers/layer_1.pth.tar')
x.requires_grad = False
weight.requires_grad = False
y = F.conv2d(x, weight, padding=1)

N, C, H, W = x.shape
K = 3

# # Manually implement the conv
# y_m = torch.zeros_like(y)
# for h in range(H):
#     for w in range(W):
#         for dh in range(-1, 2):
#             for dw in range(-1, 2):
#                 h_ = h + dh
#                 w_ = w + dw
#                 if (h_>=0) and (h_<H) and (w_>=0) and (w_<W):
#                     data = x[:, :, h_, w_]
#                     kernel = weight[:, :, dh+1, dw+1]
#                     y_m[:, :, h, w] += data @ kernel.t()
#
#
# print(y.norm())
# print((y-y_m).norm())


# Manually implement the conv by dot products
x_padded = F.pad(x, (1, 1, 1, 1))
# y_m = torch.zeros_like(y)
# for n in range(N):
#     for c in range(C):
#         for h in range(H):
#             for w in range(W):
#                 x_field = x_padded[n, :, h:h+3, w:w+3]
#                 w_field = weight[c, :, :, :]
#                 y_m[n, c, h, w] = (x_field * w_field).sum()
#     break
#
# print(y[0].norm())
# print((y[0]-y_m[0]).norm())


def unbiased_quantize(a, dim=0):
    if dim is None:
        z = a.min()
        r = a.max() - z
    else:
        z = a.min(dim, keepdim=True)[0]
        r = a.max(dim, keepdim=True)[0] - z

    a = (a - z) / r
    samples = torch.rand_like(a) >= a
    samples = samples.float() * r + z
    return samples


def mm_unbiased(a, b):
    # Randomly quantize
    a = unbiased_quantize(a, dim=1)
    b = unbiased_quantize(b, dim=0)
    return a @ b


def sign(x, honest=True):
    if honest:
        return (x >= 0).float() * 2 - 1
    else:
        return torch.sign(x)


def mm_xnornet(a, b):
    c = a @ b
    scale_a = a.abs().mean(1, keepdim=True)
    scale_b = b.abs().mean(0, keepdim=True)
    print('Scale a ', scale_a.t())
    print('Scale b ', scale_b)
    a = sign(a)
    b = sign(b)
    c_hat = a @ b
    xnornet_c = scale_a * scale_b * c_hat
    os_a = scale_a
    os_b = scale_b
    for i in range(10):
        old_os_a, old_os_b = os_a.clone(), os_b.clone()
        os_a = (os_b * c_hat * c).sum(1, keepdim=True) / \
               ((os_b * c_hat) ** 2).sum(1, keepdim=True)
        os_b = (os_a * c_hat * c).sum(0, keepdim=True) / \
               ((os_a * c_hat) ** 2).sum(0, keepdim=True)
        optimal_c = os_a * os_b * c_hat
        print('Iteration ', i, ' diff ', (old_os_a - os_a).norm(), (old_os_b - os_b).norm(), (c - optimal_c).norm())

    return xnornet_c, optimal_c


def conv_unbiased(a, b):
    # Randomly quantize
    a = unbiased_quantize(a, dim=1)
    b = unbiased_quantize(b, dim=0)
    return F.conv2d(a, b, padding=1)


def conv_xnornet(a, b):
    scale_a = a.abs().mean(dim=1, keepdim=True) # [N, 1, W, H]
    kernel = torch.ones(1, 1, 3, 3).cuda() / 9
    K = F.conv2d(scale_a, kernel, padding=1)

    scale_b = b.abs().mean([1, 2, 3]).view(1, 64, 1, 1)
    a = sign(a)
    b = sign(b)
    c_hat = F.conv2d(a, b, padding=1)
    return K, c_hat, K * scale_b * c_hat


def dot_unbiased(a, b):
    a = unbiased_quantize(a, None)
    b = unbiased_quantize(b, None)
    return (a * b).sum()

#
# A = x[:, :, 0, 0]
# B = weight[:, :, 0, 0].t()
# exact_C = A @ B
# print('Exact Norm ', exact_C.norm())
# print(A.abs().mean(), A.max() - A.min())
# xnornet_C, optimal_C = mm_xnornet(A, B)
# print('XNor Error ', (exact_C - xnornet_C).norm(), (exact_C - optimal_C).norm())
#
# errors = []
# for i in range(100):
#     C = mm_unbiased(A, B)
#     errors.append((exact_C - C).norm())
#
# print('Unbiased Error ', torch.stack(errors).mean())
# print(y.norm())

K, c_hat, xnornet_y = conv_xnornet(x, weight)
print('XNor Error ', (y - xnornet_y).norm())

errors = []
for i in range(10):
    unbiased_y = conv_unbiased(x, weight)
    errors.append((y - unbiased_y).norm())

print('Unbiased Error ', torch.stack(errors).mean())

# Let's take a look at the first dot product
x_field = x_padded[0, :, 1:4, 1:4]
w_field = weight[0, :, :, :]
y_exact = (x_field * w_field).sum()
x_hat = sign(x_field)
w_hat = sign(w_field)
x_scale = x_field.mean()
w_scale = w_field.mean()
y_binary = (x_hat * w_hat * x_scale * w_scale).sum()
y_unbiased = dot_unbiased(x_field, w_field)