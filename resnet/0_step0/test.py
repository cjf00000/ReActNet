import numpy as np
import torch
import torch.nn.functional as F

x, weight, _ = torch.load('layers/layer_1.pth.tar')
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


def unbiased_quantize(a, dim=0):
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

    return  xnornet_c, optimal_c


A = x[:, :, 0, 0]
B = weight[:, :, 0, 0].t()
exact_C = A @ B
print('Exact Norm ', exact_C.norm())
print(A.abs().mean(), A.max() - A.min())
xnornet_C, optimal_C = mm_xnornet(A, B)
print('XNor Error ', (exact_C - xnornet_C).norm(), (exact_C - optimal_C).norm())

errors = []
for i in range(100):
    C = mm_unbiased(A, B)
    errors.append((exact_C - C).norm())

print('Unbiased Error ', torch.stack(errors).mean())
