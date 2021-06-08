import torch
import quant.cpp_extension.calc_quant_bin as ext

a = torch.Tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
l, r = ext.calc_quant_bin(a)
print(l, r)