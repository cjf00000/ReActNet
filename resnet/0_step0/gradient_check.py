import torch
from quantize import binary_activation, BinaryActivation, multi_bit_quantize

binact = binary_activation()
binact0 = BinaryActivation()

act, _, _ = torch.load('layers/layer_1.pth.tar')
grad_output = torch.randn_like(act)
act.requires_grad_()

y = binact.apply(act)
loss = (y * grad_output).sum()
loss.backward()
act_grad = act.grad.clone()

act.grad = None
y0 = binact0(act)
loss0 = (y0 * grad_output).sum()
loss0.backward()
act_grad0 = act.grad.clone()

act.grad = None
y1 = multi_bit_quantize().apply(act, 1)

print(y0.norm(), (y-y0).norm(), (y1-y0).norm())
print(act_grad0.norm(), (act_grad-act_grad0).norm())

