import numpy as np
import torch
from nanotorch.tensor import Tensor

def test_slice_vs_torch():
    data = np.random.randn(5, 5)
    ta = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    ta[1:3, :].sum().backward()
    na = Tensor(data, requires_grad=True)
    out = na[1:3, :]
    out.grad = np.ones_like(out.data)
    out._backward()
    assert np.allclose(out.data, ta[1:3, :].detach().numpy())
    assert np.allclose(na.grad, ta.grad.numpy())