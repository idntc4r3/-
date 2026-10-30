import numpy as np
import torch
from nanotorch.tensor import Tensor

def test_reshape_vs_torch():
    data = np.random.randn(2, 6)
    ta = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    ta.reshape(3, 4).sum().backward()
    na = Tensor(data, requires_grad=True)
    out = na.reshape(3, 4)
    out.grad = np.ones_like(out.data)
    out._backward()
    assert np.allclose(out.data, ta.reshape(3, 4).detach().numpy())
    assert np.allclose(na.grad, ta.grad.numpy())