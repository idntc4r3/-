import numpy as np
import torch
from nanotorch.tensor import Tensor

def test_transpose_vs_torch():
    data = np.random.randn(3, 5)
    ta = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    ta.T.sum().backward()
    na = Tensor(data, requires_grad=True)
    out = na.T
    out.grad = np.ones_like(out.data)
    out._backward()
    assert np.allclose(out.data, ta.T.detach().numpy())
    assert np.allclose(na.grad, ta.grad.numpy())