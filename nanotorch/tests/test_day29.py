import numpy as np
import torch
from nanotorch.tensor import Tensor

def test_sum_backward_vs_torch():
    data = np.random.randn(3, 4)
    ta = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    ta.sum(axis=1).sum().backward()
    na = Tensor(data, requires_grad=True)
    out = na.sum(axis=1)
    out.grad = np.ones_like(out.data)
    out._backward()
    assert np.allclose(na.grad, ta.grad.numpy())