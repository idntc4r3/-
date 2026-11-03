import numpy as np
import torch
from nanotorch.tensor import Tensor

def test_mean_vs_torch():
    data = np.random.randn(4, 5)
    ta = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    ta.mean().backward()
    na = Tensor(data, requires_grad=True)
    out = na.mean()
    out.grad = np.ones_like(out.data)
    out._backward()
    # propagate through sum
    for c in out._prev:
        c._backward()
    assert np.allclose(out.data, ta.mean().detach().numpy())
    assert np.allclose(na.grad, ta.grad.numpy())