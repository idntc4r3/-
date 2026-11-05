import numpy as np
import torch
from nanotorch.tensor import Tensor

def test_full_backward():
    x = Tensor.randn(3, 3, requires_grad=True)
    w = Tensor.randn(3, 3, requires_grad=True)
    loss = (x @ w).sum()
    loss.backward()
    assert x.grad is not None and w.grad is not None