"""Unit tests for Tensor matmul backward pass (Day 24)."""

import numpy as np
import pytest
import torch
from nanotorch.tensor import Tensor


def test_matmul_backward_vs_torch() -> None:
    data_a = np.random.randn(3, 4)
    data_b = np.random.randn(4, 2)

    ta = torch.tensor(data_a, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(data_b, dtype=torch.float64, requires_grad=True)
    tc = ta @ tb
    tc.sum().backward()

    na = Tensor(data_a, requires_grad=True)
    nb = Tensor(data_b, requires_grad=True)
    nc = na @ nb
    nc.grad = np.ones_like(nc.data)
    nc._backward()

    assert np.allclose(nc.data, tc.detach().numpy())
    assert na.grad is not None and ta.grad is not None
    assert np.allclose(na.grad, ta.grad.numpy())
    assert nb.grad is not None and tb.grad is not None
    assert np.allclose(nb.grad, tb.grad.numpy())
