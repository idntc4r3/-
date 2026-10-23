"""Unit tests for Tensor negation and subtraction (Day 20)."""

import numpy as np
import pytest
import torch
from nanotorch.tensor import Tensor


def test_neg_vs_torch() -> None:
    data = np.array([-2.0, 3.5, 0.0])
    ta = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    (-ta).sum().backward()

    na = Tensor(data, requires_grad=True)
    out = -na
    out.grad = np.ones_like(out.data)
    out._backward()

    assert np.allclose(out.data, (-ta).detach().numpy())
    assert na.grad is not None and ta.grad is not None
    assert np.allclose(na.grad, ta.grad.numpy())


def test_sub_vs_torch() -> None:
    data_a = np.random.randn(3, 2)
    data_b = np.random.randn(3, 2)

    ta = torch.tensor(data_a, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(data_b, dtype=torch.float64, requires_grad=True)
    (ta - tb).sum().backward()

    na = Tensor(data_a, requires_grad=True)
    nb = Tensor(data_b, requires_grad=True)
    out = na - nb
    out.grad = np.ones_like(out.data)
    out._backward()

    assert np.allclose(out.data, (ta - tb).detach().numpy())
    assert na.grad is not None and ta.grad is not None
    assert np.allclose(na.grad, ta.grad.numpy())
