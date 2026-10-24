"""Unit tests for Tensor elementwise multiplication (Day 21)."""

import numpy as np
import pytest
import torch
from nanotorch.tensor import Tensor


@pytest.mark.parametrize("shape_a, shape_b", [
    ((3, 2), (3, 2)),
    ((3, 1), (1, 4)),
    ((4,), (1,)),
])
def test_mul_vs_torch(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
    data_a = np.random.randn(*shape_a)
    data_b = np.random.randn(*shape_b)

    ta = torch.tensor(data_a, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(data_b, dtype=torch.float64, requires_grad=True)
    (ta * tb).sum().backward()

    na = Tensor(data_a, requires_grad=True)
    nb = Tensor(data_b, requires_grad=True)
    out = na * nb
    out.grad = np.ones_like(out.data)
    out._backward()

    assert np.allclose(out.data, (ta * tb).detach().numpy())
    assert na.grad is not None and ta.grad is not None
    assert np.allclose(na.grad, ta.grad.numpy())
    assert nb.grad is not None and tb.grad is not None
    assert np.allclose(nb.grad, tb.grad.numpy())
