"""Day 19: Tensor addition tests against PyTorch."""

import numpy as np
import pytest
import torch
from nanotorch.tensor import Tensor


@pytest.mark.parametrize("shape_a, shape_b", [
    ((3, 1), (1, 4)),
    ((2, 3, 1), (1, 3, 4)),
    ((5,), (1,)),
    ((2, 2), (2, 2)),
])
def test_broadcast_add_vs_torch(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
    data_a = np.random.randn(*shape_a)
    data_b = np.random.randn(*shape_b)

    # PyTorch
    ta = torch.tensor(data_a, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(data_b, dtype=torch.float64, requires_grad=True)
    tc = ta + tb
    loss_t = tc.sum()
    loss_t.backward()

    # NanoTorch
    na = Tensor(data_a, requires_grad=True)
    nb = Tensor(data_b, requires_grad=True)
    nc = na + nb

    # Seed upstream grad
    nc.grad = np.ones_like(nc.data)
    nc._backward()

    # Compare
    assert np.allclose(nc.data, tc.detach().numpy())
    assert na.grad is not None and ta.grad is not None
    assert np.allclose(na.grad, ta.grad.numpy())
    assert nb.grad is not None and tb.grad is not None
    assert np.allclose(nb.grad, tb.grad.numpy())
