"""Day 25: Comprehensive matmul tests."""

import numpy as np
import pytest
import torch
from nanotorch.tensor import Tensor


@pytest.mark.parametrize("m, k, n", [
    (10, 20, 5),
    (8, 8, 8),
    (1, 15, 1),
    (7, 3, 11),
])
def test_matmul_shapes_vs_torch(m: int, k: int, n: int) -> None:
    data_a = np.random.randn(m, k)
    data_b = np.random.randn(k, n)

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
    assert np.allclose(na.grad, ta.grad.numpy())
    assert np.allclose(nb.grad, tb.grad.numpy())


def test_chained_matmul() -> None:
    """X @ W1 @ W2 linear pipeline."""
    X = np.random.randn(4, 8)
    W1 = np.random.randn(8, 6)
    W2 = np.random.randn(6, 2)

    tX = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    tW1 = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
    tW2 = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
    out_t = (tX @ tW1) @ tW2
    out_t.sum().backward()

    nX = Tensor(X, requires_grad=True)
    nW1 = Tensor(W1, requires_grad=True)
    nW2 = Tensor(W2, requires_grad=True)
    H = nX @ nW1
    out_n = H @ nW2

    out_n.grad = np.ones_like(out_n.data)
    out_n._backward()
    H._backward()

    assert np.allclose(out_n.data, out_t.detach().numpy())
    assert np.allclose(nW2.grad, tW2.grad.numpy())
    assert np.allclose(nW1.grad, tW1.grad.numpy())
    assert np.allclose(nX.grad, tX.grad.numpy())
