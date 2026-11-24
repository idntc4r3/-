"""Softmax values and vector-Jacobian products against PyTorch."""

import numpy as np
import pytest
import torch

from nanotorch.nn import Linear, Sequential, Softmax
from nanotorch.tensor import Tensor


@pytest.mark.parametrize("shape,dim", [((4,), 0), ((3, 4), -1), ((2, 3, 4), 0), ((2, 3, 4), 1)])
def test_values_and_gradients_match_torch(shape, dim):
    rng = np.random.default_rng(22)
    data = rng.normal(size=shape)
    seed = rng.normal(size=shape)
    x, tx = Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    y, ty = Softmax(dim)(x), torch.softmax(tx, dim=dim)
    y.backward(seed)
    ty.backward(torch.tensor(seed))
    np.testing.assert_allclose(y.data, ty.detach().numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y.data.sum(axis=dim), 1.0, atol=1e-12)


def test_large_values_do_not_overflow():
    data = np.array([[10000.0, 10001.0, 9999.0], [-10000.0, -9999.0, -10001.0]])
    x = Tensor(data, requires_grad=True)
    with np.errstate(over="raise", invalid="raise"):
        y = x.softmax(-1)
        y.backward(np.array([[1.0, 2.0, 3.0], [1.0, -1.0, 0.5]]))
    tx = torch.tensor(data, requires_grad=True)
    ty = torch.softmax(tx, dim=-1)
    ty.backward(torch.tensor([[1.0, 2.0, 3.0], [1.0, -1.0, 0.5]], dtype=torch.float64))
    np.testing.assert_allclose(y.data, ty.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-12)


def test_sequential_and_repeated_backward():
    layer = Softmax(-1)
    assert layer.parameters() == []
    net = Sequential(Linear(3, 3), layer)
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = net(x)
    y.backward([[1.0, -1.0, 0.5]])
    old = x.grad.copy()
    y.backward([[1.0, -1.0, 0.5]])
    np.testing.assert_allclose(x.grad, 2 * old, atol=1e-12)


@pytest.mark.parametrize("dim", [-3, 2])
def test_invalid_dimension(dim):
    with pytest.raises((IndexError, ValueError)):
        Softmax(dim)(Tensor(np.ones((2, 3))))
