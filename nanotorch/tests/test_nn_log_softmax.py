"""LogSoftmax forward and vector-Jacobian products against PyTorch."""

import numpy as np
import pytest
import torch

from nanotorch.nn import Linear, LogSoftmax, Sequential
from nanotorch.tensor import Tensor


@pytest.mark.parametrize("shape,dim", [((4,), 0), ((3, 4), -1), ((2, 3, 4), 0), ((2, 3, 4), 1)])
def test_values_and_gradients_match_torch(shape, dim):
    rng = np.random.default_rng(18)
    data, seed = rng.normal(size=shape), rng.normal(size=shape)
    x, tx = Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    y, ty = LogSoftmax(dim)(x), torch.log_softmax(tx, dim=dim)
    y.backward(seed)
    ty.backward(torch.tensor(seed))
    np.testing.assert_allclose(y.data, ty.detach().numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.exp(y.data).sum(axis=dim), 1, atol=1e-12)


def test_extreme_logits_remain_finite_and_match_torch():
    data = np.array([[1000.0, -1000.0, 999.0], [-1000.0, -1001.0, -3000.0]])
    seed = np.array([[1.0, 2.0, 3.0], [1.0, -1.0, 0.5]])
    x, tx = Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        y = x.log_softmax(-1)
        y.backward(seed)
    ty = torch.log_softmax(tx, dim=-1)
    ty.backward(torch.tensor(seed))
    assert np.isfinite(y.data).all()
    np.testing.assert_allclose(y.data, ty.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-12)


def test_sequential_and_repeated_backward():
    layer = LogSoftmax(-1)
    assert layer.parameters() == []
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = Sequential(Linear(3, 3), layer)(x)
    y.backward([[1.0, -1.0, 0.5]])
    old = x.grad.copy()
    y.backward([[1.0, -1.0, 0.5]])
    np.testing.assert_allclose(x.grad, 2 * old, atol=1e-12)


@pytest.mark.parametrize("dim", [-3, 2])
def test_invalid_dimension(dim):
    with pytest.raises((IndexError, ValueError)):
        LogSoftmax(dim)(Tensor(np.ones((2, 3))))


def test_empty_tensor_shape_and_backward():
    x = Tensor(np.empty((2, 0)), requires_grad=True)
    y = x.log_softmax(-1)
    assert y.shape == (2, 0)
    y.sum().backward()
    assert x.grad.shape == (2, 0)
