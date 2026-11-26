"""Flatten shape and gradient checks against PyTorch."""

import numpy as np
import pytest
import torch

from nanotorch.nn import Flatten, Linear, Sequential
from nanotorch.tensor import Tensor


@pytest.mark.parametrize("shape,start,end", [((2, 3, 4), 1, -1), ((2, 3, 4), 0, 1),
                                             ((2, 3, 4, 5), -3, -2), ((2, 0, 3), 1, -1)])
def test_forward_and_gradient_match_torch(shape, start, end):
    rng = np.random.default_rng(17)
    data = rng.normal(size=shape)
    x, tx = Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    y, ty = Flatten(start, end)(x), torch.nn.Flatten(start, end)(tx)
    seed = rng.normal(size=y.shape)
    y.backward(seed)
    ty.backward(torch.tensor(seed))
    assert y.shape == tuple(ty.shape)
    np.testing.assert_allclose(y.data, ty.detach().numpy())
    np.testing.assert_allclose(x.grad, tx.grad.numpy())


def test_network_composition_and_parameters():
    flat = Flatten()
    linear = Linear(6, 2)
    model = Sequential(flat, linear)
    x = Tensor(np.ones((4, 2, 3)), requires_grad=True)
    y = model(x)
    assert y.shape == (4, 2)
    y.sum().backward()
    assert x.grad.shape == x.shape
    assert flat.parameters() == []
    assert model.parameters() == linear.parameters()


@pytest.mark.parametrize("start,end", [(3, -1), (-4, -1), (2, 1)])
def test_invalid_dimensions(start, end):
    with pytest.raises((IndexError, ValueError)):
        Flatten(start, end)(Tensor(np.ones((2, 3, 4))))


def test_scalar_flatten_and_repeated_backward():
    x = Tensor(3.0, requires_grad=True)
    y = Flatten(0, -1)(x)
    assert y.shape == (1,)
    y.backward([2.0])
    y.backward([3.0])
    np.testing.assert_allclose(x.grad, 5.0)
