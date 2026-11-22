"""Dropout mode, scaling and gradient checks."""

import numpy as np
import pytest
import torch

from nanotorch.nn import Dropout, Linear, Sequential
from nanotorch.tensor import Tensor


@pytest.mark.parametrize("p", [-0.1, 1.01, float("nan")])
def test_invalid_probability(p):
    with pytest.raises(ValueError):
        Dropout(p)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_boundary_values_match_torch(p):
    data = np.array([1.0, -2.0, 4.0])
    x, tx = Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    y = Dropout(p)(x)
    ty = torch.nn.Dropout(p)(tx)
    y.sum().backward()
    ty.sum().backward()
    np.testing.assert_array_equal(y.data, ty.detach().numpy())
    np.testing.assert_array_equal(x.grad, tx.grad.numpy())


def test_training_mask_scale_and_gradient_match_torch_multiplication():
    np.random.seed(42)
    x = Tensor(np.ones(10000), requires_grad=True)
    y = Dropout(0.25)(x)
    assert set(np.unique(y.data)) == {0.0, 1 / 0.75}
    assert abs(np.mean(y.data) - 1) < 0.03
    seed = np.linspace(-1, 1, x.size)
    y.backward(seed)
    tx = torch.tensor(np.ones(10000), requires_grad=True)
    ty = tx * torch.tensor(y.data)
    ty.backward(torch.tensor(seed))
    np.testing.assert_allclose(x.grad, tx.grad.numpy())


def test_eval_is_identity_and_nested_sequential_modes():
    dropout = Dropout(0.8)
    net = Sequential(Linear(2, 2), Sequential(dropout))
    assert net.eval() is net
    assert not dropout.training
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = dropout(x)
    np.testing.assert_array_equal(y.data, x.data)
    y.sum().backward()
    np.testing.assert_array_equal(x.grad, [1.0, 1.0])
    assert net.train() is net
    assert dropout.training
    assert dropout.parameters() == []


def test_seed_reproducibility():
    layer = Dropout(0.5)
    np.random.seed(10)
    first = layer(Tensor(np.ones(30))).data
    np.random.seed(10)
    second = layer(Tensor(np.ones(30))).data
    np.testing.assert_array_equal(first, second)
