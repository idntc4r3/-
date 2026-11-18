"""Check sigmoid values and reverse-mode gradients against PyTorch."""

import numpy as np
import pytest
import torch

from nanotorch.nn import Linear, Sigmoid
from nanotorch.tensor import Tensor


@pytest.mark.parametrize("data", [0.0, [-4.0, 0.0, 2.0], [[-2.0, 3.0], [1.5, -0.1]]])
def test_sigmoid_values_and_gradients(data):
    x = Tensor(data, requires_grad=True)
    tx = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    y, ty = Sigmoid()(x), torch.sigmoid(tx)
    y.sum().backward()
    ty.sum().backward()
    np.testing.assert_allclose(y.data, ty.detach().numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), rtol=1e-12, atol=1e-15)


def test_extreme_values_no_overflow():
    data = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0])
    x = Tensor(data, requires_grad=True)
    tx = torch.tensor(data, requires_grad=True)
    with np.errstate(over="raise", invalid="raise"):
        y = Sigmoid()(x)
        y.sum().backward()
    ty = torch.sigmoid(tx)
    ty.sum().backward()
    np.testing.assert_allclose(y.data, ty.detach().numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), rtol=1e-12, atol=1e-15)


def test_linear_sigmoid_chain_and_custom_seed():
    data = np.array([[1.0, -2.0], [-1.0, 3.0]])
    weights = np.array([[1.5, -0.3], [-0.4, 0.9], [0.2, 0.7]])
    biases = np.array([0.5, -0.3, 0.4])
    seed = np.array([[1.0, -2.0, 0.5], [0.3, 0.4, -1.0]])
    layer = Linear(2, 3)
    layer.weight.data[:] = weights
    layer.bias.data[:] = biases
    ref = torch.nn.Linear(2, 3, dtype=torch.float64)
    with torch.no_grad():
        ref.weight.copy_(torch.from_numpy(weights))
        ref.bias.copy_(torch.from_numpy(biases))
    x, tx = Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    y, ty = Sigmoid()(layer(x)), torch.sigmoid(ref(tx))
    y.backward(seed)
    ty.backward(torch.tensor(seed))
    np.testing.assert_allclose(y.data, ty.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(layer.weight.grad, ref.weight.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(layer.bias.grad, ref.bias.grad.numpy(), atol=1e-12)


def test_no_parameters_and_repeated_backward():
    activation = Sigmoid()
    assert activation.parameters() == []
    assert not activation(Tensor([1.0])).requires_grad
    x = Tensor([0.0, 1.0], requires_grad=True)
    y = activation(x)
    y.backward([1.0, 2.0])
    y.backward([2.0, 3.0])
    expected = y.data * (1 - y.data) * [3.0, 5.0]
    np.testing.assert_allclose(x.grad, expected)
