"""Compare tensor and layer tanh values and gradients with PyTorch."""

import numpy as np
import pytest
import torch

from nanotorch.nn import Linear, Tanh
from nanotorch.tensor import Tensor


@pytest.mark.parametrize("data", [0.0, [-3.0, 0.0, 2.0], [[-2.0, 3.0], [1.5, -0.1]]])
def test_tanh_matches_torch(data):
    x = Tensor(data, requires_grad=True)
    tx = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    y, ty = Tanh()(x), torch.tanh(tx)
    y.sum().backward()
    ty.sum().backward()
    np.testing.assert_allclose(y.data, ty.detach().numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), rtol=1e-12, atol=1e-15)


def test_tanh_extreme_values():
    data = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0])
    x = Tensor(data, requires_grad=True)
    tx = torch.tensor(data, requires_grad=True)
    with np.errstate(over="raise", invalid="raise"):
        y = x.tanh()
        y.sum().backward()
    ty = torch.tanh(tx)
    ty.sum().backward()
    np.testing.assert_allclose(y.data, ty.detach().numpy(), atol=1e-15)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-15)


def test_linear_tanh_chain():
    data = np.array([[1.0, -2.0], [-1.0, 3.0]])
    weight = np.array([[1.5, -0.3], [-0.4, 0.9], [0.2, 0.7]])
    bias = np.array([0.5, -0.3, 0.4])
    seed = np.array([[1.0, -2.0, 0.5], [0.3, 0.4, -1.0]])
    layer, ref = Linear(2, 3), torch.nn.Linear(2, 3, dtype=torch.float64)
    layer.weight.data[:] = weight
    layer.bias.data[:] = bias
    with torch.no_grad():
        ref.weight.copy_(torch.from_numpy(weight))
        ref.bias.copy_(torch.from_numpy(bias))
    x, tx = Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    y, ty = Tanh()(layer(x)), torch.tanh(ref(tx))
    y.backward(seed)
    ty.backward(torch.tensor(seed))
    np.testing.assert_allclose(y.data, ty.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(layer.weight.grad, ref.weight.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(layer.bias.grad, ref.bias.grad.numpy(), atol=1e-12)


def test_tanh_no_parameters_and_repeated_backward():
    activation = Tanh()
    assert activation.parameters() == []
    assert not activation(Tensor([1.0])).requires_grad
    x = Tensor([0.0, 1.0], requires_grad=True)
    y = activation(x)
    y.backward([1.0, 2.0])
    y.backward([2.0, 3.0])
    np.testing.assert_allclose(x.grad, (1 - y.data ** 2) * [3.0, 5.0])


def test_tanh_rejects_non_tensor_input():
    with pytest.raises(TypeError):
        Tanh()([1.0])
