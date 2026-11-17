"""Check ReLU values and reverse-mode gradients against PyTorch."""

import numpy as np
import pytest
import torch

from nanotorch.nn import Linear, ReLU
from nanotorch.tensor import Tensor


@pytest.mark.parametrize("data", [0.0, [-3.0, 0.0, 2.0], [[-2.0, 0.0], [3.0, -0.1]]])
def test_relu_forward_and_gradient(data):
    x = Tensor(data, requires_grad=True)
    tx = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    y = ReLU()(x)
    ty = torch.nn.ReLU()(tx)
    np.testing.assert_allclose(y.data, ty.detach().numpy())
    y.sum().backward()
    ty.sum().backward()
    np.testing.assert_allclose(x.grad, tx.grad.numpy())


def test_relu_with_linear_and_custom_gradient():
    data = np.array([[1.0, -2.0], [-1.0, 3.0]])
    weight = np.array([[1.5, -0.3], [-0.4, 0.9], [0.2, 0.7]])
    bias = np.array([0.5, -0.3, 0.4])
    seed = np.array([[1.0, -2.0, 0.5], [0.3, 0.4, -1.0]])
    linear = Linear(2, 3)
    linear.weight.data[:] = weight
    linear.bias.data[:] = bias
    reference = torch.nn.Linear(2, 3, dtype=torch.float64)
    with torch.no_grad():
        reference.weight.copy_(torch.from_numpy(weight))
        reference.bias.copy_(torch.from_numpy(bias))
    x = Tensor(data, requires_grad=True)
    tx = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    y = ReLU()(linear(x))
    ty = torch.relu(reference(tx))
    y.backward(seed)
    ty.backward(torch.tensor(seed))
    np.testing.assert_allclose(y.data, ty.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(linear.weight.grad, reference.weight.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(linear.bias.grad, reference.bias.grad.numpy(), atol=1e-12)


def test_relu_no_parameters_and_no_gradient_tracking():
    layer = ReLU()
    y = layer(Tensor([-1.0, 2.0]))
    assert layer.parameters() == []
    assert not y.requires_grad
    np.testing.assert_array_equal(y.data, [0.0, 2.0])


def test_relu_repeated_backward_accumulates():
    x = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    y = ReLU()(x)
    y.backward([1.0, 1.0, 2.0])
    y.backward([2.0, 2.0, 3.0])
    np.testing.assert_array_equal(x.grad, [0.0, 0.0, 5.0])


def test_relu_nan_matches_torch():
    x = Tensor([np.nan], requires_grad=True)
    tx = torch.tensor([np.nan], dtype=torch.float64, requires_grad=True)
    y, ty = ReLU()(x), torch.relu(tx)
    y.sum().backward()
    ty.sum().backward()
    np.testing.assert_allclose(y.data, ty.detach().numpy(), equal_nan=True)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), equal_nan=True)
