"""LeakyReLU forward and gradient checks against PyTorch."""

import numpy as np
import pytest
import torch

from nanotorch.nn import LeakyReLU, Linear, Sequential
from nanotorch.tensor import Tensor


@pytest.mark.parametrize("slope", [0.0, 0.01, 0.2, 1.5, -0.3])
@pytest.mark.parametrize("data", [0.0, [-3.0, 0.0, 2.0], [[-2.0, 3.0], [1.5, -0.1]]])
def test_values_and_gradients_match_torch(slope, data):
    x = Tensor(data, requires_grad=True)
    tx = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    y, ty = LeakyReLU(slope)(x), torch.nn.LeakyReLU(slope)(tx)
    seed = np.full(y.shape, 1.7)
    y.backward(seed)
    ty.backward(torch.tensor(seed, dtype=torch.float64))
    np.testing.assert_allclose(y.data, ty.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-12)


def test_linear_chain_matches_torch():
    data = np.array([[1.0, -2.0], [-1.0, 3.0]])
    weights = np.array([[1.5, -0.3], [-0.4, 0.9]])
    biases = np.array([0.5, -0.3])
    linear = Linear(2, 2)
    linear.weight.data[:] = weights
    linear.bias.data[:] = biases
    ref = torch.nn.Linear(2, 2, dtype=torch.float64)
    with torch.no_grad():
        ref.weight.copy_(torch.from_numpy(weights))
        ref.bias.copy_(torch.from_numpy(biases))
    x, tx = Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    y = Sequential(linear, LeakyReLU(0.2))(x)
    ty = torch.nn.functional.leaky_relu(ref(tx), 0.2)
    y.sum().backward()
    ty.sum().backward()
    np.testing.assert_allclose(y.data, ty.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(linear.weight.grad, ref.weight.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(linear.bias.grad, ref.bias.grad.numpy(), atol=1e-12)


def test_no_parameters_and_repeated_backward():
    layer = LeakyReLU()
    assert layer.parameters() == []
    assert not layer(Tensor([1.0])).requires_grad
    x = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    y = layer(x)
    y.backward([1.0, 1.0, 2.0])
    y.backward([2.0, 2.0, 3.0])
    np.testing.assert_allclose(x.grad, [0.03, 0.03, 5.0])


def test_rejects_non_tensor_input():
    with pytest.raises(TypeError):
        LeakyReLU()([1.0])
