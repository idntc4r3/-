"""Check the affine layer against PyTorch."""

import numpy as np
import pytest
import torch

from nanotorch.nn import Linear
from nanotorch.tensor import Tensor


@pytest.mark.parametrize("shape", [(3,), (4, 3), (2, 4, 3)])
@pytest.mark.parametrize("bias", [True, False])
def test_linear_forward_and_backward_match_torch(shape, bias):
    rng = np.random.default_rng(42)
    inputs = rng.normal(size=shape)
    weights = rng.normal(size=(2, 3))
    biases = rng.normal(size=2)
    layer = Linear(3, 2, bias=bias)
    layer.weight.data[:] = weights
    if bias:
        layer.bias.data[:] = biases
    reference = torch.nn.Linear(3, 2, bias=bias, dtype=torch.float64)
    with torch.no_grad():
        reference.weight.copy_(torch.from_numpy(weights))
        if bias:
            reference.bias.copy_(torch.from_numpy(biases))
    x = Tensor(inputs, requires_grad=True)
    tx = torch.tensor(inputs, dtype=torch.float64, requires_grad=True)
    y = layer(x)
    ty = reference(tx)
    seed = rng.normal(size=y.shape)
    y.backward(seed)
    ty.backward(torch.tensor(seed, dtype=torch.float64))
    np.testing.assert_allclose(y.data, ty.detach().numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(layer.weight.grad, reference.weight.grad.numpy(), rtol=1e-12, atol=1e-12)
    if bias:
        np.testing.assert_allclose(layer.bias.grad, reference.bias.grad.numpy(), rtol=1e-12, atol=1e-12)


def test_parameters_and_initialization():
    np.random.seed(10)
    layer = Linear(4, 3)
    assert layer.weight.shape == (3, 4)
    assert layer.bias.shape == (3,)
    assert layer.parameters() == [layer.weight, layer.bias]
    assert all(p.requires_grad for p in layer.parameters())
    assert np.all(np.abs(layer.weight.data) <= 0.5)
    assert np.all(np.abs(layer.bias.data) <= 0.5)
    without_bias = Linear(4, 3, bias=False)
    assert without_bias.bias is None
    assert without_bias.parameters() == [without_bias.weight]


@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (-1, 2)])
def test_invalid_feature_sizes(shape):
    with pytest.raises(ValueError):
        Linear(*shape)
