"""Check composition and parameter discovery for sequential networks."""

import numpy as np
import pytest
import torch

from nanotorch.nn import Linear, ReLU, Sequential, Tanh
from nanotorch.tensor import Tensor


def test_sequential_matches_torch_values_and_gradients():
    rng = np.random.default_rng(42)
    xdata = rng.normal(size=(3, 2))
    weights = [rng.normal(size=(4, 2)), rng.normal(size=(1, 4))]
    biases = [rng.normal(size=4), rng.normal(size=1)]
    first, second = Linear(2, 4), Linear(4, 1)
    model = Sequential(first, ReLU(), second, Tanh())
    reference = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(),
                                    torch.nn.Linear(4, 1), torch.nn.Tanh()).double()
    for layer, ref, w, b in zip((first, second), (reference[0], reference[2]), weights, biases):
        layer.weight.data[:] = w
        layer.bias.data[:] = b
        with torch.no_grad():
            ref.weight.copy_(torch.from_numpy(w))
            ref.bias.copy_(torch.from_numpy(b))
    x, tx = Tensor(xdata, requires_grad=True), torch.tensor(xdata, requires_grad=True)
    y, ty = model(x), reference(tx)
    y.sum().backward()
    ty.sum().backward()
    np.testing.assert_allclose(y.data, ty.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-12)
    for p, tp in zip(model.parameters(), reference.parameters()):
        np.testing.assert_allclose(p.grad, tp.grad.numpy(), atol=1e-12)


def test_nested_layers_and_shared_parameters_are_unique():
    shared = Linear(2, 2)
    tail = Linear(2, 1)
    model = Sequential(shared, Sequential(ReLU(), shared), tail)
    params = model.parameters()
    assert len(params) == 4
    assert all(a is b for a, b in zip(params, [shared.weight, shared.bias, tail.weight, tail.bias]))
    assert model(Tensor([[1.0, 2.0]])).shape == (1, 1)


def test_empty_sequential_is_identity():
    x = Tensor([1.0, 2.0], requires_grad=True)
    model = Sequential()
    assert model(x) is x
    assert model.parameters() == []


def test_rejects_invalid_layers():
    with pytest.raises(TypeError):
        Sequential(object())
