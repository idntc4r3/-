"""Layer normalization values and gradients compared to PyTorch."""

import numpy as np
import pytest
import torch

from nanotorch.nn import LayerNorm, Linear, Sequential
from nanotorch.tensor import Tensor


@pytest.mark.parametrize("shape,norm_shape", [((4,), (4,)), ((3, 4), (4,)), ((2, 3, 4), (3, 4))])
@pytest.mark.parametrize("affine", [True, False])
def test_values_and_gradients_match_torch(shape, norm_shape, affine):
    rng = np.random.default_rng(8)
    data = rng.normal(size=shape)
    seed = rng.normal(size=shape)
    layer = LayerNorm(norm_shape, eps=1e-4, elementwise_affine=affine)
    ref = torch.nn.LayerNorm(norm_shape, eps=1e-4, elementwise_affine=affine, dtype=torch.float64)
    if affine:
        weights, bias = rng.normal(size=norm_shape), rng.normal(size=norm_shape)
        layer.weight.data[:] = weights
        layer.bias.data[:] = bias
        with torch.no_grad():
            ref.weight.copy_(torch.from_numpy(weights))
            ref.bias.copy_(torch.from_numpy(bias))
    x, tx = Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    y, ty = layer(x), ref(tx)
    y.backward(seed)
    ty.backward(torch.tensor(seed))
    np.testing.assert_allclose(y.data, ty.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-10)
    if affine:
        np.testing.assert_allclose(layer.weight.grad, ref.weight.grad.numpy(), atol=1e-10)
        np.testing.assert_allclose(layer.bias.grad, ref.bias.grad.numpy(), atol=1e-10)
    else:
        assert layer.parameters() == []


def test_constant_input_and_sequential_parameters():
    norm = LayerNorm(3)
    model = Sequential(Linear(2, 3), norm)
    assert len(model.parameters()) == 4
    x = Tensor([[7.0, 7.0, 7.0]], requires_grad=True)
    y = norm(x)
    np.testing.assert_allclose(y.data, 0)
    y.sum().backward()
    np.testing.assert_allclose(x.grad, 0, atol=1e-10)


@pytest.mark.parametrize("bad_shape", [(), (0,), (-1,), (2, 0)])
def test_invalid_normalized_shape(bad_shape):
    with pytest.raises(ValueError):
        LayerNorm(bad_shape)


def test_rejects_input_with_wrong_tail_shape():
    with pytest.raises(ValueError):
        LayerNorm((3, 4))(Tensor(np.zeros((2, 4, 3))))
