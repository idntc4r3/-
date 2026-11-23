"""Compare embedding lookups and accumulated gradients with PyTorch."""

import numpy as np
import pytest
import torch

from nanotorch.nn import Embedding, Linear, Sequential


@pytest.mark.parametrize("indices", [3, [1, 1, 4, 0], [[0, 2, 0], [3, 1, 2]], np.array([], dtype=np.int64)])
def test_lookup_and_repeated_index_gradients(indices):
    rng = np.random.default_rng(19)
    weight = rng.normal(size=(5, 3))
    layer = Embedding(5, 3)
    layer.weight.data[:] = weight
    ref = torch.nn.Embedding(5, 3, dtype=torch.float64)
    with torch.no_grad():
        ref.weight.copy_(torch.from_numpy(weight))
    y = layer(indices)
    ty = ref(torch.tensor(indices, dtype=torch.int64))
    seed = rng.normal(size=y.shape)
    y.backward(seed)
    ty.backward(torch.tensor(seed, dtype=torch.float64))
    np.testing.assert_allclose(y.data, ty.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(layer.weight.grad, ref.weight.grad.numpy(), atol=1e-12)


def test_forward_copies_indices_for_backward():
    layer = Embedding(4, 2)
    indices = np.array([0, 1, 1])
    y = layer(indices)
    indices[:] = 3
    y.sum().backward()
    np.testing.assert_array_equal(layer.weight.grad, [[1, 1], [2, 2], [0, 0], [0, 0]])


@pytest.mark.parametrize("indices", [[-1], [3], [1.0], np.array([True])])
def test_rejects_invalid_indices(indices):
    with pytest.raises((TypeError, IndexError)):
        Embedding(3, 2)(indices)


@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (-1, 2)])
def test_rejects_invalid_dimensions(shape):
    with pytest.raises(ValueError):
        Embedding(*shape)


def test_embedding_in_sequential():
    embedding = Embedding(4, 3)
    model = Sequential(embedding, Linear(3, 2))
    assert model([1, 2, 1]).shape == (3, 2)
    assert model.parameters()[0] is embedding.weight
    model([1, 2, 1]).sum().backward()
    assert np.any(embedding.weight.grad[1])
