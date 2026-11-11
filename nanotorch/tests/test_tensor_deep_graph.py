"""Backward traversal supports deep and shared computation graphs."""
import numpy as np

from nanotorch.tensor import Tensor


def test_backward_handles_deep_graph():
    x = Tensor(2.0, requires_grad=True)
    result = x
    for _ in range(2000):
        result = result + 1
    result.backward()
    np.testing.assert_allclose(x.grad, 1.0)


def test_backward_handles_shared_intermediates():
    x = Tensor(2.0, requires_grad=True)
    shared = x * x
    result = shared * shared + shared
    result.backward()
    np.testing.assert_allclose(x.grad, 36.0)
