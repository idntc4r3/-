"""Repeated backward calls should accumulate only fresh gradients."""

import numpy as np

from nanotorch.tensor import Tensor


def test_repeated_backward_accumulates_leaf_gradients() -> None:
    x = Tensor([2.0, 3.0], requires_grad=True)
    loss = ((x * x) * 3).sum()
    loss.backward()
    np.testing.assert_array_equal(x.grad, [12.0, 18.0])
    loss.backward()
    np.testing.assert_array_equal(x.grad, [24.0, 36.0])


def test_leaf_backward_accumulates_seed() -> None:
    x = Tensor(2.0, requires_grad=True)
    x.backward()
    x.backward()
    np.testing.assert_array_equal(x.grad, 2.0)
