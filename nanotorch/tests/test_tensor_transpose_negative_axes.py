"""Transpose gradients must handle negative axis indices."""
import numpy as np

from nanotorch.tensor import Tensor


def test_transpose_negative_axes_backward():
    x = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)
    weights = np.arange(24).reshape(4, 2, 3)
    (x.transpose(-1, 0, 1) * weights).sum().backward()
    np.testing.assert_array_equal(x.grad, weights.transpose(1, 2, 0))
