"""Regression tests for the gradient of a zero power."""

import numpy as np

from nanotorch.tensor import Tensor


def test_zero_power_has_zero_gradient_at_zero() -> None:
    x = Tensor([0.0, 2.0, -3.0], requires_grad=True)
    y = (x ** 0).sum()

    with np.errstate(divide="raise", invalid="raise"):
        y.backward()

    np.testing.assert_array_equal(y.data, 3.0)
    np.testing.assert_array_equal(x.grad, [0.0, 0.0, 0.0])
