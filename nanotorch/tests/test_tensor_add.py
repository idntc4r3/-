"""Unit tests for Tensor.__add__ (Day 18)."""

import numpy as np
import pytest
from nanotorch.tensor import Tensor


class TestTensorAdd:
    def test_add_simple(self) -> None:
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        c = a + b
        assert np.array_equal(c.data, [4.0, 6.0])
        assert c.requires_grad is True

    def test_add_scalar(self) -> None:
        a = Tensor([1.0, 2.0], requires_grad=True)
        c = a + 5.0
        assert np.array_equal(c.data, [6.0, 7.0])
        assert c.requires_grad is True

    def test_radd_scalar(self) -> None:
        a = Tensor([1.0, 2.0])
        c = 10.0 + a
        assert np.array_equal(c.data, [11.0, 12.0])
