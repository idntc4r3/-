"""Unit tests for Tensor matmul forward pass (Day 23)."""

import numpy as np
import pytest
from nanotorch.tensor import Tensor


def test_2d_matmul() -> None:
    a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    b_data = np.array([[2.0, 0.0], [1.0, 2.0]])
    a = Tensor(a_data)
    b = Tensor(b_data)
    c = a @ b

    expected = np.matmul(a_data, b_data)
    assert np.allclose(c.data, expected)
    assert c.shape == (2, 2)


def test_rect_matmul() -> None:
    a = Tensor.randn(3, 4)
    b = Tensor.randn(4, 5)
    c = a @ b
    assert c.shape == (3, 5)
