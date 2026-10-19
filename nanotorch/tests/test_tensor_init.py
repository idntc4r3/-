"""Unit tests for Tensor class (Day 16)."""

import numpy as np
import pytest
from nanotorch.tensor import Tensor


class TestTensorInit:
    def test_from_list(self) -> None:
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.ndim == 1
        assert t.dtype == np.float64
        assert t.requires_grad is False
        assert t.grad is None

    def test_from_numpy(self) -> None:
        arr = np.array([[1, 2], [3, 4]])
        t = Tensor(arr, requires_grad=True)
        assert t.shape == (2, 2)
        assert t.requires_grad is True

    def test_zeros(self) -> None:
        t = Tensor.zeros(2, 3)
        assert t.shape == (2, 3)
        assert np.all(t.data == 0.0)

    def test_ones(self) -> None:
        t = Tensor.ones(3, 1, requires_grad=True)
        assert t.shape == (3, 1)
        assert np.all(t.data == 1.0)
        assert t.requires_grad is True

    def test_randn(self) -> None:
        t = Tensor.randn(4, 4)
        assert t.shape == (4, 4)
