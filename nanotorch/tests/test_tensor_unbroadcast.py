"""Unit tests for unbroadcast function (Day 17)."""

import numpy as np
import pytest
from nanotorch.tensor import unbroadcast


class TestUnbroadcast:
    def test_exact_match(self) -> None:
        g = np.ones((2, 3))
        res = unbroadcast(g, (2, 3))
        assert res.shape == (2, 3)
        assert np.array_equal(res, g)

    def test_prepended_dimension(self) -> None:
        g = np.ones((2, 3))
        res = unbroadcast(g, (3,))
        assert res.shape == (3,)
        assert np.all(res == 2.0)

    def test_broadcast_axis(self) -> None:
        g = np.ones((3, 4))
        res = unbroadcast(g, (3, 1))
        assert res.shape == (3, 1)
        assert np.all(res == 4.0)

    def test_both_axes(self) -> None:
        g = np.ones((2, 3, 4))
        res = unbroadcast(g, (1, 4))
        assert res.shape == (1, 4)
        assert np.all(res == 6.0)
