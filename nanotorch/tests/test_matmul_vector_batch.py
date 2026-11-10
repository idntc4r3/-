"""Gradient checks for vector and batched matrix multiplication."""
import numpy as np
import pytest
from nanotorch.tensor import Tensor


@pytest.mark.parametrize("sa,sb", [((3,), (3,)), ((3,), (2, 3, 4)),
                                  ((2, 4, 3), (3,)), ((2, 4, 3), (1, 3, 5))])
def test_matmul_gradient(sa, sb):
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=sa), rng.normal(size=sb)
    x, y = Tensor(a.copy(), True), Tensor(b.copy(), True)
    (x @ y).sum().backward()
    for data, grad, other, first in [(a, x.grad, b, True), (b, y.grad, a, False)]:
        for idx in np.ndindex(data.shape):
            old = data[idx]
            data[idx] = old + 1e-6
            plus = np.matmul(data, other).sum() if first else np.matmul(other, data).sum()
            data[idx] = old - 1e-6
            minus = np.matmul(data, other).sum() if first else np.matmul(other, data).sum()
            data[idx] = old
            np.testing.assert_allclose(grad[idx], (plus - minus) / 2e-6, rtol=1e-5, atol=1e-6)
