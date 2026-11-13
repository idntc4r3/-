"""Tests for explicitly seeded reverse-mode differentiation."""
import unittest

import numpy as np

from nanotorch.tensor import Tensor

try:
    import torch
except ImportError:
    torch = None


class TestBackwardSeed(unittest.TestCase):
    def test_weighted_vector_jacobian(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        (x * x).backward([2.0, -1.0, 0.5])
        np.testing.assert_allclose(x.grad, [4.0, -4.0, 3.0])

    def test_accumulates_on_repeated_backward(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = x * x
        y.backward([1.0, 2.0])
        y.backward([3.0, 4.0])
        np.testing.assert_allclose(x.grad, [8.0, 24.0])

    def test_rejects_mismatched_shape_before_changing_gradients(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaises(ValueError):
            (x * x).backward([1.0])
        self.assertIsNone(x.grad)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_matches_pytorch_for_matrix_output(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        seed = [[0.5, -1.0], [2.0, 3.0]]
        x = Tensor(data, requires_grad=True)
        (x * x + x).backward(seed)
        reference = torch.tensor(data, dtype=torch.float64, requires_grad=True)
        (reference * reference + reference).backward(torch.tensor(seed, dtype=torch.float64))
        np.testing.assert_allclose(x.grad, reference.grad.numpy())


if __name__ == "__main__":
    unittest.main()
