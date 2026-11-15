"""Compare elementwise logarithms and their gradients with PyTorch."""

import numpy as np
import pytest
import torch

from nanotorch.tensor import Tensor


@pytest.mark.parametrize("data", [2.5, [0.1, 1.0, 7.0], [[0.2, 3.0], [1.1, 10.0]]])
def test_log_forward_and_gradient(data):
    x = Tensor(data, requires_grad=True)
    tx = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    y, ty = x.log(), tx.log()
    np.testing.assert_allclose(y.data, ty.detach().numpy(), rtol=1e-12, atol=1e-12)
    y.sum().backward()
    ty.sum().backward()
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), rtol=1e-12, atol=1e-12)


def test_log_broadcast_chain_and_seed():
    data = np.array([[0.5], [2.0]])
    scale = np.array([[0.5, 1.5, 3.0]])
    seed = np.array([[0.5, -1.0, 2.0], [1.1, 0.3, -0.4]])
    x, w = Tensor(data, requires_grad=True), Tensor(scale, requires_grad=True)
    tx, tw = torch.tensor(data, requires_grad=True), torch.tensor(scale, requires_grad=True)
    y, ty = (x * w).log(), (tx * tw).log()
    y.backward(seed)
    ty.backward(torch.tensor(seed))
    np.testing.assert_allclose(y.data, ty.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(w.grad, tw.grad.numpy(), rtol=1e-12, atol=1e-12)


def test_log_repeated_backward_accumulates_leaf_gradient():
    x = Tensor([0.5, 2.0], requires_grad=True)
    y = x.log()
    y.backward([1.0, 2.0])
    y.backward([-1.0, 4.0])
    np.testing.assert_allclose(x.grad, [0.0, 3.0])


def test_log_without_gradient_tracking():
    x = Tensor([0.5, 2.0])
    y = x.log()
    assert not y.requires_grad
    np.testing.assert_allclose(y.data, np.log(x.data))


def test_log_nonpositive_values_match_pytorch():
    x = Tensor([-1.0, 0.0], requires_grad=True)
    tx = torch.tensor([-1.0, 0.0], dtype=torch.float64, requires_grad=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = x.log()
        y.sum().backward()
    ty = tx.log()
    ty.sum().backward()
    np.testing.assert_allclose(y.data, ty.detach().numpy(), equal_nan=True)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), equal_nan=True)
