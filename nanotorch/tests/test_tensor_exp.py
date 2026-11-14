"""Forward and backward checks for the elementwise exponential."""

import numpy as np
import pytest
import torch

from nanotorch.tensor import Tensor


@pytest.mark.parametrize("data", [0.0, [-2.0, 0.0, 1.5], [[-1.0], [2.0]]])
def test_exp_forward_and_gradient_match_pytorch(data):
    x = Tensor(data, requires_grad=True)
    tx = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    y = x.exp()
    ty = tx.exp()
    np.testing.assert_allclose(y.data, ty.detach().numpy(), rtol=1e-12)
    y.sum().backward()
    ty.sum().backward()
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), rtol=1e-12)


def test_exp_chain_with_broadcasting_and_seed():
    data = np.array([[-0.8], [0.7]])
    weights = np.array([[1.0, -2.0, 0.4]])
    seed = np.array([[0.5, 1.0, -1.0], [2.0, -0.3, 0.2]])
    x, w = Tensor(data, requires_grad=True), Tensor(weights, requires_grad=True)
    tx = torch.tensor(data, requires_grad=True)
    tw = torch.tensor(weights, requires_grad=True)
    out = (x * w).exp()
    expected = (tx * tw).exp()
    out.backward(seed)
    expected.backward(torch.tensor(seed))
    np.testing.assert_allclose(out.data, expected.detach().numpy(), rtol=1e-12)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), rtol=1e-12)
    np.testing.assert_allclose(w.grad, tw.grad.numpy(), rtol=1e-12)


def test_exp_without_grad_tracking():
    x = Tensor([0.0, 1.0])
    y = x.exp()
    assert not y.requires_grad
    np.testing.assert_allclose(y.data, np.exp(x.data))
