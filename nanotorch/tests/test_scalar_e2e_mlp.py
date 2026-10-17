"""Day 14: End-to-end multi-layer neural network test cross-verified vs PyTorch."""

from __future__ import annotations
import pytest
import torch
from nanotorch.scalar import Value

ATOL = 1e-6
RTOL = 1e-5


def test_2_layer_mlp_vs_torch() -> None:
    """Build a 2-layer MLP (2 -> 4 -> 1) with ReLU and Sigmoid, compare all grads vs PyTorch."""
    # Inputs and target
    x1_val, x2_val, y_true = 0.5, -0.3, 1.0

    # Weights layer 1 (2x4)
    w1_vals = [
        [0.2, -0.4, 0.6, -0.1],
        [-0.5, 0.3, -0.2, 0.7],
    ]
    b1_vals = [0.1, -0.1, 0.05, -0.05]

    # Weights layer 2 (4x1)
    w2_vals = [0.4, -0.8, 0.5, -0.3]
    b2_val = 0.2

    # --- PyTorch forward & backward ---
    x1_t = torch.tensor(x1_val, dtype=torch.float64)
    x2_t = torch.tensor(x2_val, dtype=torch.float64)

    w1_t = [[torch.tensor(w, dtype=torch.float64, requires_grad=True) for w in row] for row in w1_vals]
    b1_t = [torch.tensor(b, dtype=torch.float64, requires_grad=True) for b in b1_vals]

    w2_t = [torch.tensor(w, dtype=torch.float64, requires_grad=True) for w in w2_vals]
    b2_t = torch.tensor(b2_val, dtype=torch.float64, requires_grad=True)

    # Hidden layer: h_j = relu(x1*w1[0][j] + x2*w1[1][j] + b1[j])
    h_t = []
    for j in range(4):
        pre = x1_t * w1_t[0][j] + x2_t * w1_t[1][j] + b1_t[j]
        h_t.append(torch.relu(pre))

    # Output: pred = sigmoid(sum(h_j * w2_j) + b2)
    out_pre_t = sum((h_t[j] * w2_t[j] for j in range(4)), b2_t)
    pred_t = torch.sigmoid(out_pre_t)
    loss_t = (pred_t - y_true) ** 2
    loss_t.backward()

    # --- NanoTorch forward & backward ---
    x1_v = Value(x1_val)
    x2_v = Value(x2_val)

    w1_v = [[Value(w) for w in row] for row in w1_vals]
    b1_v = [Value(b) for b in b1_vals]

    w2_v = [Value(w) for w in w2_vals]
    b2_v = Value(b2_val)

    h_v = []
    for j in range(4):
        pre_v = x1_v * w1_v[0][j] + x2_v * w1_v[1][j] + b1_v[j]
        h_v.append(pre_v.relu())

    out_pre_v = b2_v
    for j in range(4):
        out_pre_v = out_pre_v + h_v[j] * w2_v[j]
    pred_v = out_pre_v.sigmoid()
    loss_v = (pred_v - y_true) ** 2
    loss_v.backward()

    # Compare forward loss
    assert loss_v.data == pytest.approx(loss_t.item(), abs=ATOL)

    # Compare layer 2 weights and bias
    for j in range(4):
        assert w2_t[j].grad is not None
        assert w2_v[j].grad == pytest.approx(w2_t[j].grad.item(), abs=ATOL)
    assert b2_t.grad is not None
    assert b2_v.grad == pytest.approx(b2_t.grad.item(), abs=ATOL)

    # Compare layer 1 weights and biases
    for i in range(2):
        for j in range(4):
            assert w1_t[i][j].grad is not None
            assert w1_v[i][j].grad == pytest.approx(w1_t[i][j].grad.item(), abs=ATOL)
    for j in range(4):
        assert b1_t[j].grad is not None
        assert b1_v[j].grad == pytest.approx(b1_t[j].grad.item(), abs=ATOL)


def test_sgd_step_convergence() -> None:
    """Verify that multiple SGD steps on a scalar neuron decrease loss."""
    w = Value(2.0)
    b = Value(0.5)
    x = Value(1.5)
    y_target = 1.0
    lr = 0.1

    prev_loss = float("inf")
    for _ in range(5):
        pred = (w * x + b).sigmoid()
        loss = (pred - y_target) ** 2
        loss.backward()

        assert loss.data < prev_loss
        prev_loss = loss.data

        # SGD parameter update
        w.data -= lr * w.grad
        b.data -= lr * b.grad
