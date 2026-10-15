"""Unit tests for Value.sigmoid activation (Day 12)."""

from __future__ import annotations
import pytest
import torch
from nanotorch.scalar import Value

ATOL = 1e-9
RTOL = 1e-7

def _allclose(nano: float, ref: float, *, label: str = "") -> None:
    tn = torch.tensor(nano, dtype=torch.float64)
    tr = torch.tensor(ref, dtype=torch.float64)
    assert torch.allclose(tn, tr, atol=ATOL, rtol=RTOL), f"{label}: nano={nano} vs ref={ref}"


class TestSigmoidForward:
    """Forward pass of sigmoid matches torch.sigmoid."""

    @pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 2.5, -2.5, 10.0, -10.0])
    def test_forward_vs_torch(self, x: float) -> None:
        tx = torch.tensor(x, dtype=torch.float64)
        t_out = torch.sigmoid(tx).item()
        v = Value(x)
        _allclose(v.sigmoid().data, t_out, label=f"sigmoid forward x={x}")

    def test_zero_is_half(self) -> None:
        assert Value(0.0).sigmoid().data == pytest.approx(0.5)

    def test_op_tag(self) -> None:
        v = Value(1.0)
        out = v.sigmoid()
        assert out._op == "sigmoid"
        assert out._prev == frozenset({v})


class TestSigmoidBackward:
    """Backward pass of sigmoid matches torch autograd."""

    @pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 2.5, -2.5])
    def test_grad_vs_torch(self, x: float) -> None:
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        torch.sigmoid(tx).backward()
        vx = Value(x)
        vx.sigmoid().backward()
        assert tx.grad is not None
        _allclose(vx.grad, tx.grad.item(), label=f"sigmoid grad x={x}")

    def test_grad_at_zero(self) -> None:
        """σ'(0) = 0.5 * 0.5 = 0.25."""
        v = Value(0.0)
        v.sigmoid().backward()
        _allclose(v.grad, 0.25, label="sigmoid grad at 0")

    def test_logistic_regression_loss(self) -> None:
        """L = (sigmoid(w*x + b) - y)^2."""
        wv, xv, bv, yv = 1.5, 0.8, -0.2, 1.0
        tw = torch.tensor(wv, dtype=torch.float64, requires_grad=True)
        tx = torch.tensor(xv, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        pred_t = torch.sigmoid(tw * tx + tb)
        loss_t = (pred_t - yv) ** 2
        loss_t.backward()

        vw, vx, vb = Value(wv), Value(xv), Value(bv)
        pred_v = (vw * vx + vb).sigmoid()
        loss_v = (pred_v - yv) ** 2
        loss_v.backward()

        for vv, tv, lbl in [(vw, tw, "w"), (vx, tx, "x"), (vb, tb, "b")]:
            assert tv.grad is not None
            _allclose(vv.grad, tv.grad.item(), label=f"logistic loss {lbl}")
