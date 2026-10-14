"""Unit tests for Value.relu activation (Day 11)."""

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


class TestReluForward:
    """Forward pass of ReLU matches torch.relu."""

    @pytest.mark.parametrize("x", [3.5, 0.1, 100.0, -2.5, -0.01, -100.0, 0.0])
    def test_forward_vs_torch(self, x: float) -> None:
        tx = torch.tensor(x, dtype=torch.float64)
        t_out = torch.relu(tx).item()
        v = Value(x)
        v_out = v.relu()
        _allclose(v_out.data, t_out, label=f"relu forward x={x}")

    def test_op_tag(self) -> None:
        v = Value(2.0)
        out = v.relu()
        assert out._op == "ReLU"
        assert out._prev == frozenset({v})


class TestReluBackward:
    """Backward pass of ReLU matches torch autograd."""

    @pytest.mark.parametrize("x", [2.0, 0.5, 10.0])
    def test_positive_grad(self, x: float) -> None:
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        torch.relu(tx).backward()
        vx = Value(x)
        vx.relu().backward()
        assert tx.grad is not None
        _allclose(vx.grad, tx.grad.item(), label=f"relu grad positive x={x}")
        assert vx.grad == pytest.approx(1.0)

    @pytest.mark.parametrize("x", [-2.0, -0.5, -10.0])
    def test_negative_grad(self, x: float) -> None:
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        torch.relu(tx).backward()
        vx = Value(x)
        vx.relu().backward()
        assert tx.grad is not None
        _allclose(vx.grad, tx.grad.item(), label=f"relu grad negative x={x}")
        assert vx.grad == pytest.approx(0.0)

    def test_zero_grad(self) -> None:
        """PyTorch subgradient at 0 is 0."""
        tx = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        torch.relu(tx).backward()
        vx = Value(0.0)
        vx.relu().backward()
        assert tx.grad is not None
        _allclose(vx.grad, tx.grad.item(), label="relu grad at zero")

    def test_affine_plus_relu(self) -> None:
        """L = relu(w*x + b)."""
        wv, xv, bv = 2.0, -1.5, 4.0  # 2*(-1.5) + 4 = 1.0 > 0
        tw = torch.tensor(wv, dtype=torch.float64, requires_grad=True)
        tx = torch.tensor(xv, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        torch.relu(tw * tx + tb).backward()

        vw, vx, vb = Value(wv), Value(xv), Value(bv)
        (vw * vx + vb).relu().backward()

        for vv, tv, lbl in [(vw, tw, "w"), (vx, tx, "x"), (vb, tb, "b")]:
            assert tv.grad is not None
            _allclose(vv.grad, tv.grad.item(), label=f"affine relu {lbl}")
