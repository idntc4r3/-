"""Unit tests for Value.tanh activation (Day 13)."""

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


class TestTanhForward:
    """Forward pass of tanh matches torch.tanh."""

    @pytest.mark.parametrize("x", [0.0, 0.5, -0.5, 1.0, -1.0, 3.0, -3.0])
    def test_forward_vs_torch(self, x: float) -> None:
        tx = torch.tensor(x, dtype=torch.float64)
        t_out = torch.tanh(tx).item()
        v = Value(x)
        _allclose(v.tanh().data, t_out, label=f"tanh forward x={x}")

    def test_zero_is_zero(self) -> None:
        assert Value(0.0).tanh().data == pytest.approx(0.0)

    def test_op_tag(self) -> None:
        v = Value(0.5)
        out = v.tanh()
        assert out._op == "tanh"
        assert out._prev == frozenset({v})


class TestTanhBackward:
    """Backward pass of tanh matches torch autograd."""

    @pytest.mark.parametrize("x", [0.0, 0.5, -0.5, 1.5, -1.5])
    def test_grad_vs_torch(self, x: float) -> None:
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        torch.tanh(tx).backward()
        vx = Value(x)
        vx.tanh().backward()
        assert tx.grad is not None
        _allclose(vx.grad, tx.grad.item(), label=f"tanh grad x={x}")

    def test_grad_at_zero(self) -> None:
        """tanh'(0) = 1 - 0 = 1."""
        v = Value(0.0)
        v.tanh().backward()
        _allclose(v.grad, 1.0, label="tanh grad at 0")

    def test_deep_tanh_network(self) -> None:
        """L = tanh(tanh(w*x + b))."""
        wv, xv, bv = 0.7, 1.2, -0.3
        tw = torch.tensor(wv, dtype=torch.float64, requires_grad=True)
        tx = torch.tensor(xv, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        h_t = torch.tanh(tw * tx + tb)
        out_t = torch.tanh(h_t)
        out_t.backward()

        vw, vx, vb = Value(wv), Value(xv), Value(bv)
        h_v = (vw * vx + vb).tanh()
        out_v = h_v.tanh()
        out_v.backward()

        for vv, tv, lbl in [(vw, tw, "w"), (vx, tx, "x"), (vb, tb, "b")]:
            assert tv.grad is not None
            _allclose(vv.grad, tv.grad.item(), label=f"deep tanh {lbl}")
