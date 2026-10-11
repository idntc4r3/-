"""Unit tests for Value.__sub__, __rsub__, __truediv__, __rtruediv__ (Day 8).

Implementation strategy under test:
    a - b  = a + (-b)        → no new backward closure
    a / b  = a * b**(-1)     → no new backward closure

All values and gradients are cross-verified against torch.Tensor via
torch.allclose (atol=1e-9, rtol=1e-7).
"""

from __future__ import annotations

import pytest
import torch
from nanotorch.scalar import Value

ATOL = 1e-9
RTOL = 1e-7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allclose(nano: float, ref: float, *, label: str = "") -> None:
    tn = torch.tensor(nano, dtype=torch.float64)
    tr = torch.tensor(ref, dtype=torch.float64)
    assert torch.allclose(tn, tr, atol=ATOL, rtol=RTOL), (
        f"{label}: nano={nano} vs ref={ref}"
    )


def _topo_backward(root: Value) -> None:
    """Full topological backward sweep."""
    order: list[Value] = []
    seen: set[int] = set()

    def _dfs(v: Value) -> None:
        if id(v) not in seen:
            seen.add(id(v))
            for c in v._prev:
                _dfs(c)
            order.append(v)

    _dfs(root)
    root.grad = 1.0
    for node in reversed(order):
        node._backward()


def _torch_sub(a: float, b: float) -> tuple[float, float, float]:
    ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    (ta - tb).backward()
    assert ta.grad is not None and tb.grad is not None
    return (ta - tb).detach().item(), ta.grad.item(), tb.grad.item()


def _torch_div(a: float, b: float) -> tuple[float, float, float]:
    ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    L = ta / tb
    L.backward()
    assert ta.grad is not None and tb.grad is not None
    return L.detach().item(), ta.grad.item(), tb.grad.item()


# ---------------------------------------------------------------------------
# 1. __sub__ — forward
# ---------------------------------------------------------------------------

class TestSubForward:
    """Value.__sub__ forward values match torch."""

    @pytest.mark.parametrize("a,b", [
        (5.0, 3.0), (0.0, 0.0), (-1.0, -1.0),
        (1e6, 1e-6), (-3.0, 4.0), (0.5, 1.5),
    ])
    def test_value_minus_value(self, a: float, b: float) -> None:
        expected = a - b
        result = Value(a) - Value(b)
        _allclose(result.data, expected, label=f"a={a} b={b}")

    def test_result_is_value(self) -> None:
        assert isinstance(Value(5.0) - Value(3.0), Value)

    def test_sub_self_is_zero(self) -> None:
        v = Value(7.0)
        _allclose((v - v).data, 0.0, label="v-v")

    def test_sub_zero(self) -> None:
        v = Value(3.14)
        _allclose((v - Value(0.0)).data, 3.14, label="v-0")

    def test_sub_commutative_asymmetry(self) -> None:
        """a - b ≠ b - a (non-commutative)."""
        a, b = Value(5.0), Value(3.0)
        assert (a - b).data != (b - a).data

    def test_value_minus_int(self) -> None:
        _allclose((Value(7.0) - 2).data, 5.0, label="Value-int")

    def test_value_minus_float(self) -> None:
        _allclose((Value(7.0) - 2.5).data, 4.5, label="Value-float")


# ---------------------------------------------------------------------------
# 2. __sub__ — backward
# ---------------------------------------------------------------------------

class TestSubBackward:
    """Value.__sub__ gradients: ∂L/∂a = 1, ∂L/∂b = -1."""

    @pytest.mark.parametrize("a,b", [
        (5.0, 3.0), (-1.0, -4.0), (0.0, 0.0), (1.0, -1.0),
    ])
    def test_grad_a(self, a: float, b: float) -> None:
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
        (ta - tb).backward()
        assert ta.grad is not None

        va, vb = Value(a), Value(b)
        _topo_backward(va - vb)
        _allclose(va.grad, ta.grad.item(), label=f"grad_a a={a} b={b}")

    @pytest.mark.parametrize("a,b", [
        (5.0, 3.0), (-1.0, -4.0), (0.0, 0.0), (1.0, -1.0),
    ])
    def test_grad_b(self, a: float, b: float) -> None:
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
        (ta - tb).backward()
        assert tb.grad is not None

        va, vb = Value(a), Value(b)
        _topo_backward(va - vb)
        _allclose(vb.grad, tb.grad.item(), label=f"grad_b a={a} b={b}")

    def test_grad_a_is_plus_one(self) -> None:
        """∂(a-b)/∂a == +1 always."""
        va, vb = Value(3.0), Value(1.0)
        _topo_backward(va - vb)
        _allclose(va.grad, 1.0, label="grad_a == +1")

    def test_grad_b_is_minus_one(self) -> None:
        """∂(a-b)/∂b == -1 always."""
        va, vb = Value(3.0), Value(1.0)
        _topo_backward(va - vb)
        _allclose(vb.grad, -1.0, label="grad_b == -1")

    @pytest.mark.parametrize("upstream", [2.0, -0.5, 0.0, 10.0])
    def test_upstream_scaling(self, upstream: float) -> None:
        ta = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        (ta - tb).backward(torch.tensor(upstream, dtype=torch.float64))
        assert ta.grad is not None and tb.grad is not None

        va, vb = Value(3.0), Value(1.0)
        c = va - vb
        c.grad = upstream
        # manual topo for just the top node — sub is 2 nodes deep
        c._backward()
        neg_b = next(n for n in c._prev if n is not va)
        neg_b._backward()

        _allclose(va.grad, ta.grad.item(), label=f"up={upstream} grad_a")
        _allclose(vb.grad, tb.grad.item(), label=f"up={upstream} grad_b")


# ---------------------------------------------------------------------------
# 3. __rsub__ — forward + backward
# ---------------------------------------------------------------------------

class TestRSubForward:
    """scalar - Value reflected subtraction."""

    @pytest.mark.parametrize("scalar,val", [
        (5, 3.0), (0, 1.0), (-1, 2.0), (10, -4.0),
    ])
    def test_int_minus_value(self, scalar: int, val: float) -> None:
        _allclose((scalar - Value(val)).data, scalar - val,
                  label=f"{scalar}-Value({val})")

    def test_float_minus_value(self) -> None:
        _allclose((3.5 - Value(1.5)).data, 2.0, label="float-Value")

    def test_rsub_non_commutative(self) -> None:
        v = Value(3.0)
        _allclose((5 - v).data, 2.0, label="5-v")
        _allclose((v - 5).data, -2.0, label="v-5")

    def test_rsub_backward(self) -> None:
        """5 - v: ∂L/∂v = -1."""
        tv = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        (5.0 - tv).backward()
        assert tv.grad is not None

        vv = Value(3.0)
        _topo_backward(Value(5.0) - vv)
        _allclose(vv.grad, tv.grad.item(), label="rsub grad_v")


# ---------------------------------------------------------------------------
# 4. __truediv__ — forward
# ---------------------------------------------------------------------------

class TestDivForward:
    """Value.__truediv__ forward values match torch."""

    @pytest.mark.parametrize("a,b", [
        (6.0, 2.0), (1.0, 4.0), (-6.0, 3.0),
        (5.0, -2.0), (0.0, 3.0), (1.0, 3.0),
        (1e6, 1e3), (1.0, 1e-3),
    ])
    def test_value_div_value(self, a: float, b: float) -> None:
        expected = a / b
        result = Value(a) / Value(b)
        _allclose(result.data, expected, label=f"a={a} b={b}")

    def test_result_is_value(self) -> None:
        assert isinstance(Value(6.0) / Value(2.0), Value)

    def test_div_by_one(self) -> None:
        v = Value(7.77)
        _allclose((v / Value(1.0)).data, 7.77, label="v/1")

    def test_div_self_is_one(self) -> None:
        v = Value(5.0)
        _allclose((v / v).data, 1.0, label="v/v")

    def test_zero_div_nonzero(self) -> None:
        _allclose((Value(0.0) / Value(3.0)).data, 0.0, label="0/3")

    def test_value_div_int(self) -> None:
        _allclose((Value(6.0) / 2).data, 3.0, label="Value/int")

    def test_value_div_float(self) -> None:
        _allclose((Value(1.0) / 0.5).data, 2.0, label="Value/float")


# ---------------------------------------------------------------------------
# 5. __truediv__ — backward
# ---------------------------------------------------------------------------

class TestDivBackward:
    """Value.__truediv__ gradients match torch.allclose."""

    @pytest.mark.parametrize("a,b", [
        (6.0, 2.0), (1.0, 4.0), (-6.0, 3.0),
        (5.0, -2.0), (1.0, 3.0),
    ])
    def test_grad_a(self, a: float, b: float) -> None:
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
        (ta / tb).backward()
        assert ta.grad is not None

        va, vb = Value(a), Value(b)
        _topo_backward(va / vb)
        _allclose(va.grad, ta.grad.item(), label=f"div grad_a a={a} b={b}")

    @pytest.mark.parametrize("a,b", [
        (6.0, 2.0), (1.0, 4.0), (-6.0, 3.0),
        (5.0, -2.0), (1.0, 3.0),
    ])
    def test_grad_b(self, a: float, b: float) -> None:
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
        (ta / tb).backward()
        assert tb.grad is not None

        va, vb = Value(a), Value(b)
        _topo_backward(va / vb)
        _allclose(vb.grad, tb.grad.item(), label=f"div grad_b a={a} b={b}")

    def test_grad_formula_a(self) -> None:
        """∂(a/b)/∂a = 1/b."""
        a, b = 6.0, 3.0
        va, vb = Value(a), Value(b)
        _topo_backward(va / vb)
        _allclose(va.grad, 1.0 / b, label="grad_a = 1/b")

    def test_grad_formula_b(self) -> None:
        """∂(a/b)/∂b = -a/b²."""
        a, b = 6.0, 3.0
        va, vb = Value(a), Value(b)
        _topo_backward(va / vb)
        _allclose(vb.grad, -a / (b ** 2), label="grad_b = -a/b²")

    @pytest.mark.parametrize("upstream", [2.0, -1.0, 0.5, 0.0])
    def test_upstream_scaling(self, upstream: float) -> None:
        a, b = 4.0, 2.0
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
        (ta / tb).backward(torch.tensor(upstream, dtype=torch.float64))
        assert ta.grad is not None and tb.grad is not None

        va, vb = Value(a), Value(b)
        L = va / vb
        L.grad = upstream
        # manually unroll topo for the div sub-graph
        nodes: list[Value] = []
        seen: set[int] = set()
        def _dfs(v: Value) -> None:
            if id(v) not in seen:
                seen.add(id(v))
                for c in v._prev:
                    _dfs(c)
                nodes.append(v)
        _dfs(L)
        for n in reversed(nodes):
            n._backward()

        _allclose(va.grad, ta.grad.item(), label=f"up={upstream} grad_a")
        _allclose(vb.grad, tb.grad.item(), label=f"up={upstream} grad_b")


# ---------------------------------------------------------------------------
# 6. __rtruediv__ — forward + backward
# ---------------------------------------------------------------------------

class TestRDivForward:
    """scalar / Value reflected division."""

    @pytest.mark.parametrize("scalar,val", [
        (6, 2.0), (1, 4.0), (0, 3.0), (-6, 3.0),
    ])
    def test_int_div_value(self, scalar: int, val: float) -> None:
        _allclose((scalar / Value(val)).data, scalar / val,
                  label=f"{scalar}/Value({val})")

    def test_float_div_value(self) -> None:
        _allclose((1.0 / Value(4.0)).data, 0.25, label="1.0/Value(4.0)")

    def test_rdiv_backward(self) -> None:
        """6 / v: ∂L/∂v = -6/v²."""
        tv = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        (6.0 / tv).backward()
        assert tv.grad is not None

        vv = Value(2.0)
        _topo_backward(Value(6.0) / vv)
        _allclose(vv.grad, tv.grad.item(), label="rdiv grad_v")


# ---------------------------------------------------------------------------
# 7. Complex expressions mixing sub + div
# ---------------------------------------------------------------------------

class TestMixedSubDiv:
    """Composite expressions with sub and div vs torch."""

    def test_slope_formula(self) -> None:
        """slope = (y2 - y1) / (x2 - x1)."""
        x1, y1, x2, y2 = 1.0, 2.0, 4.0, 8.0
        tx1 = torch.tensor(x1, dtype=torch.float64, requires_grad=True)
        ty1 = torch.tensor(y1, dtype=torch.float64, requires_grad=True)
        tx2 = torch.tensor(x2, dtype=torch.float64, requires_grad=True)
        ty2 = torch.tensor(y2, dtype=torch.float64, requires_grad=True)
        L_t = (ty2 - ty1) / (tx2 - tx1)
        L_t.backward()

        vx1, vy1 = Value(x1), Value(y1)
        vx2, vy2 = Value(x2), Value(y2)
        _topo_backward((vy2 - vy1) / (vx2 - vx1))

        _allclose(vy2.grad, ty2.grad.item(), label="slope grad_y2")  # type: ignore[union-attr]
        _allclose(vy1.grad, ty1.grad.item(), label="slope grad_y1")  # type: ignore[union-attr]
        _allclose(vx2.grad, tx2.grad.item(), label="slope grad_x2")  # type: ignore[union-attr]
        _allclose(vx1.grad, tx1.grad.item(), label="slope grad_x1")  # type: ignore[union-attr]

    def test_mean_of_two(self) -> None:
        """mean = (a + b) / 2."""
        a, b = 3.0, 7.0
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
        ((ta + tb) / 2).backward()
        assert ta.grad is not None and tb.grad is not None

        va, vb = Value(a), Value(b)
        _topo_backward((va + vb) / 2)

        _allclose(va.grad, ta.grad.item(), label="mean grad_a")
        _allclose(vb.grad, tb.grad.item(), label="mean grad_b")

    def test_variance_like(self) -> None:
        """L = (a - mu)² / n  — variance of one element."""
        a, mu, n = 3.0, 2.0, 4.0
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tmu = torch.tensor(mu, dtype=torch.float64, requires_grad=True)
        ((ta - tmu) ** 2 / n).backward()
        assert ta.grad is not None and tmu.grad is not None

        va, vmu = Value(a), Value(mu)
        _topo_backward((va - vmu) ** 2 / n)

        _allclose(va.grad, ta.grad.item(), label="var grad_a")
        _allclose(vmu.grad, tmu.grad.item(), label="var grad_mu")

    def test_normalize(self) -> None:
        """L = (x - mean) / std  — z-score normalization."""
        x, mean, std = 5.0, 2.0, 1.5
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        tmean = torch.tensor(mean, dtype=torch.float64, requires_grad=True)
        tstd = torch.tensor(std, dtype=torch.float64, requires_grad=True)
        ((tx - tmean) / tstd).backward()
        assert tx.grad is not None and tmean.grad is not None and tstd.grad is not None

        vx, vmean, vstd = Value(x), Value(mean), Value(std)
        _topo_backward((vx - vmean) / vstd)

        _allclose(vx.grad, tx.grad.item(), label="norm grad_x")
        _allclose(vmean.grad, tmean.grad.item(), label="norm grad_mean")
        _allclose(vstd.grad, tstd.grad.item(), label="norm grad_std")

    def test_chain_sub_div_pow(self) -> None:
        """L = ((a - b) / c) ** 2."""
        a, b, c = 5.0, 2.0, 3.0
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
        tc = torch.tensor(c, dtype=torch.float64, requires_grad=True)
        (((ta - tb) / tc) ** 2).backward()
        assert ta.grad is not None and tb.grad is not None and tc.grad is not None

        va, vb, vc = Value(a), Value(b), Value(c)
        _topo_backward(((va - vb) / vc) ** 2)

        _allclose(va.grad, ta.grad.item(), label="chain grad_a")
        _allclose(vb.grad, tb.grad.item(), label="chain grad_b")
        _allclose(vc.grad, tc.grad.item(), label="chain grad_c")
