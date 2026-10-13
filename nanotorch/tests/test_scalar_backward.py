"""Unit tests for Value.backward() — full reverse-mode autograd (Day 10).

Every gradient is cross-verified against torch.Tensor via torch.allclose.
Tests cover:
- Leaf-only (single node)
- Simple binary operations (+, *, **, -, /)
- Multi-level chains
- Diamond DAGs (shared nodes)
- Fan-out patterns
- Repeated backward() calls (gradient reset)
- Complex expressions (10+ nodes) vs torch
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


# ---------------------------------------------------------------------------
# 1. Single-node backward
# ---------------------------------------------------------------------------

class TestSingleNode:
    """backward() on a leaf sets its grad to 1.0."""

    def test_leaf_grad_is_one(self) -> None:
        v = Value(5.0)
        v.backward()
        _allclose(v.grad, 1.0, label="leaf grad")

    def test_leaf_data_unchanged(self) -> None:
        v = Value(3.14)
        v.backward()
        assert v.data == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# 2. Resets gradients on each call
# ---------------------------------------------------------------------------

class TestGradientReset:
    """backward() resets all grads to 0 before sweeping — safe to call twice."""

    def test_repeated_backward_gives_same_result(self) -> None:
        a = Value(2.0)
        b = Value(3.0)
        L = a * b + a
        L.backward()
        ga1, gb1 = a.grad, b.grad
        L.backward()
        ga2, gb2 = a.grad, b.grad
        _allclose(ga1, ga2, label="repeated ga")
        _allclose(gb1, gb2, label="repeated gb")

    def test_stale_grad_from_previous_call_cleared(self) -> None:
        a = Value(2.0)
        b = Value(3.0)
        L = a + b
        L.backward()
        assert a.grad == pytest.approx(1.0)
        # Second call must reset a.grad before accumulating
        L.backward()
        assert a.grad == pytest.approx(1.0)  # not 2.0


# ---------------------------------------------------------------------------
# 3. Simple operations vs torch
# ---------------------------------------------------------------------------

class TestSimpleOps:
    """backward() on single-operation graphs matches torch."""

    @pytest.mark.parametrize("av,bv", [
        (2.0, 3.0), (-1.0, 4.0), (0.0, 5.0), (1.5, -2.5),
    ])
    def test_add(self, av: float, bv: float) -> None:
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        (ta + tb).backward()

        va, vb = Value(av), Value(bv)
        (va + vb).backward()

        assert ta.grad is not None and tb.grad is not None
        _allclose(va.grad, ta.grad.item(), label=f"add ga a={av}")
        _allclose(vb.grad, tb.grad.item(), label=f"add gb b={bv}")

    @pytest.mark.parametrize("av,bv", [
        (2.0, 3.0), (-1.0, 4.0), (3.0, -2.0),
    ])
    def test_mul(self, av: float, bv: float) -> None:
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        (ta * tb).backward()

        va, vb = Value(av), Value(bv)
        (va * vb).backward()

        assert ta.grad is not None and tb.grad is not None
        _allclose(va.grad, ta.grad.item(), label=f"mul ga a={av}")
        _allclose(vb.grad, tb.grad.item(), label=f"mul gb b={bv}")

    @pytest.mark.parametrize("xv,n", [
        (2.0, 2), (2.0, 3), (-3.0, 2), (4.0, 0.5), (2.0, -1),
    ])
    def test_pow(self, xv: float, n: float) -> None:
        tx = torch.tensor(xv, dtype=torch.float64, requires_grad=True)
        (tx ** n).backward()

        vx = Value(xv)
        (vx ** n).backward()

        assert tx.grad is not None
        _allclose(vx.grad, tx.grad.item(), label=f"pow gx x={xv} n={n}")

    @pytest.mark.parametrize("av,bv", [
        (5.0, 3.0), (-2.0, 4.0), (1.0, -1.0),
    ])
    def test_sub(self, av: float, bv: float) -> None:
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        (ta - tb).backward()

        va, vb = Value(av), Value(bv)
        (va - vb).backward()

        assert ta.grad is not None and tb.grad is not None
        _allclose(va.grad, ta.grad.item(), label=f"sub ga a={av}")
        _allclose(vb.grad, tb.grad.item(), label=f"sub gb b={bv}")

    @pytest.mark.parametrize("av,bv", [
        (6.0, 2.0), (1.0, 4.0), (-6.0, 3.0),
    ])
    def test_div(self, av: float, bv: float) -> None:
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        (ta / tb).backward()

        va, vb = Value(av), Value(bv)
        (va / vb).backward()

        assert ta.grad is not None and tb.grad is not None
        _allclose(va.grad, ta.grad.item(), label=f"div ga a={av}")
        _allclose(vb.grad, tb.grad.item(), label=f"div gb b={bv}")


# ---------------------------------------------------------------------------
# 4. Multi-level chains
# ---------------------------------------------------------------------------

class TestChains:
    """Multi-node chains propagate gradients end-to-end."""

    def test_affine(self) -> None:
        """L = w*x + b."""
        wv, xv, bv = 2.0, 3.0, 1.0
        tw = torch.tensor(wv, dtype=torch.float64, requires_grad=True)
        tx = torch.tensor(xv, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        (tw * tx + tb).backward()

        vw, vx, vb = Value(wv), Value(xv), Value(bv)
        (vw * vx + vb).backward()

        for (vv, tv, lbl) in [(vw, tw, "w"), (vx, tx, "x"), (vb, tb, "b")]:
            assert tv.grad is not None
            _allclose(vv.grad, tv.grad.item(), label=f"affine grad_{lbl}")

    def test_quadratic(self) -> None:
        """L = a² + 2a + 1  →  ∂L/∂a = 2a + 2."""
        av = 3.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        (ta ** 2 + 2 * ta + 1).backward()

        va = Value(av)
        (va ** 2 + 2 * va + 1).backward()

        assert ta.grad is not None
        _allclose(va.grad, ta.grad.item(), label="quadratic grad_a")

    def test_cubic(self) -> None:
        """L = 3a³ - 2a² + 5a - 7  →  ∂L/∂a = 9a² - 4a + 5."""
        av = 2.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        (3 * ta ** 3 + (-2) * ta ** 2 + 5 * ta + (-7)).backward()

        va = Value(av)
        (3 * va ** 3 + (-2) * va ** 2 + 5 * va + (-7)).backward()

        assert ta.grad is not None
        _allclose(va.grad, ta.grad.item(), label="cubic grad_a")

    def test_rational(self) -> None:
        """L = (a + b) / (a - b)."""
        av, bv = 5.0, 2.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        ((ta + tb) / (ta - tb)).backward()

        va, vb = Value(av), Value(bv)
        ((va + vb) / (va - vb)).backward()

        for (vv, tv, lbl) in [(va, ta, "a"), (vb, tb, "b")]:
            assert tv.grad is not None
            _allclose(vv.grad, tv.grad.item(), label=f"rational grad_{lbl}")


# ---------------------------------------------------------------------------
# 5. Diamond DAGs — shared nodes
# ---------------------------------------------------------------------------

class TestDiamondDag:
    """Nodes used multiple times accumulate gradients correctly."""

    def test_a_plus_a(self) -> None:
        """L = a + a  →  ∂L/∂a = 2."""
        ta = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        (ta + ta).backward()

        va = Value(3.0)
        (va + va).backward()

        assert ta.grad is not None
        _allclose(va.grad, ta.grad.item(), label="a+a grad")

    def test_a_times_a(self) -> None:
        """L = a * a  →  ∂L/∂a = 2a."""
        ta = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        (ta * ta).backward()

        va = Value(3.0)
        (va * va).backward()

        assert ta.grad is not None
        _allclose(va.grad, ta.grad.item(), label="a*a grad")

    def test_diamond(self) -> None:
        """c = a+b; d = a+b; L = c*d  →  each parent counted twice."""
        av, bv = 2.0, 3.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        tc = ta + tb
        td = ta + tb
        (tc * td).backward()

        va, vb = Value(av), Value(bv)
        vc = va + vb
        vd = va + vb
        (vc * vd).backward()

        for (vv, tv, lbl) in [(va, ta, "a"), (vb, tb, "b")]:
            assert tv.grad is not None
            _allclose(vv.grad, tv.grad.item(), label=f"diamond grad_{lbl}")

    def test_shared_intermediate(self) -> None:
        """c = a*b; L = c*c*c  (c reused 3 times)."""
        av, bv = 2.0, 3.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        tc = ta * tb
        (tc * tc * tc).backward()

        va, vb = Value(av), Value(bv)
        vc = va * vb
        (vc * vc * vc).backward()

        for (vv, tv, lbl) in [(va, ta, "a"), (vb, tb, "b")]:
            assert tv.grad is not None
            _allclose(vv.grad, tv.grad.item(), label=f"shared_c^3 grad_{lbl}")


# ---------------------------------------------------------------------------
# 6. Fan-out gradient accumulation
# ---------------------------------------------------------------------------

class TestFanOut:
    """Single leaf contributes to many branches."""

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_sum_of_a_times_bi(self, n: int) -> None:
        """L = a*b0 + a*b1 + … + a*b(n-1)  →  ∂L/∂a = sum(bi)."""
        av = 2.0
        bvs = [float(i + 1) for i in range(n)]

        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tbs = [torch.tensor(b, dtype=torch.float64, requires_grad=True)
               for b in bvs]
        L_t = sum(ta * tb for tb in tbs)  # type: ignore[arg-type]
        assert isinstance(L_t, torch.Tensor)
        L_t.backward()
        assert ta.grad is not None

        va = Value(av)
        vbs = [Value(b) for b in bvs]
        terms = [va * vb for vb in vbs]
        L: Value = terms[0]
        for t in terms[1:]:
            L = L + t
        L.backward()

        _allclose(va.grad, ta.grad.item(), label=f"fan-out n={n} grad_a")


# ---------------------------------------------------------------------------
# 7. Large expression tree (10+ nodes) vs torch
# ---------------------------------------------------------------------------

class TestLargeExpressions:
    """End-to-end backward on complex graphs."""

    def test_norm_and_affine(self) -> None:
        """L = (w*x + b)**2 / ((x**2 + 1)**0.5).

        12 Value nodes; all gradients checked against torch.
        """
        wv, xv, bv = 0.5, 2.0, 0.1
        tw = torch.tensor(wv, dtype=torch.float64, requires_grad=True)
        tx = torch.tensor(xv, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        L_t = (tw * tx + tb) ** 2 / ((tx ** 2 + 1) ** 0.5)
        L_t.backward()

        vw, vx, vb = Value(wv), Value(xv), Value(bv)
        L = (vw * vx + vb) ** 2 / ((vx ** 2 + 1) ** 0.5)
        L.backward()

        for (vv, tv, lbl) in [(vw, tw, "w"), (vx, tx, "x"), (vb, tb, "b")]:
            assert tv.grad is not None
            _allclose(vv.grad, tv.grad.item(), label=f"norm_affine grad_{lbl}")

    def test_bilinear_cross_entropy_like(self) -> None:
        """L = -(a * log_score + b * (1 - log_score))
        approximated as polynomials (no log/exp yet):
        L = -(a * p + b * (1 - p))  where p = sigmoid-like = x/(1+x).
        """
        av, bv, xv = 1.0, 0.0, 2.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        tx = torch.tensor(xv, dtype=torch.float64, requires_grad=True)
        p_t = tx / (1 + tx)   # ≈ sigmoid for x>0
        L_t = -(ta * p_t + tb * (1 - p_t))
        L_t.backward()

        va, vb, vx = Value(av), Value(bv), Value(xv)
        p = vx / (1 + vx)
        L = -(va * p + vb * (1 - p))
        L.backward()

        for (vv, tv, lbl) in [(va, ta, "a"), (vb, tb, "b"), (vx, tx, "x")]:
            assert tv.grad is not None
            _allclose(vv.grad, tv.grad.item(), label=f"ce_like grad_{lbl}")

    def test_10_node_expression(self) -> None:
        """Verify a hand-crafted expression with exactly 10 named nodes.

        L = (a - b)**2 + c * (a + d)**2 / (b**2 + 1)

        Nodes: a, b, c, d, (a-b), (a-b)², (a+d), (a+d)², b², (b²+1),
               c*(a+d)², ..., L  (total > 10 intermediate values)
        """
        av, bv, cv, dv = 1.0, 2.0, 0.5, 3.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        tc = torch.tensor(cv, dtype=torch.float64, requires_grad=True)
        td = torch.tensor(dv, dtype=torch.float64, requires_grad=True)
        L_t = (ta - tb) ** 2 + tc * (ta + td) ** 2 / (tb ** 2 + 1)
        L_t.backward()

        va, vb, vc, vd = Value(av), Value(bv), Value(cv), Value(dv)
        L = (va - vb) ** 2 + vc * (va + vd) ** 2 / (vb ** 2 + 1)
        L.backward()

        for (vv, tv, lbl) in [
            (va, ta, "a"), (vb, tb, "b"), (vc, tc, "c"), (vd, td, "d"),
        ]:
            assert tv.grad is not None
            _allclose(vv.grad, tv.grad.item(), label=f"10node grad_{lbl}")


# ---------------------------------------------------------------------------
# 8. Docstring example sanity check
# ---------------------------------------------------------------------------

class TestDocstringExample:
    """L = a*b + a  (the example from Value.backward() docstring)."""

    def test_forward_value(self) -> None:
        a, b = Value(2.0), Value(3.0)
        L = a * b + a
        assert L.data == pytest.approx(8.0)

    def test_grad_a(self) -> None:
        """∂L/∂a = b + 1 = 4."""
        a, b = Value(2.0), Value(3.0)
        (a * b + a).backward()
        _allclose(a.grad, 4.0, label="docstring grad_a")

    def test_grad_b(self) -> None:
        """∂L/∂b = a = 2."""
        a, b = Value(2.0), Value(3.0)
        (a * b + a).backward()
        _allclose(b.grad, 2.0, label="docstring grad_b")
