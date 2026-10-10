"""Unit tests for Value.__pow__ and Value.__neg__ (Day 7).

Gradient formulas under test:
    __pow__:  c = x**n  →  ∂L/∂x = ∂L/∂c · n · x^(n-1)
    __neg__:  c = -x    →  ∂L/∂x = ∂L/∂c · (-1)

All forward values and gradients are cross-verified against torch.Tensor
using torch.allclose (atol=1e-9, rtol=1e-7) and central-difference checks.
"""

from __future__ import annotations

import math
import pytest
import torch
from nanotorch.scalar import Value

ATOL = 1e-9
RTOL = 1e-7


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _allclose(nano: float, ref: float, *, label: str = "") -> None:
    tn = torch.tensor(nano, dtype=torch.float64)
    tr = torch.tensor(ref, dtype=torch.float64)
    assert torch.allclose(tn, tr, atol=ATOL, rtol=RTOL), (
        f"{label}: nano={nano} vs ref={ref}"
    )


def _topo_backward(root: Value) -> None:
    """Topological sort + backward (preview of Day 10)."""
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


# ---------------------------------------------------------------------------
# __pow__ — forward pass
# ---------------------------------------------------------------------------

class TestPowForward:
    """Value.__pow__ forward values match torch."""

    @pytest.mark.parametrize("x,n", [
        (2.0, 0),  (2.0, 1),  (2.0, 2),  (2.0, 3),  (2.0, 4),
        (-3.0, 2), (-3.0, 3), (0.5, 2),  (0.5, 3),
        (4.0, 0.5),           # square root
        (8.0, 1/3),           # cube root
        (2.0, -1),            # reciprocal
        (2.0, -2),            # inverse square
    ])
    def test_forward_value(self, x: float, n: float) -> None:
        tx = torch.tensor(x, dtype=torch.float64)
        ref = float(tx ** n)
        result = Value(x) ** n
        _allclose(result.data, ref, label=f"x={x} n={n}")

    def test_result_is_value(self) -> None:
        assert isinstance(Value(2.0) ** 3, Value)

    def test_op_tag_integer(self) -> None:
        assert (Value(2.0) ** 3)._op == "**3"

    def test_op_tag_float(self) -> None:
        assert (Value(4.0) ** 0.5)._op == "**0.5"

    def test_prev_contains_self(self) -> None:
        x = Value(2.0)
        c = x ** 2
        assert x in c._prev

    def test_prev_length(self) -> None:
        # only one operand for __pow__
        assert len((Value(2.0) ** 3)._prev) == 1

    def test_power_zero(self) -> None:
        """x**0 == 1 for any x."""
        for x in [2.0, -3.0, 0.5]:
            _allclose((Value(x) ** 0).data, 1.0, label=f"x**0 x={x}")

    def test_power_one(self) -> None:
        """x**1 == x."""
        _allclose((Value(7.77) ** 1).data, 7.77, label="x**1")

    def test_type_error_on_value_exponent(self) -> None:
        with pytest.raises(TypeError, match="int/float"):
            Value(2.0) ** Value(3.0)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# __pow__ — backward pass
# ---------------------------------------------------------------------------

class TestPowBackward:
    """Value.__pow__ gradients match torch and finite-difference checks."""

    @pytest.mark.parametrize("x,n", [
        (2.0, 2), (2.0, 3), (2.0, 4),
        (-3.0, 2), (-3.0, 3),
        (0.5, 2), (0.5, 3),
        (3.0, -1), (3.0, -2),
        (4.0, 0.5),
    ])
    def test_grad_vs_torch(self, x: float, n: float) -> None:
        """∂(x**n)/∂x vs torch."""
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        (tx ** n).backward()
        assert tx.grad is not None

        vx = Value(x)
        c = vx ** n
        c.grad = 1.0
        c._backward()
        _allclose(vx.grad, tx.grad.item(), label=f"grad x={x} n={n}")

    @pytest.mark.parametrize("upstream", [2.0, -0.5, 3.0, 0.0])
    def test_upstream_scaling(self, upstream: float) -> None:
        """Gradient scales linearly with upstream gradient."""
        x, n = 2.0, 3
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        (tx ** n).backward(torch.tensor(upstream, dtype=torch.float64))
        assert tx.grad is not None

        vx = Value(x)
        c = vx ** n
        c.grad = upstream
        c._backward()
        _allclose(vx.grad, tx.grad.item(), label=f"upstream={upstream}")

    @pytest.mark.parametrize("x,n", [
        (2.0, 2), (3.0, 3), (2.0, 0.5), (5.0, -1),
    ])
    def test_finite_difference(self, x: float, n: float) -> None:
        """Central-difference check: ∂(x**n)/∂x ≈ [f(x+ε)-f(x-ε)]/(2ε)."""
        eps = 1e-5
        fd = ((Value(x + eps) ** n).data
              - (Value(x - eps) ** n).data) / (2 * eps)
        vx = Value(x)
        (vx ** n).grad = 1.0  # type: ignore[union-attr]
        c = vx ** n
        c.grad = 1.0
        c._backward()
        # Use relaxed tolerance for numerical differentiation
        assert math.isclose(vx.grad, fd, rel_tol=1e-5), (
            f"FD check failed x={x} n={n}: nano={vx.grad} fd={fd}"
        )

    def test_power_two_matches_self_mul(self) -> None:
        """x**2 gradient must equal x*x gradient (both give 2x)."""
        x = 3.0
        vx1, vx2 = Value(x), Value(x)

        c1 = vx1 ** 2
        c1.grad = 1.0
        c1._backward()

        c2 = vx2 * vx2
        c2.grad = 1.0
        c2._backward()

        _allclose(vx1.grad, vx2.grad, label="x**2 vs x*x grad")

    def test_power_three_matches_mul_chain(self) -> None:
        """x**3 gradient must equal x*x*x gradient (both give 3x²)."""
        x = 2.0
        vx1 = Value(x)
        c1 = vx1 ** 3
        c1.grad = 1.0
        c1._backward()

        vx2 = Value(x)
        _topo_backward(vx2 * vx2 * vx2)

        _allclose(vx1.grad, vx2.grad, label="x**3 vs x*x*x grad")


# ---------------------------------------------------------------------------
# __pow__ in expressions
# ---------------------------------------------------------------------------

class TestPowInExpressions:
    """__pow__ composed with add and mul produces correct gradients."""

    def test_sum_of_squares(self) -> None:
        """L = a² + b²  →  ∂L/∂a=2a, ∂L/∂b=2b."""
        av, bv = 3.0, -4.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        (ta ** 2 + tb ** 2).backward()
        assert ta.grad is not None and tb.grad is not None

        va, vb = Value(av), Value(bv)
        _topo_backward(va ** 2 + vb ** 2)

        _allclose(va.grad, ta.grad.item(), label="sum_sq grad_a")
        _allclose(vb.grad, tb.grad.item(), label="sum_sq grad_b")

    def test_mse_like(self) -> None:
        """L = (y_pred - y_true)²  — building block of MSE loss."""
        yp, yt = 2.5, 1.0
        tp = torch.tensor(yp, dtype=torch.float64, requires_grad=True)
        tt = torch.tensor(yt, dtype=torch.float64)
        ((tp - tt) ** 2).backward()
        assert tp.grad is not None

        vp = Value(yp)
        vt = Value(yt)
        _topo_backward((vp + (-1) * vt) ** 2)

        _allclose(vp.grad, tp.grad.item(), label="mse_like grad_yp")

    def test_reciprocal_in_chain(self) -> None:
        """L = a * b**(-1) = a/b  →  ∂L/∂b = -a/b²."""
        av, bv = 6.0, 3.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        (ta * tb ** -1).backward()
        assert ta.grad is not None and tb.grad is not None

        va, vb = Value(av), Value(bv)
        _topo_backward(va * vb ** -1)

        _allclose(va.grad, ta.grad.item(), label="recip grad_a")
        _allclose(vb.grad, tb.grad.item(), label="recip grad_b")

    def test_sqrt_in_chain(self) -> None:
        """L = (a² + b²) ** 0.5  (Euclidean norm)."""
        av, bv = 3.0, 4.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        (ta ** 2 + tb ** 2) ** 0.5
        L_t = (ta ** 2 + tb ** 2) ** 0.5
        L_t.backward()
        assert ta.grad is not None and tb.grad is not None

        va, vb = Value(av), Value(bv)
        _topo_backward((va ** 2 + vb ** 2) ** 0.5)

        _allclose(va.grad, ta.grad.item(), label="norm grad_a")
        _allclose(vb.grad, tb.grad.item(), label="norm grad_b")


# ---------------------------------------------------------------------------
# __neg__ — forward pass
# ---------------------------------------------------------------------------

class TestNegForward:
    """Value.__neg__ forward values."""

    @pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 3.14, -2.71, 1e6, -1e-6])
    def test_negation_value(self, x: float) -> None:
        _allclose((-Value(x)).data, -x, label=f"neg x={x}")

    def test_double_negation(self) -> None:
        """--x == x."""
        x = Value(5.0)
        _allclose((-(-x)).data, 5.0, label="double neg")

    def test_neg_zero(self) -> None:
        """Negation of zero is zero."""
        _allclose((-Value(0.0)).data, 0.0, label="neg zero")

    def test_neg_result_is_value(self) -> None:
        assert isinstance(-Value(1.0), Value)

    def test_neg_op_tag(self) -> None:
        """__neg__ delegates to __mul__(-1) so op tag is '*'."""
        result = -Value(1.0)
        assert result._op == "*"


# ---------------------------------------------------------------------------
# __neg__ — backward pass
# ---------------------------------------------------------------------------

class TestNegBackward:
    """Value.__neg__ gradient = -1 * upstream."""

    @pytest.mark.parametrize("x", [1.0, -2.0, 0.0, 3.14, -0.5])
    def test_grad_vs_torch(self, x: float) -> None:
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        (-tx).backward()
        assert tx.grad is not None

        vx = Value(x)
        c = -vx
        c.grad = 1.0
        c._backward()
        _allclose(vx.grad, tx.grad.item(), label=f"neg grad x={x}")

    @pytest.mark.parametrize("upstream", [1.0, -2.0, 0.5, 0.0])
    def test_upstream_scaling(self, upstream: float) -> None:
        x = 3.0
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        (-tx).backward(torch.tensor(upstream, dtype=torch.float64))
        assert tx.grad is not None

        vx = Value(x)
        c = -vx
        c.grad = upstream
        c._backward()
        _allclose(vx.grad, tx.grad.item(), label=f"neg upstream={upstream}")

    def test_neg_in_expression(self) -> None:
        """L = a - b expressed as a + (-b); ∂L/∂b = -1."""
        av, bv = 5.0, 3.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        (ta + (-tb)).backward()
        assert ta.grad is not None and tb.grad is not None

        va, vb = Value(av), Value(bv)
        _topo_backward(va + (-vb))

        _allclose(va.grad, ta.grad.item(), label="a+(-b) grad_a")
        _allclose(vb.grad, tb.grad.item(), label="a+(-b) grad_b")

    def test_neg_pow_chain(self) -> None:
        """L = -(x**2) + 3*x; ∂L/∂x = -2x + 3."""
        xv = 2.0
        tx = torch.tensor(xv, dtype=torch.float64, requires_grad=True)
        (-(tx ** 2) + 3 * tx).backward()
        assert tx.grad is not None

        vx = Value(xv)
        _topo_backward(-(vx ** 2) + 3 * vx)

        _allclose(vx.grad, tx.grad.item(), label="neg_pow_chain grad")

    def test_double_neg_grad(self) -> None:
        """∂(-(-x))/∂x = 1."""
        tx = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        (-(-tx)).backward()
        assert tx.grad is not None

        vx = Value(2.0)
        _topo_backward(-(-vx))

        _allclose(vx.grad, tx.grad.item(), label="double neg grad")
