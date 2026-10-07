"""Unit tests for scalar multiplication: Value.__mul__ and __rmul__ (Day 5).

Every forward value and gradient is cross-checked against torch.Tensor.

Gradient formulas under test:
    c = a * b
    ∂L/∂a = ∂L/∂c · b
    ∂L/∂b = ∂L/∂c · a

Special cases:
    c = a * a  →  ∂L/∂a = ∂L/∂c · 2a   (gradient accumulates twice)
    c = a * k  →  ∂L/∂a = ∂L/∂c · k    (k is a constant scalar)
"""

from __future__ import annotations

import pytest
import torch
from nanotorch.scalar import Value

# ---------------------------------------------------------------------------
# Tolerance (float64 throughout)
# ---------------------------------------------------------------------------
ATOL = 1e-9
RTOL = 1e-7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allclose(nano: float, ref: float, *, label: str = "") -> None:
    t_n = torch.tensor(nano, dtype=torch.float64)
    t_r = torch.tensor(ref, dtype=torch.float64)
    assert torch.allclose(t_n, t_r, atol=ATOL, rtol=RTOL), (
        f"{label}: nano={nano} vs ref={ref}"
    )


def _topo_backward(root: Value) -> None:
    """Run full topological backward (preview of Day 10)."""
    nodes: list[Value] = []
    visited: set[int] = set()

    def _visit(v: Value) -> None:
        if id(v) not in visited:
            visited.add(id(v))
            for c in v._prev:
                _visit(c)
            nodes.append(v)

    _visit(root)
    root.grad = 1.0
    for n in reversed(nodes):
        n._backward()


def _torch_mul(a: float, b: float) -> tuple[float, float, float]:
    """Return (forward, grad_a, grad_b) from torch for c = a * b."""
    ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    tc = ta * tb
    tc.backward()
    assert ta.grad is not None and tb.grad is not None
    return tc.detach().item(), ta.grad.item(), tb.grad.item()


# ---------------------------------------------------------------------------
# 1. Forward pass
# ---------------------------------------------------------------------------

class TestMulForward:
    """Value.__mul__ forward value matches torch."""

    @pytest.mark.parametrize("a,b", [
        (2.0, 3.0),
        (-1.0, 4.0),
        (0.0, 99.0),
        (1.0, 1.0),
        (-3.5, -2.0),
        (1e6, 1e-6),
        (0.5, 0.5),
    ])
    def test_forward_value(self, a: float, b: float) -> None:
        ref, _, _ = _torch_mul(a, b)
        result = Value(a) * Value(b)
        _allclose(result.data, ref, label=f"fwd a={a} b={b}")

    def test_result_is_value(self) -> None:
        assert isinstance(Value(2.0) * Value(3.0), Value)

    def test_op_tag(self) -> None:
        assert (Value(2.0) * Value(3.0))._op == "*"

    def test_prev_contains_both(self) -> None:
        a, b = Value(2.0), Value(3.0)
        c = a * b
        assert a in c._prev and b in c._prev

    def test_zero_times_anything(self) -> None:
        assert (Value(0.0) * Value(999.0)).data == 0.0

    def test_identity_one(self) -> None:
        v = Value(7.77)
        _allclose((v * Value(1.0)).data, 7.77, label="identity")

    def test_commutativity_forward(self) -> None:
        a, b = 2.5, 3.7
        _allclose((Value(a) * Value(b)).data,
                  (Value(b) * Value(a)).data,
                  label="commutative")


# ---------------------------------------------------------------------------
# 2. Backward pass — single step
# ---------------------------------------------------------------------------

class TestMulBackward:
    """Gradients after a single _backward() call match torch."""

    @pytest.mark.parametrize("a,b", [
        (2.0, 3.0), (-4.0, 7.0), (0.0, 5.0),
        (1.0, -1.0), (100.0, -200.0), (0.5, 0.5),
    ])
    def test_grad_a(self, a: float, b: float) -> None:
        _, ref_ga, _ = _torch_mul(a, b)
        va, vb = Value(a), Value(b)
        vc = va * vb
        vc.grad = 1.0
        vc._backward()
        _allclose(va.grad, ref_ga, label=f"grad_a a={a} b={b}")

    @pytest.mark.parametrize("a,b", [
        (2.0, 3.0), (-4.0, 7.0), (0.0, 5.0),
        (1.0, -1.0), (100.0, -200.0), (0.5, 0.5),
    ])
    def test_grad_b(self, a: float, b: float) -> None:
        _, _, ref_gb = _torch_mul(a, b)
        va, vb = Value(a), Value(b)
        vc = va * vb
        vc.grad = 1.0
        vc._backward()
        _allclose(vb.grad, ref_gb, label=f"grad_b a={a} b={b}")

    @pytest.mark.parametrize("upstream", [2.0, -3.0, 0.5, 0.0, 1e3])
    def test_arbitrary_upstream(self, upstream: float) -> None:
        """Gradients scale correctly with arbitrary upstream gradient."""
        a, b = 3.0, 4.0
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
        tc = ta * tb
        tc.backward(torch.tensor(upstream, dtype=torch.float64))
        assert ta.grad is not None and tb.grad is not None

        va, vb = Value(a), Value(b)
        vc = va * vb
        vc.grad = upstream
        vc._backward()

        _allclose(va.grad, ta.grad.item(), label=f"upstream={upstream} grad_a")
        _allclose(vb.grad, tb.grad.item(), label=f"upstream={upstream} grad_b")


# ---------------------------------------------------------------------------
# 3. Self-multiplication: a * a  →  grad = 2a
# ---------------------------------------------------------------------------

class TestSelfMultiplication:
    """c = a * a should give ∂L/∂a = 2a via gradient accumulation."""

    @pytest.mark.parametrize("a", [-3.0, -1.0, 0.0, 1.0, 2.0, 5.0])
    def test_self_mul_grad(self, a: float) -> None:
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        (ta * ta).backward()
        assert ta.grad is not None

        va = Value(a)
        vc = va * va
        vc.grad = 1.0
        vc._backward()
        _allclose(va.grad, ta.grad.item(), label=f"a*a grad a={a}")

    def test_self_mul_forward(self) -> None:
        _allclose((Value(4.0) * Value(4.0)).data, 16.0, label="4*4")


# ---------------------------------------------------------------------------
# 4. Scalar (int / float) operands
# ---------------------------------------------------------------------------

class TestMulWithScalar:
    """Value * scalar and scalar * Value work for forward and backward."""

    @pytest.mark.parametrize("scalar,val", [
        (2, 3.0), (-1, 5.0), (0, 7.0), (3, -2.5), (100, 0.01),
    ])
    def test_value_times_int_forward(self, scalar: int, val: float) -> None:
        _allclose((Value(val) * scalar).data, val * scalar,
                  label=f"Value*{scalar}")

    @pytest.mark.parametrize("scalar,val", [
        (0.5, 4.0), (-2.0, 3.0), (1.0, -1.0),
    ])
    def test_value_times_float_forward(self, scalar: float, val: float) -> None:
        _allclose((Value(val) * scalar).data, val * scalar,
                  label=f"Value*{scalar}")

    def test_int_times_value_rmul(self) -> None:
        v = Value(3.0)
        result = 2 * v
        _allclose(result.data, 6.0, label="2*Value fwd")

    def test_float_times_value_rmul(self) -> None:
        v = Value(4.0)
        result = 0.5 * v
        _allclose(result.data, 2.0, label="0.5*Value fwd")

    def test_scalar_mul_backward_grad_v(self) -> None:
        """3 * v: ∂L/∂v = 3."""
        tv = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        (3 * tv).backward()
        assert tv.grad is not None

        vv = Value(5.0)
        out = 3 * vv
        out.grad = 1.0
        out._backward()
        _allclose(vv.grad, tv.grad.item(), label="3*Value grad")

    def test_value_times_zero_grad(self) -> None:
        """v * 0: gradient of v = 0 (other.data = 0)."""
        tv = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        (tv * 0).backward()
        assert tv.grad is not None

        vv = Value(5.0)
        out = vv * 0
        out.grad = 1.0
        out._backward()
        _allclose(vv.grad, tv.grad.item(), label="v*0 grad")


# ---------------------------------------------------------------------------
# 5. Mixed add + mul expression chains
# ---------------------------------------------------------------------------

class TestMixedExpressions:
    """Chains mixing addition and multiplication."""

    def test_affine(self) -> None:
        """L = w * x + b  (classic neuron output, bias not trained here)."""
        w_v, x_v, b_v = 2.0, 3.0, 1.0

        tw = torch.tensor(w_v, dtype=torch.float64, requires_grad=True)
        tx = torch.tensor(x_v, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(b_v, dtype=torch.float64, requires_grad=True)
        L_t = tw * tx + tb
        L_t.backward()
        assert tw.grad is not None and tx.grad is not None and tb.grad is not None

        vw, vx, vb = Value(w_v), Value(x_v), Value(b_v)
        L = vw * vx + vb
        _topo_backward(L)

        _allclose(L.data, L_t.detach().item(), label="affine fwd")
        _allclose(vw.grad, tw.grad.item(), label="affine grad_w")
        _allclose(vx.grad, tx.grad.item(), label="affine grad_x")
        _allclose(vb.grad, tb.grad.item(), label="affine grad_b")

    def test_quadratic(self) -> None:
        """L = a*a + 2*a + 1  →  ∂L/∂a = 2a + 2."""
        a_v = 3.0
        ta = torch.tensor(a_v, dtype=torch.float64, requires_grad=True)
        L_t = ta * ta + 2 * ta + 1
        L_t.backward()
        assert ta.grad is not None

        va = Value(a_v)
        L = va * va + 2 * va + 1
        _topo_backward(L)

        _allclose(L.data, L_t.detach().item(), label="quadratic fwd")
        _allclose(va.grad, ta.grad.item(), label="quadratic grad_a")

    def test_dot_product(self) -> None:
        """L = a1*b1 + a2*b2 + a3*b3 (3-dim dot product)."""
        avs = [1.0, -2.0, 3.0]
        bvs = [4.0, 5.0, -6.0]

        t_as = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in avs]
        t_bs = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in bvs]
        L_t = sum(a * b for a, b in zip(t_as, t_bs))  # type: ignore[arg-type]
        assert isinstance(L_t, torch.Tensor)
        L_t.backward()

        v_as = [Value(v) for v in avs]
        v_bs = [Value(v) for v in bvs]
        terms = [a * b for a, b in zip(v_as, v_bs)]
        L: Value = terms[0]
        for t in terms[1:]:
            L = L + t
        _topo_backward(L)

        _allclose(L.data, L_t.detach().item(), label="dot fwd")
        for i, (va, ta) in enumerate(zip(v_as, t_as)):
            assert ta.grad is not None
            _allclose(va.grad, ta.grad.item(), label=f"dot grad_a{i}")

    def test_product_of_sum(self) -> None:
        """L = (a + b) * (c + d)."""
        av, bv, cv, dv = 1.0, 2.0, 3.0, 4.0

        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        tc = torch.tensor(cv, dtype=torch.float64, requires_grad=True)
        td = torch.tensor(dv, dtype=torch.float64, requires_grad=True)
        L_t = (ta + tb) * (tc + td)
        L_t.backward()

        va, vb, vc, vd = Value(av), Value(bv), Value(cv), Value(dv)
        L = (va + vb) * (vc + vd)
        _topo_backward(L)

        _allclose(L.data, L_t.detach().item(), label="prod_of_sum fwd")
        for nano, ref, lbl in [
            (va.grad, ta.grad, "a"), (vb.grad, tb.grad, "b"),
            (vc.grad, tc.grad, "c"), (vd.grad, td.grad, "d"),
        ]:
            assert ref is not None
            _allclose(nano, ref.item(), label=f"prod_of_sum grad_{lbl}")


# ---------------------------------------------------------------------------
# 6. Numerical edge cases
# ---------------------------------------------------------------------------

class TestMulEdgeCases:
    """Multiplication at numerical extremes."""

    def test_large_values(self) -> None:
        a, b = 1e8, 1e8
        ref, ref_ga, ref_gb = _torch_mul(a, b)
        va, vb = Value(a), Value(b)
        vc = va * vb
        vc.grad = 1.0
        vc._backward()
        _allclose(vc.data, ref, label="large fwd")
        _allclose(va.grad, ref_ga, label="large grad_a")
        _allclose(vb.grad, ref_gb, label="large grad_b")

    def test_negative_times_negative(self) -> None:
        ref, ref_ga, ref_gb = _torch_mul(-3.0, -4.0)
        va, vb = Value(-3.0), Value(-4.0)
        vc = va * vb
        vc.grad = 1.0
        vc._backward()
        _allclose(vc.data, ref, label="neg*neg fwd")
        _allclose(va.grad, ref_ga, label="neg*neg grad_a")

    def test_multiply_by_negative_one(self) -> None:
        """v * (-1) should negate value and flip gradient sign."""
        tv = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        (tv * -1).backward()
        assert tv.grad is not None

        vv = Value(5.0)
        out = vv * (-1)
        out.grad = 1.0
        out._backward()
        _allclose(vv.grad, tv.grad.item(), label="v*(-1) grad")

    def test_finite_difference_grad_a(self) -> None:
        """Central-difference check for ∂(a*b)/∂a."""
        a, b, eps = 2.0, 3.0, 1e-5
        fd = ((Value(a + eps) * Value(b)).data
              - (Value(a - eps) * Value(b)).data) / (2 * eps)
        _, nano_ga, _ = Value(a).data, 0.0, 0.0
        va, vb = Value(a), Value(b)
        vc = va * vb
        vc.grad = 1.0
        vc._backward()
        _allclose(va.grad, fd, label="FD grad_a")

    def test_finite_difference_grad_b(self) -> None:
        """Central-difference check for ∂(a*b)/∂b."""
        a, b, eps = 2.0, 3.0, 1e-5
        fd = ((Value(a) * Value(b + eps)).data
              - (Value(a) * Value(b - eps)).data) / (2 * eps)
        va, vb = Value(a), Value(b)
        vc = va * vb
        vc.grad = 1.0
        vc._backward()
        _allclose(vb.grad, fd, label="FD grad_b")
