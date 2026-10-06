"""Day 4 – Extended tests for scalar addition using torch as reference.

Goals
-----
* Compare **every** forward value with ``torch.allclose`` (not just approx).
* Verify gradients of complex multi-node addition *expression trees* against
  PyTorch's autograd engine.
* Stress-test gradient accumulation on diamond-shaped and fan-out DAGs.
* Validate behaviour at numerical extremes (large values, tiny deltas).
* Confirm correctness via finite-difference gradient checking (manual
  ``gradcheck``-style), mirroring what ``torch.autograd.gradcheck`` does.

Convention
----------
All tensors use ``dtype=torch.float64`` so that finite-difference checks
are numerically reliable (float32 has insufficient precision at eps=1e-5).
"""

from __future__ import annotations

import math
import pytest
import torch
from nanotorch.scalar import Value

# ---------------------------------------------------------------------------
# Shared tolerance for torch.allclose
# ---------------------------------------------------------------------------
ATOL = 1e-9
RTOL = 1e-7


# ---------------------------------------------------------------------------
# Utility: run the same scalar expression in torch and nanotorch, then compare
# ---------------------------------------------------------------------------

def _nano_add_two(a: float, b: float) -> tuple[float, float, float]:
    """NanoTorch: return (out, grad_a, grad_b) for c = a + b."""
    va = Value(a)
    vb = Value(b)
    vc = va + vb
    vc.grad = 1.0
    vc._backward()
    return vc.data, va.grad, vb.grad


def _torch_add_two(a: float, b: float) -> tuple[float, float, float]:
    """PyTorch: return (out, grad_a, grad_b) for c = a + b."""
    ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    tc = ta + tb
    tc.backward()
    assert ta.grad is not None and tb.grad is not None
    return tc.detach().item(), ta.grad.item(), tb.grad.item()


def _allclose(nano: float, ref: float, *, label: str = "") -> None:
    """Assert torch.allclose between two Python floats (wrapped as tensors)."""
    t_nano = torch.tensor(nano, dtype=torch.float64)
    t_ref = torch.tensor(ref, dtype=torch.float64)
    assert torch.allclose(t_nano, t_ref, atol=ATOL, rtol=RTOL), (
        f"{label}: nano={nano} vs ref={ref} (atol={ATOL}, rtol={RTOL})"
    )


# ---------------------------------------------------------------------------
# 1. Pairwise forward + backward vs torch — exhaustive parameter grid
# ---------------------------------------------------------------------------

class TestAllcloseBasic:
    """Forward value and both gradients match torch.allclose for a+b."""

    VALUES = [-100.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 100.0]

    @pytest.mark.parametrize("a", VALUES)
    @pytest.mark.parametrize("b", VALUES)
    def test_forward_allclose(self, a: float, b: float) -> None:
        nano_out, _, _ = _nano_add_two(a, b)
        ref_out, _, _ = _torch_add_two(a, b)
        _allclose(nano_out, ref_out, label=f"forward a={a} b={b}")

    @pytest.mark.parametrize("a", VALUES)
    @pytest.mark.parametrize("b", VALUES)
    def test_grad_a_allclose(self, a: float, b: float) -> None:
        _, nano_ga, _ = _nano_add_two(a, b)
        _, ref_ga, _ = _torch_add_two(a, b)
        _allclose(nano_ga, ref_ga, label=f"grad_a a={a} b={b}")

    @pytest.mark.parametrize("a", VALUES)
    @pytest.mark.parametrize("b", VALUES)
    def test_grad_b_allclose(self, a: float, b: float) -> None:
        _, _, nano_gb = _nano_add_two(a, b)
        _, _, ref_gb = _torch_add_two(a, b)
        _allclose(nano_gb, ref_gb, label=f"grad_b a={a} b={b}")


# ---------------------------------------------------------------------------
# 2. Finite-difference gradient check (manual gradcheck style)
# ---------------------------------------------------------------------------

def _finite_diff(fn_val: float, a: float, b: float, eps: float = 1e-5) -> tuple[float, float]:
    """Numerical ∂(a+b)/∂a and ∂(a+b)/∂b via central differences.

    Central difference formula:
        f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)
    """
    # ∂/∂a
    fwd_a = (Value(a + eps) + Value(b)).data
    bwd_a = (Value(a - eps) + Value(b)).data
    grad_a = (fwd_a - bwd_a) / (2 * eps)

    # ∂/∂b
    fwd_b = (Value(a) + Value(b + eps)).data
    bwd_b = (Value(a) + Value(b - eps)).data
    grad_b = (fwd_b - bwd_b) / (2 * eps)

    return grad_a, grad_b


class TestFiniteDifference:
    """Analytical gradients match central-difference numerical estimates."""

    @pytest.mark.parametrize("a,b", [
        (1.0, 2.0), (-3.0, 5.0), (0.0, 0.0), (1e3, -1e3),
    ])
    def test_grad_a_finite_diff(self, a: float, b: float) -> None:
        _, nano_ga, _ = _nano_add_two(a, b)
        fd_ga, _ = _finite_diff(a + b, a, b)
        _allclose(nano_ga, fd_ga, label=f"FD grad_a a={a} b={b}")

    @pytest.mark.parametrize("a,b", [
        (1.0, 2.0), (-3.0, 5.0), (0.0, 0.0), (1e3, -1e3),
    ])
    def test_grad_b_finite_diff(self, a: float, b: float) -> None:
        _, _, nano_gb = _nano_add_two(a, b)
        _, fd_gb = _finite_diff(a + b, a, b)
        _allclose(nano_gb, fd_gb, label=f"FD grad_b a={a} b={b}")


# ---------------------------------------------------------------------------
# 3. Multi-node expression trees — forward and manual backward vs torch
# ---------------------------------------------------------------------------

class TestExpressionTrees:
    """Complex addition trees produce correct values and gradients."""

    def test_linear_combination(self) -> None:
        """L = 2a + 3b  →  ∂L/∂a = 2, ∂L/∂b = 3.

        Expressed as repeated addition:  L = a + a + b + b + b
        """
        a_t = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        b_t = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
        L_t = a_t + a_t + b_t + b_t + b_t
        L_t.backward()
        assert a_t.grad is not None and b_t.grad is not None

        va = Value(1.5)
        vb = Value(-2.0)
        L = va + va + vb + vb + vb

        # Manually unroll backward through the linear chain
        # Each intermediate node passes its grad downstream
        L.grad = 1.0
        # Topological order (right to left):
        # L = ((va+va+vb+vb)+vb)
        # We call _backward on each node manually (full backward is Day 10)
        nodes: list[Value] = []
        visited: set[int] = set()

        def _topo(v: Value) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    _topo(child)
                nodes.append(v)

        _topo(L)
        for node in reversed(nodes):
            node._backward()

        _allclose(L.data, L_t.detach().item(), label="linear_combo forward")
        _allclose(va.grad, a_t.grad.item(), label="linear_combo grad_a")
        _allclose(vb.grad, b_t.grad.item(), label="linear_combo grad_b")

    def test_diamond_dag(self) -> None:
        """Diamond DAG: c = a + b; d = a + b; L = c + d.

        Here both paths from a lead to L, so ∂L/∂a = 4 (a contributes via
        c and d, each of which contributes once to L).
        """
        a_t = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        b_t = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        c_t = a_t + b_t
        d_t = a_t + b_t
        L_t = c_t + d_t
        L_t.backward()
        assert a_t.grad is not None and b_t.grad is not None

        va = Value(2.0)
        vb = Value(3.0)
        vc = va + vb
        vd = va + vb
        L = vc + vd

        nodes2: list[Value] = []
        visited2: set[int] = set()

        def _topo2(v: Value) -> None:
            if id(v) not in visited2:
                visited2.add(id(v))
                for child in v._prev:
                    _topo2(child)
                nodes2.append(v)

        _topo2(L)
        L.grad = 1.0
        for node in reversed(nodes2):
            node._backward()

        _allclose(L.data, L_t.detach().item(), label="diamond forward")
        _allclose(va.grad, a_t.grad.item(), label="diamond grad_a")
        _allclose(vb.grad, b_t.grad.item(), label="diamond grad_b")

    def test_fan_out_five(self) -> None:
        """a used in five separate additions: L = (a+b1)+(a+b2)+(a+b3)+(a+b4)+(a+b5).

        ∂L/∂a = 5
        """
        a_v = 1.0
        bs = [0.1, 0.2, 0.3, 0.4, 0.5]

        # torch reference
        a_t = torch.tensor(a_v, dtype=torch.float64, requires_grad=True)
        b_ts = [torch.tensor(b, dtype=torch.float64, requires_grad=True) for b in bs]
        L_t = sum(a_t + b for b in b_ts)  # type: ignore[arg-type]
        assert isinstance(L_t, torch.Tensor)
        L_t.backward()
        assert a_t.grad is not None

        # nanotorch
        va = Value(a_v)
        vbs = [Value(b) for b in bs]
        terms = [va + vb for vb in vbs]
        L: Value = terms[0]
        for t in terms[1:]:
            L = L + t

        nodes3: list[Value] = []
        visited3: set[int] = set()

        def _topo3(v: Value) -> None:
            if id(v) not in visited3:
                visited3.add(id(v))
                for child in v._prev:
                    _topo3(child)
                nodes3.append(v)

        _topo3(L)
        L.grad = 1.0
        for node in reversed(nodes3):
            node._backward()

        _allclose(L.data, L_t.detach().item(), label="fan_out forward")
        _allclose(va.grad, a_t.grad.item(), label="fan_out grad_a")

    def test_nested_addition(self) -> None:
        """Deeply nested: L = ((((a + b) + c) + d) + e).

        All leaves get gradient 1.
        """
        vals = [1.0, -2.0, 3.0, -4.0, 5.0]
        labels = ["a", "b", "c", "d", "e"]

        # torch
        t_leaves = [
            torch.tensor(v, dtype=torch.float64, requires_grad=True)
            for v in vals
        ]
        L_t = t_leaves[0]
        for t in t_leaves[1:]:
            L_t = L_t + t
        L_t.backward()

        # nanotorch
        v_leaves = [Value(v) for v in vals]
        L: Value = v_leaves[0]
        for v in v_leaves[1:]:
            L = L + v

        nodes4: list[Value] = []
        visited4: set[int] = set()

        def _topo4(node: Value) -> None:
            if id(node) not in visited4:
                visited4.add(id(node))
                for child in node._prev:
                    _topo4(child)
                nodes4.append(node)

        _topo4(L)
        L.grad = 1.0
        for node in reversed(nodes4):
            node._backward()

        _allclose(L.data, L_t.detach().item(), label="nested forward")
        for i, (vl, tl) in enumerate(zip(v_leaves, t_leaves)):
            assert tl.grad is not None
            _allclose(vl.grad, tl.grad.item(), label=f"nested grad_{labels[i]}")


# ---------------------------------------------------------------------------
# 4. Scalar + Value  /  Value + scalar — systematic vs torch
# ---------------------------------------------------------------------------

class TestScalarOperandVsTorch:
    """Mixed scalar/Value operands match torch for both forward and backward."""

    @pytest.mark.parametrize("scalar,val", [
        (0, 1.0), (1, -1.0), (-3, 2.5), (100, 0.001),
    ])
    def test_int_plus_value_forward(self, scalar: int, val: float) -> None:
        expected = scalar + val
        result = scalar + Value(val)
        _allclose(result.data, expected, label=f"{scalar}+Value({val}) fwd")

    @pytest.mark.parametrize("scalar,val", [
        (0.5, 1.0), (-1.5, -1.0), (3.14, 0.0),
    ])
    def test_float_plus_value_forward(self, scalar: float, val: float) -> None:
        expected = scalar + val
        result = scalar + Value(val)
        _allclose(result.data, expected, label=f"{scalar}+Value({val}) fwd")

    def test_int_plus_value_backward(self) -> None:
        """2 + v: only v receives gradient."""
        t = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        L_t = 2 + t
        L_t.backward()
        assert t.grad is not None

        vv = Value(3.0)
        L = 2 + vv
        L.grad = 1.0
        L._backward()
        _allclose(vv.grad, t.grad.item(), label="int+Value grad_v")

    def test_value_plus_float_backward(self) -> None:
        """v + 2.5: only v receives gradient."""
        t = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
        L_t = t + 2.5
        L_t.backward()
        assert t.grad is not None

        vv = Value(-1.0)
        L = vv + 2.5
        L.grad = 1.0
        L._backward()
        _allclose(vv.grad, t.grad.item(), label="Value+float grad_v")


# ---------------------------------------------------------------------------
# 5. Gradient accumulation stress test
# ---------------------------------------------------------------------------

class TestGradAccumulation:
    """Validates +=  semantics in _backward for reused nodes."""

    @pytest.mark.parametrize("n", [2, 5, 10, 20])
    def test_node_used_n_times(self, n: int) -> None:
        """L = a + a + … (n times)  →  ∂L/∂a = n."""
        a_t = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        L_t = a_t
        for _ in range(n - 1):
            L_t = L_t + a_t
        L_t.backward()
        assert a_t.grad is not None

        va = Value(1.0)
        L: Value = va
        for _ in range(n - 1):
            L = L + va

        nodes: list[Value] = []
        visited: set[int] = set()

        def _topo(v: Value) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._prev:
                    _topo(c)
                nodes.append(v)

        _topo(L)
        L.grad = 1.0
        for node in reversed(nodes):
            node._backward()

        _allclose(va.grad, a_t.grad.item(), label=f"n={n} reuse grad_a")


# ---------------------------------------------------------------------------
# 6. Edge cases and numerical extremes
# ---------------------------------------------------------------------------

class TestNumericalEdgeCases:
    """Verify behaviour at floating-point extremes."""

    def test_large_positive(self) -> None:
        a, b = 1e15, 1e15
        nano_out, nano_ga, nano_gb = _nano_add_two(a, b)
        ref_out, ref_ga, ref_gb = _torch_add_two(a, b)
        _allclose(nano_out, ref_out, label="large+ fwd")
        _allclose(nano_ga, ref_ga, label="large+ ga")
        _allclose(nano_gb, ref_gb, label="large+ gb")

    def test_large_negative(self) -> None:
        a, b = -1e15, -1e15
        nano_out, _, _ = _nano_add_two(a, b)
        ref_out, _, _ = _torch_add_two(a, b)
        _allclose(nano_out, ref_out, label="large- fwd")

    def test_tiny_delta(self) -> None:
        a, b = 1.0, 1e-15
        nano_out, _, _ = _nano_add_two(a, b)
        ref_out, _, _ = _torch_add_two(a, b)
        # floating point limits: use slightly relaxed tolerance
        assert math.isclose(nano_out, ref_out, rel_tol=1e-10)

    def test_cancellation(self) -> None:
        """a + (-a) = 0: catastrophic cancellation, grad still correct."""
        a = 1.23456789
        nano_out, nano_ga, nano_gb = _nano_add_two(a, -a)
        _allclose(nano_out, 0.0, label="cancel fwd")
        _allclose(nano_ga, 1.0, label="cancel ga")
        _allclose(nano_gb, 1.0, label="cancel gb")

    def test_identity_zero_left(self) -> None:
        """0 + b = b."""
        b = 7.77
        result = Value(0.0) + Value(b)
        _allclose(result.data, b, label="0+b fwd")

    def test_identity_zero_right(self) -> None:
        """a + 0 = a."""
        a = -3.14
        result = Value(a) + Value(0.0)
        _allclose(result.data, a, label="a+0 fwd")

    def test_commutativity(self) -> None:
        """a + b == b + a (forward values)."""
        a, b = 2.718, 3.141
        ab = (Value(a) + Value(b)).data
        ba = (Value(b) + Value(a)).data
        _allclose(ab, ba, label="commutativity")

    def test_associativity(self) -> None:
        """(a + b) + c == a + (b + c) (forward values)."""
        a, b, c = 1.1, 2.2, 3.3
        lhs = ((Value(a) + Value(b)) + Value(c)).data
        rhs = (Value(a) + (Value(b) + Value(c))).data
        _allclose(lhs, rhs, label="associativity")
