"""Day 6 – Extended tests for scalar multiplication chains (torch reference).

Goals
-----
* Compare complex multi-node multiplication graphs against PyTorch autograd.
* Verify gradients via ``torch.allclose`` (atol=1e-9).
* Stress-test gradient accumulation in fan-out / diamond / polynomial DAGs.
* Central-difference finite-difference checks on non-trivial expressions.
* Cover power-like patterns (a³, a⁴) built purely from repeated ``__mul__``.

All tensors use ``dtype=torch.float64`` for reliable numerical precision.
"""

from __future__ import annotations

import pytest
import torch
from nanotorch.scalar import Value

ATOL = 1e-9
RTOL = 1e-7


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _allclose(nano: float, ref: float, *, label: str = "") -> None:
    """Assert torch.allclose between two Python floats."""
    tn = torch.tensor(nano, dtype=torch.float64)
    tr = torch.tensor(ref, dtype=torch.float64)
    assert torch.allclose(tn, tr, atol=ATOL, rtol=RTOL), (
        f"{label}: nano={nano} vs ref={ref}"
    )


def _topo_backward(root: Value) -> None:
    """Topological sort + backward sweep (preview of Day 10)."""
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


def _finite_diff(fn: Value, leaf: Value, eps: float = 1e-5) -> float:
    """Numerical ∂fn/∂leaf via central difference — NOT used with shared
    Value nodes; each call rebuilds the expression from fresh leaves."""
    raise NotImplementedError("Use expression-specific helpers below.")


# ---------------------------------------------------------------------------
# 1. Polynomial chains: a^n built from repeated multiplication
# ---------------------------------------------------------------------------

class TestPolynomialChains:
    """Verify a*a*a... chains produce correct forward values and gradients.

    Mathematical reference:
        f(a) = aⁿ  →  f'(a) = n · aⁿ⁻¹
    """

    @pytest.mark.parametrize("a,n", [
        (2.0, 2), (2.0, 3), (2.0, 4), (2.0, 5),
        (-1.5, 2), (-1.5, 3), (0.5, 4), (3.0, 3),
    ])
    def test_power_forward(self, a: float, n: int) -> None:
        """aⁿ forward value matches torch."""
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        L_t = ta
        for _ in range(n - 1):
            L_t = L_t * ta
        ref = L_t.detach().item()

        va = Value(a)
        L: Value = va
        for _ in range(n - 1):
            L = L * va
        _allclose(L.data, ref, label=f"a^{n} fwd a={a}")

    @pytest.mark.parametrize("a,n", [
        (2.0, 2), (2.0, 3), (2.0, 4),
        (-1.5, 2), (-1.5, 3), (0.5, 4), (3.0, 3),
    ])
    def test_power_grad(self, a: float, n: int) -> None:
        """∂(aⁿ)/∂a = n·aⁿ⁻¹ matches torch.allclose."""
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        L_t = ta
        for _ in range(n - 1):
            L_t = L_t * ta
        L_t.backward()
        assert ta.grad is not None

        va = Value(a)
        L: Value = va
        for _ in range(n - 1):
            L = L * va
        _topo_backward(L)

        _allclose(va.grad, ta.grad.item(), label=f"a^{n} grad a={a}")


# ---------------------------------------------------------------------------
# 2. Product chains: a * b * c * ... (all distinct leaves)
# ---------------------------------------------------------------------------

class TestProductChains:
    """Chain of distinct-leaf multiplications."""

    def test_three_leaf_product(self) -> None:
        """L = a * b * c; gradients: ∂L/∂a=bc, ∂L/∂b=ac, ∂L/∂c=ab."""
        av, bv, cv = 2.0, 3.0, 4.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        tc = torch.tensor(cv, dtype=torch.float64, requires_grad=True)
        (ta * tb * tc).backward()
        assert ta.grad is not None and tb.grad is not None and tc.grad is not None

        va, vb, vc = Value(av), Value(bv), Value(cv)
        _topo_backward(va * vb * vc)

        _allclose(va.grad, ta.grad.item(), label="abc grad_a")
        _allclose(vb.grad, tb.grad.item(), label="abc grad_b")
        _allclose(vc.grad, tc.grad.item(), label="abc grad_c")

    def test_five_leaf_product(self) -> None:
        """L = a*b*c*d*e."""
        vals = [1.0, 2.0, -1.0, 0.5, 3.0]
        t_leaves = [
            torch.tensor(v, dtype=torch.float64, requires_grad=True)
            for v in vals
        ]
        L_t = t_leaves[0]
        for t in t_leaves[1:]:
            L_t = L_t * t
        L_t.backward()

        v_leaves = [Value(v) for v in vals]
        L: Value = v_leaves[0]
        for v in v_leaves[1:]:
            L = L * v
        _topo_backward(L)

        _allclose(L.data, L_t.detach().item(), label="5-prod fwd")
        for i, (vl, tl) in enumerate(zip(v_leaves, t_leaves)):
            assert tl.grad is not None
            _allclose(vl.grad, tl.grad.item(), label=f"5-prod grad_{i}")

    @pytest.mark.parametrize("n", [2, 4, 6, 8])
    def test_alternating_sign_product(self, n: int) -> None:
        """L = 1 * (-1) * 1 * (-1) ... (n terms) — sign alternation."""
        vals = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
        t_leaves = [
            torch.tensor(v, dtype=torch.float64, requires_grad=True)
            for v in vals
        ]
        L_t = t_leaves[0]
        for t in t_leaves[1:]:
            L_t = L_t * t
        L_t.backward()

        v_leaves = [Value(v) for v in vals]
        L: Value = v_leaves[0]
        for v in v_leaves[1:]:
            L = L * v
        _topo_backward(L)

        _allclose(L.data, L_t.detach().item(), label=f"alt-sign n={n} fwd")
        for i, (vl, tl) in enumerate(zip(v_leaves, t_leaves)):
            assert tl.grad is not None
            _allclose(vl.grad, tl.grad.item(), label=f"alt-sign n={n} grad_{i}")


# ---------------------------------------------------------------------------
# 3. Mixed add + mul expression trees (more complex than Day 5)
# ---------------------------------------------------------------------------

class TestComplexMixedChains:
    """Multi-operation DAGs combining addition and multiplication."""

    def test_polynomial_two_vars(self) -> None:
        """L = a²b + ab² = ab(a+b).

        ∂L/∂a = 2ab + b²
        ∂L/∂b = a²  + 2ab
        """
        av, bv = 2.0, 3.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        L_t = ta * ta * tb + ta * tb * tb
        L_t.backward()
        assert ta.grad is not None and tb.grad is not None

        va, vb = Value(av), Value(bv)
        L = va * va * vb + va * vb * vb
        _topo_backward(L)

        _allclose(L.data, L_t.detach().item(), label="a2b+ab2 fwd")
        _allclose(va.grad, ta.grad.item(), label="a2b+ab2 grad_a")
        _allclose(vb.grad, tb.grad.item(), label="a2b+ab2 grad_b")

    def test_bilinear_form(self) -> None:
        """L = (a + b) * (a - b) = a² - b²  (uses only +, *, scalar mul).

        Expressed as: L = (a + b) * (a + (-1)*b)
        ∂L/∂a = 2a,  ∂L/∂b = -2b
        """
        av, bv = 3.0, 2.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        L_t = (ta + tb) * (ta + (-1) * tb)
        L_t.backward()
        assert ta.grad is not None and tb.grad is not None

        va, vb = Value(av), Value(bv)
        L = (va + vb) * (va + (-1) * vb)
        _topo_backward(L)

        _allclose(L.data, L_t.detach().item(), label="bilinear fwd")
        _allclose(va.grad, ta.grad.item(), label="bilinear grad_a")
        _allclose(vb.grad, tb.grad.item(), label="bilinear grad_b")

    def test_nested_product_of_sums(self) -> None:
        """L = (a+b) * (b+c) * (c+a)."""
        av, bv, cv = 1.0, 2.0, 3.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        tc = torch.tensor(cv, dtype=torch.float64, requires_grad=True)
        L_t = (ta + tb) * (tb + tc) * (tc + ta)
        L_t.backward()
        assert ta.grad is not None

        va, vb, vc = Value(av), Value(bv), Value(cv)
        L = (va + vb) * (vb + vc) * (vc + va)
        _topo_backward(L)

        _allclose(L.data, L_t.detach().item(), label="prod3sums fwd")
        _allclose(va.grad, ta.grad.item(), label="prod3sums grad_a")
        _allclose(vb.grad, tb.grad.item(), label="prod3sums grad_b")
        _allclose(vc.grad, tc.grad.item(), label="prod3sums grad_c")

    def test_cubic_polynomial(self) -> None:
        """L = 3a³ - 2a² + 5a - 7.

        Built using only *, +, and scalar constants.
        ∂L/∂a = 9a² - 4a + 5
        """
        av = 2.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        L_t = 3 * ta * ta * ta + (-2) * ta * ta + 5 * ta + (-7)
        L_t.backward()
        assert ta.grad is not None

        va = Value(av)
        L = 3 * va * va * va + (-2) * va * va + 5 * va + (-7)
        _topo_backward(L)

        _allclose(L.data, L_t.detach().item(), label="cubic fwd")
        _allclose(va.grad, ta.grad.item(), label="cubic grad_a")

    def test_weighted_sum_of_products(self) -> None:
        """L = w1*(x1*x2) + w2*(x2*x3) + w3*(x3*x1).

        Tests that weight scaling propagates correctly through mul chains.
        """
        w1, w2, w3 = 0.5, 2.0, -1.0
        x1, x2, x3 = 3.0, -1.0, 2.0

        tw1 = torch.tensor(w1, dtype=torch.float64, requires_grad=True)
        tw2 = torch.tensor(w2, dtype=torch.float64, requires_grad=True)
        tw3 = torch.tensor(w3, dtype=torch.float64, requires_grad=True)
        tx1 = torch.tensor(x1, dtype=torch.float64, requires_grad=True)
        tx2 = torch.tensor(x2, dtype=torch.float64, requires_grad=True)
        tx3 = torch.tensor(x3, dtype=torch.float64, requires_grad=True)
        L_t = tw1 * (tx1 * tx2) + tw2 * (tx2 * tx3) + tw3 * (tx3 * tx1)
        L_t.backward()

        vw1, vw2, vw3 = Value(w1), Value(w2), Value(w3)
        vx1, vx2, vx3 = Value(x1), Value(x2), Value(x3)
        L = vw1 * (vx1 * vx2) + vw2 * (vx2 * vx3) + vw3 * (vx3 * vx1)
        _topo_backward(L)

        for (vv, tv, lbl) in [
            (vw1, tw1, "w1"), (vw2, tw2, "w2"), (vw3, tw3, "w3"),
            (vx1, tx1, "x1"), (vx2, tx2, "x2"), (vx3, tx3, "x3"),
        ]:
            assert tv.grad is not None
            _allclose(vv.grad, tv.grad.item(), label=f"wsum grad_{lbl}")


# ---------------------------------------------------------------------------
# 4. Diamond and fan-out DAGs with multiplication
# ---------------------------------------------------------------------------

class TestDiamondAndFanOut:
    """Nodes shared across multiple multiplication paths."""

    def test_diamond_mul(self) -> None:
        """Diamond: c = a*b; d = a*b; L = c*d = (ab)².

        ∂L/∂a = 2a·b², ∂L/∂b = 2a²·b
        """
        av, bv = 2.0, 3.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        tc = ta * tb
        td = ta * tb
        L_t = tc * td
        L_t.backward()
        assert ta.grad is not None and tb.grad is not None

        va, vb = Value(av), Value(bv)
        vc = va * vb
        vd = va * vb
        L = vc * vd
        _topo_backward(L)

        _allclose(L.data, L_t.detach().item(), label="diamond_mul fwd")
        _allclose(va.grad, ta.grad.item(), label="diamond_mul grad_a")
        _allclose(vb.grad, tb.grad.item(), label="diamond_mul grad_b")

    def test_shared_leaf_in_product_and_sum(self) -> None:
        """L = a*b + a*c  →  ∂L/∂a = b + c."""
        av, bv, cv = 2.0, 3.0, 4.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        tc = torch.tensor(cv, dtype=torch.float64, requires_grad=True)
        L_t = ta * tb + ta * tc
        L_t.backward()
        assert ta.grad is not None

        va, vb, vc = Value(av), Value(bv), Value(cv)
        L = va * vb + va * vc
        _topo_backward(L)

        _allclose(va.grad, ta.grad.item(), label="factored grad_a")
        _allclose(vb.grad, tb.grad.item(), label="factored grad_b")  # type: ignore[union-attr]
        _allclose(vc.grad, tc.grad.item(), label="factored grad_c")  # type: ignore[union-attr]

    @pytest.mark.parametrize("n_branches", [3, 5, 7])
    def test_fan_out_product(self, n_branches: int) -> None:
        """L = a*b0 + a*b1 + … + a*b(n-1)  →  ∂L/∂a = sum(bi)."""
        av = 2.0
        bvs = [float(i + 1) for i in range(n_branches)]

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
        _topo_backward(L)

        _allclose(va.grad, ta.grad.item(), label=f"fan-out n={n_branches} grad_a")


# ---------------------------------------------------------------------------
# 5. Finite-difference gradient checks on complex expressions
# ---------------------------------------------------------------------------

class TestFiniteDifference:
    """Central-difference numerical gradient checks for complex mul chains."""

    EPS = 1e-5

    def _build_and_eval(self, a: float, b: float) -> float:
        """Evaluate L = a³ + a²b + ab² + b³ = (a+b)³ / ... for given a, b."""
        va, vb = Value(a), Value(b)
        L = va * va * va + va * va * vb + va * vb * vb + vb * vb * vb
        return L.data

    def test_cubic_grad_a_fd(self) -> None:
        """∂L/∂a for L = a³+a²b+ab²+b³ via finite difference."""
        a, b = 1.5, -0.5
        fd = (self._build_and_eval(a + self.EPS, b)
              - self._build_and_eval(a - self.EPS, b)) / (2 * self.EPS)

        va, vb = Value(a), Value(b)
        L = va * va * va + va * va * vb + va * vb * vb + vb * vb * vb
        _topo_backward(L)

        _allclose(va.grad, fd, label="cubic2var FD grad_a")

    def test_cubic_grad_b_fd(self) -> None:
        """∂L/∂b for L = a³+a²b+ab²+b³ via finite difference."""
        a, b = 1.5, -0.5
        fd = (self._build_and_eval(a, b + self.EPS)
              - self._build_and_eval(a, b - self.EPS)) / (2 * self.EPS)

        va, vb = Value(a), Value(b)
        L = va * va * va + va * va * vb + va * vb * vb + vb * vb * vb
        _topo_backward(L)

        _allclose(vb.grad, fd, label="cubic2var FD grad_b")

    def test_product_chain_fd(self) -> None:
        """∂(a*b*c)/∂b via finite difference."""
        a, b, c = 2.0, 3.0, 4.0
        eps = self.EPS

        fwd = lambda bv: (Value(a) * Value(bv) * Value(c)).data  # noqa: E731
        fd = (fwd(b + eps) - fwd(b - eps)) / (2 * eps)

        va, vb, vc = Value(a), Value(b), Value(c)
        _topo_backward(va * vb * vc)

        _allclose(vb.grad, fd, label="abc FD grad_b")


# ---------------------------------------------------------------------------
# 6. Gradient accumulation stress — mul version
# ---------------------------------------------------------------------------

class TestMulGradAccumulation:
    """Nodes reused across multiplication sub-expressions."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_a_squared_n_times_accumulated(self, n: int) -> None:
        """L = (a²)ⁿ built as a*a*a*a… (2n times).

        ∂L/∂a = 2n · a^(2n-1)
        """
        av = 1.5
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        L_t = ta
        for _ in range(2 * n - 1):
            L_t = L_t * ta
        L_t.backward()
        assert ta.grad is not None

        va = Value(av)
        L: Value = va
        for _ in range(2 * n - 1):
            L = L * va
        _topo_backward(L)

        _allclose(va.grad, ta.grad.item(), label=f"a^(2*{n}) grad_a")

    def test_shared_intermediate_node(self) -> None:
        """c = a*b; L = c*c*c  (c reused 3 times).

        ∂L/∂a = 3c² · b = 3(ab)²·b
        ∂L/∂b = 3c² · a = 3(ab)²·a
        """
        av, bv = 2.0, 3.0
        ta = torch.tensor(av, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bv, dtype=torch.float64, requires_grad=True)
        tc = ta * tb
        L_t = tc * tc * tc
        L_t.backward()
        assert ta.grad is not None and tb.grad is not None

        va, vb = Value(av), Value(bv)
        vc = va * vb
        L = vc * vc * vc
        _topo_backward(L)

        _allclose(L.data, L_t.detach().item(), label="c^3 fwd")
        _allclose(va.grad, ta.grad.item(), label="c^3 grad_a")
        _allclose(vb.grad, tb.grad.item(), label="c^3 grad_b")
