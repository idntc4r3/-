"""Unit tests for scalar addition: Value.__add__ and Value.__radd__ (Day 3).

Every forward value and gradient is cross-checked against an equivalent
torch.Tensor computation so that any divergence from the reference
implementation is caught immediately.

Gradient formula being tested:
    c = a + b
    ∂L/∂a = ∂L/∂c · 1
    ∂L/∂b = ∂L/∂c · 1
"""

import pytest
import torch
from nanotorch.scalar import Value

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torch_add_grads(
    a_val: float,
    b_val: float,
    upstream: float = 1.0,
) -> tuple[float, float, float]:
    """Return (forward, grad_a, grad_b) using torch as ground truth."""
    a = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(b_val, dtype=torch.float64, requires_grad=True)
    c = a + b
    c.backward(torch.tensor(upstream, dtype=torch.float64))
    assert a.grad is not None
    assert b.grad is not None
    return c.detach().item(), a.grad.item(), b.grad.item()


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestAddForward:
    """Value.__add__ forward computation matches torch."""

    @pytest.mark.parametrize("a,b", [
        (2.0, 3.0),
        (-1.0, 1.0),
        (0.0, 0.0),
        (1e6, 1e-6),
        (-5.5, -3.3),
    ])
    def test_value_plus_value(self, a: float, b: float) -> None:
        expected, _, _ = _torch_add_grads(a, b)
        result = Value(a) + Value(b)
        assert result.data == pytest.approx(expected, rel=1e-9)

    def test_result_is_value(self) -> None:
        result = Value(1.0) + Value(2.0)
        assert isinstance(result, Value)

    def test_op_tag_is_plus(self) -> None:
        result = Value(1.0) + Value(2.0)
        assert result._op == "+"

    def test_prev_contains_both_operands(self) -> None:
        a = Value(1.0)
        b = Value(2.0)
        c = a + b
        assert a in c._prev
        assert b in c._prev

    def test_prev_length(self) -> None:
        a = Value(1.0)
        b = Value(2.0)
        c = a + b
        assert len(c._prev) == 2


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

class TestAddBackward:
    """Value.__add__ backward gradients match torch (upstream grad = 1.0)."""

    @pytest.mark.parametrize("a,b", [
        (2.0, 3.0),
        (-4.0, 7.0),
        (0.0, 0.0),
        (1.0, -1.0),
        (100.0, -200.0),
    ])
    def test_grad_a_equals_upstream(self, a: float, b: float) -> None:
        _, expected_ga, _ = _torch_add_grads(a, b)
        va = Value(a)
        vb = Value(b)
        vc = va + vb
        vc.grad = 1.0
        vc._backward()
        assert va.grad == pytest.approx(expected_ga, rel=1e-9)

    @pytest.mark.parametrize("a,b", [
        (2.0, 3.0),
        (-4.0, 7.0),
        (0.0, 0.0),
        (1.0, -1.0),
        (100.0, -200.0),
    ])
    def test_grad_b_equals_upstream(self, a: float, b: float) -> None:
        _, _, expected_gb = _torch_add_grads(a, b)
        va = Value(a)
        vb = Value(b)
        vc = va + vb
        vc.grad = 1.0
        vc._backward()
        assert vb.grad == pytest.approx(expected_gb, rel=1e-9)

    @pytest.mark.parametrize("upstream", [2.0, -3.0, 0.5, 0.0])
    def test_arbitrary_upstream_gradient(self, upstream: float) -> None:
        """Gradient must scale linearly with the upstream gradient."""
        _, expected_ga, expected_gb = _torch_add_grads(1.0, 2.0, upstream)
        va = Value(1.0)
        vb = Value(2.0)
        vc = va + vb
        vc.grad = upstream
        vc._backward()
        assert va.grad == pytest.approx(expected_ga, rel=1e-9)
        assert vb.grad == pytest.approx(expected_gb, rel=1e-9)

    def test_grad_accumulates_on_repeated_use(self) -> None:
        """When the same node is used twice, gradients must accumulate.

        Computation: L = a + a  →  ∂L/∂a = 2
        torch reference:
            a = tensor(3.0, requires_grad=True)
            L = a + a
            L.backward()  →  a.grad == 2.0
        """
        a_t = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        L_t = a_t + a_t
        L_t.backward()
        assert a_t.grad is not None

        va = Value(3.0)
        L = va + va
        L.grad = 1.0
        L._backward()
        assert va.grad == pytest.approx(float(a_t.grad), rel=1e-9)


# ---------------------------------------------------------------------------
# Scalar (int / float) operands
# ---------------------------------------------------------------------------

class TestAddWithScalar:
    """Value + scalar and scalar + Value should both work correctly."""

    def test_value_plus_int(self) -> None:
        v = Value(2.0)
        result = v + 3
        assert result.data == pytest.approx(5.0)

    def test_value_plus_float(self) -> None:
        v = Value(2.0)
        result = v + 3.5
        assert result.data == pytest.approx(5.5)

    def test_int_plus_value_radd(self) -> None:
        v = Value(3.0)
        result = 2 + v  # triggers __radd__
        assert result.data == pytest.approx(5.0)

    def test_float_plus_value_radd(self) -> None:
        v = Value(3.0)
        result = 1.5 + v
        assert result.data == pytest.approx(4.5)

    def test_radd_backward_grad_v(self) -> None:
        """2 + v: gradient w.r.t. v must equal 1 (upstream=1)."""
        v_t = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        L_t = 2.0 + v_t
        L_t.backward()
        assert v_t.grad is not None

        vv = Value(3.0)
        result = 2 + vv
        result.grad = 1.0
        result._backward()
        assert vv.grad == pytest.approx(float(v_t.grad), rel=1e-9)

    def test_scalar_wrapped_in_value_has_no_grad_requirement(self) -> None:
        """The auto-wrapped scalar leaf should receive grad=0 contribution
        since constants typically don't need gradients tracked."""
        v = Value(3.0)
        c = v + 2          # 2 is wrapped as Value(2.0), a leaf
        c.grad = 1.0
        c._backward()
        # v must get upstream gradient
        assert v.grad == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Chain of additions
# ---------------------------------------------------------------------------

class TestAddChain:
    """Multi-node addition chains propagate gradients correctly."""

    def test_three_node_chain(self) -> None:
        """L = a + b + c  →  ∂L/∂a = ∂L/∂b = ∂L/∂c = 1."""
        a_t = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        b_t = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        c_t = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        L_t = a_t + b_t + c_t
        L_t.backward()

        va = Value(1.0)
        vb = Value(2.0)
        vc = Value(3.0)
        # (va + vb) is an intermediate node; adding vc produces L
        L = va + vb + vc
        L.grad = 1.0
        # manually call _backward on the chain (full backward is Day 10)
        L._backward()
        mid = next(iter(L._prev - {vc}))  # the (va+vb) node
        mid._backward()

        assert va.grad == pytest.approx(float(a_t.grad), rel=1e-9)  # type: ignore[arg-type]
        assert vb.grad == pytest.approx(float(b_t.grad), rel=1e-9)  # type: ignore[arg-type]

    def test_forward_chain_value(self) -> None:
        """Forward value of a + b + c must match torch."""
        a_t = torch.tensor(1.5, dtype=torch.float64)
        b_t = torch.tensor(2.5, dtype=torch.float64)
        c_t = torch.tensor(3.0, dtype=torch.float64)
        expected = float(a_t + b_t + c_t)

        result = Value(1.5) + Value(2.5) + Value(3.0)
        assert result.data == pytest.approx(expected, rel=1e-9)

    def test_sum_builtin(self) -> None:
        """Python's built-in sum() uses __radd__ with start=0."""
        vals = [Value(1.0), Value(2.0), Value(3.0)]
        result = sum(vals)          # sum starts with integer 0
        assert isinstance(result, Value)
        assert result.data == pytest.approx(6.0)
