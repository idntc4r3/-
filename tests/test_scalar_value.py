"""Unit tests for nanotorch.scalar.Value (Day 2).

Tests cover:
- Construction (int, float, negative, zero)
- Field defaults (data, grad, _prev, _op, label, _backward)
- Graph structure (_children / _prev relationship)
- Dunder helpers (__repr__, __str__, __format__, __bool__,
                  __float__, __int__, __hash__, __eq__)
- Leaf nodes are hashable and can live in sets / dicts
"""

import math
import pytest
from nanotorch.scalar import Value


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    """Value can be constructed from int or float."""

    def test_from_float(self) -> None:
        v = Value(3.14)
        assert v.data == pytest.approx(3.14)

    def test_from_int(self) -> None:
        v = Value(7)
        assert v.data == 7.0
        assert isinstance(v.data, float)

    def test_from_zero(self) -> None:
        v = Value(0)
        assert v.data == 0.0

    def test_from_negative(self) -> None:
        v = Value(-5.5)
        assert v.data == pytest.approx(-5.5)


# ---------------------------------------------------------------------------
# Default field values
# ---------------------------------------------------------------------------

class TestDefaults:
    """Freshly constructed Value must have correct default attributes."""

    def test_grad_default(self) -> None:
        v = Value(1.0)
        assert v.grad == 0.0

    def test_prev_default_empty(self) -> None:
        v = Value(1.0)
        assert v._prev == frozenset()

    def test_op_default_empty(self) -> None:
        v = Value(1.0)
        assert v._op == ""

    def test_label_default_empty(self) -> None:
        v = Value(1.0)
        assert v.label == ""

    def test_backward_default_is_noop(self) -> None:
        """Default _backward must be callable and must not raise."""
        v = Value(1.0)
        v._backward()  # should not raise

    def test_backward_default_does_not_change_grad(self) -> None:
        v = Value(1.0)
        v._backward()
        assert v.grad == 0.0


# ---------------------------------------------------------------------------
# Optional constructor arguments
# ---------------------------------------------------------------------------

class TestConstructorArgs:
    """Internal keyword arguments are stored correctly."""

    def test_label_kwarg(self) -> None:
        v = Value(2.0, label="x")
        assert v.label == "x"

    def test_op_kwarg(self) -> None:
        a = Value(1.0)
        b = Value(2.0)
        c = Value(3.0, _children=(a, b), _op="+")
        assert c._op == "+"

    def test_children_stored_as_frozenset(self) -> None:
        a = Value(1.0)
        b = Value(2.0)
        c = Value(3.0, _children=(a, b), _op="+")
        assert c._prev == frozenset({a, b})

    def test_leaf_has_no_children(self) -> None:
        v = Value(5.0)
        assert len(v._prev) == 0

    def test_intermediate_node_has_children(self) -> None:
        a = Value(1.0)
        b = Value(2.0)
        c = Value(3.0, _children=(a, b))
        assert a in c._prev
        assert b in c._prev


# ---------------------------------------------------------------------------
# Dunder helpers
# ---------------------------------------------------------------------------

class TestDunders:
    """Test all dunder methods defined on Value."""

    def test_repr_no_label(self) -> None:
        v = Value(1.5)
        assert repr(v) == "Value(data=1.5, grad=0.0)"

    def test_repr_with_label(self) -> None:
        v = Value(1.5, label="w")
        assert repr(v) == "Value(data=1.5, grad=0.0, label='w')"

    def test_repr_updates_after_grad_change(self) -> None:
        v = Value(1.0)
        v.grad = 2.5
        assert "grad=2.5" in repr(v)

    def test_str(self) -> None:
        v = Value(3.0)
        assert str(v) == "Value(3.0)"

    def test_format_float_spec(self) -> None:
        v = Value(3.14159)
        assert f"{v:.2f}" == "3.14"

    def test_format_general(self) -> None:
        v = Value(42.0)
        assert f"{v:g}" == "42"

    def test_bool_truthy(self) -> None:
        assert bool(Value(1.0)) is True
        assert bool(Value(-0.001)) is True

    def test_bool_falsy(self) -> None:
        assert bool(Value(0.0)) is False

    def test_float_conversion(self) -> None:
        assert float(Value(2.5)) == 2.5
        assert isinstance(float(Value(2.5)), float)

    def test_int_conversion_truncates(self) -> None:
        assert int(Value(2.9)) == 2
        assert int(Value(-2.9)) == -2

    def test_hash_is_identity_based(self) -> None:
        a = Value(1.0)
        b = Value(1.0)
        # same data, different objects → different hashes (with overwhelming probability)
        assert hash(a) != hash(b) or a is b  # allow collision but not identity confusion

    def test_eq_identity(self) -> None:
        a = Value(1.0)
        b = a
        assert a == b

    def test_eq_different_objects(self) -> None:
        a = Value(1.0)
        b = Value(1.0)
        assert a != b

    def test_eq_non_value(self) -> None:
        a = Value(1.0)
        assert a != 1.0
        assert a != "Value(1.0)"


# ---------------------------------------------------------------------------
# Hashability (set / dict usage)
# ---------------------------------------------------------------------------

class TestHashability:
    """Value instances must be usable as dict keys and set members."""

    def test_in_set(self) -> None:
        a = Value(1.0)
        b = Value(2.0)
        s = {a, b}
        assert len(s) == 2

    def test_same_object_deduplicated_in_set(self) -> None:
        a = Value(1.0)
        s = {a, a}
        assert len(s) == 1

    def test_as_dict_key(self) -> None:
        a = Value(1.0)
        d = {a: "hello"}
        assert d[a] == "hello"

    def test_frozenset_of_values(self) -> None:
        a = Value(1.0)
        b = Value(2.0)
        fs = frozenset({a, b})
        assert a in fs
        assert b in fs


# ---------------------------------------------------------------------------
# Grad mutability
# ---------------------------------------------------------------------------

class TestGradMutability:
    """grad must be a mutable float attribute."""

    def test_grad_can_be_set(self) -> None:
        v = Value(1.0)
        v.grad = 3.14
        assert v.grad == pytest.approx(3.14)

    def test_grad_can_accumulate(self) -> None:
        v = Value(1.0)
        v.grad += 1.0
        v.grad += 2.0
        assert v.grad == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# NaN / Inf edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Value should faithfully preserve IEEE-754 special values."""

    def test_nan_data(self) -> None:
        v = Value(float("nan"))
        assert math.isnan(v.data)

    def test_inf_data(self) -> None:
        v = Value(float("inf"))
        assert math.isinf(v.data)

    def test_neg_inf_data(self) -> None:
        v = Value(float("-inf"))
        assert math.isinf(v.data) and v.data < 0
