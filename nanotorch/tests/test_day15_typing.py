"""Day 15: Type checking and API contract verification."""

from __future__ import annotations
import inspect
from nanotorch.scalar import Value, topo_sort


def test_api_exports() -> None:
    """Verify scalar subpackage exports exact intended public symbols."""
    import nanotorch.scalar as s
    assert hasattr(s, "Value")
    assert hasattr(s, "topo_sort")
    assert getattr(s, "__all__") == ["Value", "topo_sort"]


def test_value_method_signatures() -> None:
    """Verify methods have full type annotations on Value."""
    methods = ["relu", "sigmoid", "tanh", "backward"]
    for m_name in methods:
        assert hasattr(Value, m_name), f"Missing method {m_name}"
        sig = inspect.signature(getattr(Value, m_name))
        assert sig.return_annotation is not inspect.Signature.empty
