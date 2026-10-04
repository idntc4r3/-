"""Scalar computation-graph node: the ``Value`` class.

Each ``Value`` wraps a single Python float and participates in a directed
acyclic graph (DAG) that records the forward computation.  During the
backward pass the graph is traversed in reverse-topological order and each
node accumulates its gradient via the chain rule.

Mathematical background
-----------------------
For a scalar function  L = f(x₁, x₂, …)  the gradient of the loss with
respect to node  xᵢ  is computed by the chain rule::

    ∂L/∂xᵢ = Σⱼ (∂L/∂yⱼ) · (∂yⱼ/∂xᵢ)

where the sum runs over all nodes  yⱼ  that directly consume  xᵢ.
Each operation stores the local Jacobian factor  ∂yⱼ/∂xᵢ  inside
``_backward`` as a closure, so that when ``backward()`` is called the
gradient flows from outputs back to inputs automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    pass  # future imports for type-checkers only

__all__ = ["Value"]

# Type alias for the raw numeric data accepted by Value.__init__
Numeric = int | float


class Value:
    """A node in the scalar autograd computation graph.

    Attributes:
        data:     The scalar value stored at this node (a Python float).
        grad:     Accumulated gradient ∂L/∂self accumulated during
                  ``backward()``.  Initialised to 0.0.
        label:    Optional human-readable name for debugging / visualisation.
        _prev:    Frozen set of direct predecessor ``Value`` nodes that were
                  consumed to produce this node.  Used to traverse the DAG.
        _op:      String tag of the operation that created this node
                  (e.g. ``'+'``, ``'*'``, ``'relu'``).  Empty string for
                  leaf nodes (user-created constants / parameters).
        _backward: Zero-argument callable that propagates the gradient from
                  this node back to its ``_prev`` inputs.  Each arithmetic
                  method replaces this with a closure capturing the relevant
                  local derivatives.  Defaults to a no-op so that leaf nodes
                  silently absorb the backward call.
    """

    __slots__ = ("data", "grad", "label", "_prev", "_op", "_backward")

    def __init__(
        self,
        data: Numeric,
        *,
        _children: tuple[Value, ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        """Initialise a leaf or intermediate computation node.

        Args:
            data:      Scalar value (int or float); stored as ``float``.
            _children: Tuple of predecessor ``Value`` nodes.  Should only be
                       set by operation methods, not by end users.
            _op:       Operation tag (e.g. ``'+'``, ``'*'``).  Empty for
                       user-created leaves.
            label:     Optional display label for graph visualisation.
        """
        self.data: float = float(data)
        self.grad: float = 0.0
        self.label: str = label
        self._prev: frozenset[Value] = frozenset(_children)
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None  # no-op for leaves

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return unambiguous developer-facing string.

        Example:
            >>> Value(3.0, label='x')
            Value(data=3.0, grad=0.0, label='x')
        """
        label_part = f", label={self.label!r}" if self.label else ""
        return f"Value(data={self.data}, grad={self.grad}{label_part})"

    def __str__(self) -> str:
        """Return concise user-facing string.

        Example:
            >>> str(Value(3.0, label='x'))
            'Value(3.0)'
        """
        return f"Value({self.data})"

    def __format__(self, format_spec: str) -> str:
        """Delegate format spec to the underlying float.

        Example:
            >>> f'{Value(3.14159):.2f}'
            '3.14'
        """
        return format(self.data, format_spec)

    def __bool__(self) -> bool:
        """Truthiness mirrors the underlying float.

        Example:
            >>> bool(Value(0.0))
            False
            >>> bool(Value(1.0))
            True
        """
        return bool(self.data)

    def __float__(self) -> float:
        """Allow explicit conversion to Python float.

        Example:
            >>> float(Value(2.5))
            2.5
        """
        return self.data

    def __int__(self) -> int:
        """Allow explicit conversion to Python int (truncates).

        Example:
            >>> int(Value(2.9))
            2
        """
        return int(self.data)

    def __hash__(self) -> int:
        """Hash by object identity so Values can live in sets / dicts."""
        return id(self)

    def __eq__(self, other: object) -> bool:
        """Equality by object identity (consistent with ``__hash__``).

        Note:
            We intentionally do **not** compare ``data`` values here because
            two distinct nodes may carry the same scalar but are different
            nodes in the computation graph.  Use ``value.data == other.data``
            for numeric comparisons.
        """
        return self is other
