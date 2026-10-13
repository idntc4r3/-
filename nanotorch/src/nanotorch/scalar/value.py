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

# Runtime import of topo_sort is deferred to Value.backward() to avoid
# a circular import:  graph.py  →  Value  →  graph.py

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

    # ------------------------------------------------------------------
    # Arithmetic: addition
    # ------------------------------------------------------------------

    def __add__(self, other: Value | Numeric) -> Value:
        """Forward pass of addition: c = self + other.

        Mathematical background:
            Given  c = a + b, the partial derivatives are::

                ∂c/∂a = 1
                ∂c/∂b = 1

            By the chain rule the upstream gradient ∂L/∂c flows back
            unchanged to both inputs::

                ∂L/∂a += ∂L/∂c · 1
                ∂L/∂b += ∂L/∂c · 1

            We use **accumulation** (``+=``) rather than assignment so that
            a node used multiple times in the graph (a.k.a. a *multivariate*
            node) correctly sums all gradient contributions.

        Args:
            other: The right-hand operand.  Raw scalars (int / float) are
                   wrapped in a leaf ``Value`` automatically.

        Returns:
            A new ``Value`` node whose ``data = self.data + other.data``
            and whose ``_backward`` propagates gradients to both operands.

        Example:
            >>> a = Value(2.0)
            >>> b = Value(3.0)
            >>> c = a + b          # forward: c.data == 5.0
            >>> c.grad = 1.0
            >>> c._backward()      # backward: a.grad == 1.0, b.grad == 1.0
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op="+")

        def _backward() -> None:
            # ∂L/∂self  += ∂L/∂out · 1
            # ∂L/∂other += ∂L/∂out · 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Value | Numeric) -> Value:
        """Support  scalar + Value  (reflected addition).

        Python calls ``other.__radd__(self)`` when ``other.__add__(self)``
        returns ``NotImplemented``.  We simply delegate to ``__add__`` since
        addition is commutative: a + b = b + a.

        Args:
            other: Left-hand operand (typically a raw int or float).

        Returns:
            ``self.__add__(other)`` — same result as ``Value.__add__``.

        Example:
            >>> v = Value(3.0)
            >>> result = 2 + v     # calls v.__radd__(2)
            >>> result.data
            5.0
        """
        return self.__add__(other)

    # ------------------------------------------------------------------
    # Arithmetic: multiplication
    # ------------------------------------------------------------------

    def __mul__(self, other: Value | Numeric) -> Value:
        """Forward pass of multiplication: c = self * other.

        Mathematical background:
            Given  c = a · b, the partial derivatives are::

                ∂c/∂a = b
                ∂c/∂b = a

            By the chain rule::

                ∂L/∂a += ∂L/∂c · b
                ∂L/∂b += ∂L/∂c · a

            Note that ``self.data`` and ``other.data`` are captured **by
            value** inside the closure at forward-pass time.  This is
            essential: if the same ``Value`` node appears on both sides
            (e.g. ``a * a``), ``other`` is the *same object* as ``self``,
            so reading ``other.data`` later would give the updated value
            rather than the original.  The snapshot avoids this hazard.

        Args:
            other: The right-hand operand.  Raw scalars (int / float) are
                   wrapped in a leaf ``Value`` automatically.

        Returns:
            A new ``Value`` node whose ``data = self.data * other.data``
            and whose ``_backward`` propagates scaled gradients.

        Example:
            >>> a = Value(2.0)
            >>> b = Value(3.0)
            >>> c = a * b          # forward: c.data == 6.0
            >>> c.grad = 1.0
            >>> c._backward()      # backward: a.grad == 3.0, b.grad == 2.0
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")

        # Snapshot the forward-pass values so the closure is hermetic.
        self_data = self.data
        other_data = other.data

        def _backward() -> None:
            # ∂L/∂self  += ∂L/∂out · other_data   (= ∂c/∂a · upstream)
            # ∂L/∂other += ∂L/∂out · self_data    (= ∂c/∂b · upstream)
            self.grad += out.grad * other_data
            other.grad += out.grad * self_data

        out._backward = _backward
        return out

    def __rmul__(self, other: Value | Numeric) -> Value:
        """Support  scalar * Value  (reflected multiplication).

        Python calls ``other.__rmul__(self)`` when ``other.__mul__(self)``
        returns ``NotImplemented``.  Multiplication is commutative
        (a · b = b · a), so we simply delegate to ``__mul__``.

        Args:
            other: Left-hand operand (typically a raw int or float).

        Returns:
            ``self.__mul__(other)`` — same result as ``Value.__mul__``.

        Example:
            >>> v = Value(3.0)
            >>> result = 2 * v     # calls v.__rmul__(2)
            >>> result.data
            6.0
        """
        return self.__mul__(other)

    # ------------------------------------------------------------------
    # Arithmetic: exponentiation
    # ------------------------------------------------------------------

    def __pow__(self, exponent: int | float) -> Value:
        """Forward pass of exponentiation: c = self ** exponent.

        Mathematical background:
            Given  c = xⁿ  the derivative is::

                ∂c/∂x = n · x^(n-1)

            By the chain rule::

                ∂L/∂x += ∂L/∂c · n · x^(n-1)

            The base value ``self.data`` is snapshotted at forward time so
            the closure is hermetic (same reason as in ``__mul__``).

            This implementation supports any real exponent (int or float),
            so fractional powers (``x ** 0.5``) and negative powers
            (``x ** -1``) are both valid, subject to the usual domain
            constraints (e.g.  ``(-1) ** 0.5``  would produce NaN).

        Args:
            exponent: The power to raise ``self`` to.  Must be an ``int``
                      or ``float`` — raising a ``Value`` to another
                      ``Value`` is not supported here (would require
                      ``exp`` / ``log``, which come in later days).

        Returns:
            A new ``Value`` node whose ``data = self.data ** exponent``
            and whose ``_backward`` accumulates the power-rule gradient.

        Raises:
            TypeError: If ``exponent`` is not ``int`` or ``float``.

        Example:
            >>> x = Value(3.0)
            >>> c = x ** 2        # forward: c.data == 9.0
            >>> c.grad = 1.0
            >>> c._backward()     # backward: x.grad == 6.0  (2 · 3¹)
        """
        if not isinstance(exponent, int | float):
            raise TypeError(
                f"Value.__pow__ only supports int/float exponents, "
                f"got {type(exponent).__name__!r}"
            )
        base = self.data  # snapshot for hermetic closure
        out = Value(base ** exponent, _children=(self,), _op=f"**{exponent}")

        def _backward() -> None:
            # ∂L/∂self += ∂L/∂out · n · base^(n-1)
            self.grad += out.grad * exponent * (base ** (exponent - 1))

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Arithmetic: negation
    # ------------------------------------------------------------------

    def __neg__(self) -> Value:
        """Unary negation: c = -self.

        Mathematical background:
            Negation is equivalent to scaling by -1::

                c = -x = (-1) · x

            So the gradient is::

                ∂L/∂x += ∂L/∂c · (-1)

            We implement ``__neg__`` by delegating to ``__mul__`` so that
            the backward pass is handled by the already-tested
            multiplication closure and no new code paths are introduced.

        Returns:
            A new ``Value`` node with ``data = -self.data``.

        Example:
            >>> x = Value(3.0)
            >>> c = -x            # forward: c.data == -3.0
            >>> c.grad = 1.0
            >>> c._backward()     # backward: x.grad == -1.0
        """
        return self.__mul__(-1)

    # ------------------------------------------------------------------
    # Arithmetic: subtraction
    # ------------------------------------------------------------------

    def __sub__(self, other: Value | Numeric) -> Value:
        """Forward pass of subtraction: c = self - other.

        Mathematical background:
            Subtraction is decomposed as addition of the negation::

                a - b = a + (-b)

            Gradients follow directly from the chain rule applied to the
            underlying ``__add__`` and ``__neg__`` operations::

                ∂L/∂a += ∂L/∂c · 1
                ∂L/∂b += ∂L/∂c · (-1)

            No new backward closure is required — the computation graph
            created by ``__add__`` and ``__neg__`` carries the gradients
            automatically.

        Args:
            other: The right-hand operand.  Raw scalars are auto-wrapped.

        Returns:
            A new ``Value`` node with ``data = self.data - other.data``.

        Example:
            >>> a = Value(5.0)
            >>> b = Value(3.0)
            >>> c = a - b          # forward: c.data == 2.0
            >>> c.grad = 1.0
            >>> c._backward()      # a.grad == 1.0, (-b).grad == 1.0 → b.grad == -1.0
        """
        other = other if isinstance(other, Value) else Value(other)
        return self.__add__(-other)

    def __rsub__(self, other: Value | Numeric) -> Value:
        """Support  scalar - Value  (reflected subtraction).

        Python calls ``other.__rsub__(self)`` when ``other.__sub__(self)``
        returns ``NotImplemented``.

        Subtraction is **not** commutative, so we cannot simply swap::

            other - self  ≠  self - other

        Instead we wrap ``other`` in a ``Value`` and compute
        ``Value(other) - self``.

        Args:
            other: Left-hand operand (typically a raw int or float).

        Returns:
            A new ``Value`` with ``data = other - self.data``.

        Example:
            >>> v = Value(3.0)
            >>> result = 5 - v     # calls v.__rsub__(5)
            >>> result.data
            2.0
        """
        other = other if isinstance(other, Value) else Value(other)
        return other.__add__(-self)

    # ------------------------------------------------------------------
    # Arithmetic: division
    # ------------------------------------------------------------------

    def __truediv__(self, other: Value | Numeric) -> Value:
        """Forward pass of division: c = self / other.

        Mathematical background:
            Division is decomposed as multiplication by the reciprocal::

                a / b = a · b⁻¹ = a · b^(-1)

            Gradients::

                ∂L/∂a += ∂L/∂c · (1/b)
                ∂L/∂b += ∂L/∂c · (-a / b²)

            Again, no new backward closure — the graph built by ``__mul__``
            and ``__pow__(-1)`` carries the full gradient computation.

        Args:
            other: The right-hand operand (divisor).  Raw scalars
                   are auto-wrapped.  Division by zero will produce
                   ``inf`` / ``nan`` consistent with IEEE-754.

        Returns:
            A new ``Value`` node with ``data = self.data / other.data``.

        Example:
            >>> a = Value(6.0)
            >>> b = Value(2.0)
            >>> c = a / b          # forward: c.data == 3.0
            >>> c.grad = 1.0       # backward: a.grad == 0.5, b.grad == -1.5
        """
        other = other if isinstance(other, Value) else Value(other)
        return self.__mul__(other.__pow__(-1))

    def __rtruediv__(self, other: Value | Numeric) -> Value:
        """Support  scalar / Value  (reflected division).

        Python calls ``other.__rtruediv__(self)`` when
        ``other.__truediv__(self)`` returns ``NotImplemented``.

        Division is **not** commutative, so::

            other / self  =  Value(other) · self^(-1)

        Args:
            other: Left-hand operand (numerator; typically int or float).

        Returns:
            A new ``Value`` with ``data = other / self.data``.

        Example:
            >>> v = Value(2.0)
            >>> result = 6 / v     # calls v.__rtruediv__(6)
            >>> result.data
            3.0
        """
        other = other if isinstance(other, Value) else Value(other)
        return other.__mul__(self.__pow__(-1))

    # ------------------------------------------------------------------
    # Autograd: backward pass
    # ------------------------------------------------------------------

    def backward(self) -> None:
        """Run reverse-mode automatic differentiation from this node.

        After calling ``backward()`` every ``Value`` node reachable from
        ``self`` will have its ``grad`` attribute set to
        ``∂self / ∂node`` — the partial derivative of ``self`` with
        respect to that node.

        Algorithm
        ---------
        1. **Reset** all reachable gradients to ``0.0`` so that repeated
           calls to ``backward()`` produce correct, fresh results.
        2. **Seed** the root gradient: ``self.grad = 1.0`` because
           ``∂self/∂self = 1``.
        3. **Topological sort** the DAG via ``topo_sort(self)`` (Day 9),
           giving nodes in *leaves-first* order.
        4. **Reverse sweep**: iterate the sorted list in reverse order
           (root first) and call each node's ``_backward()`` closure,
           which accumulates ``grad`` into the node's predecessors using
           the chain rule::

               for node in reversed(topo_sort(self)):
                   node._backward()

        Mathematical justification
        --------------------------
        By the chain rule, the gradient flowing into a node ``v`` is the
        sum of all upstream contributions::

            ∂L/∂v = Σ  (∂L/∂u) · (∂u/∂v)   for each consumer u of v

        Processing nodes in reverse-topological order guarantees that when
        we process node ``v``, all consumers ``u`` have already had their
        ``_backward`` called and have accumulated the correct ``u.grad``
        before the contribution ``u.grad · ∂u/∂v`` is added to ``v.grad``.

        Side effects
        ------------
        * Modifies ``grad`` on every reachable ``Value`` node.
        * Safe to call multiple times — gradients are reset on each call.

        Example::

            a = Value(2.0)
            b = Value(3.0)
            L = a * b + a   # L = 2·3 + 2 = 8
            L.backward()
            # a.grad == 4.0  (∂L/∂a = b + 1 = 4)
            # b.grad == 2.0  (∂L/∂b = a     = 2)
        """
        # Deferred import to avoid circular dependency at module load time.
        from nanotorch.scalar.graph import topo_sort

        order = topo_sort(self)

        # Step 1: reset all gradients so repeated backward() is safe.
        for node in order:
            node.grad = 0.0

        # Step 2: seed — derivative of the output w.r.t. itself is 1.
        self.grad = 1.0

        # Step 3: reverse-topological sweep — propagate gradients.
        for node in reversed(order):
            node._backward()
