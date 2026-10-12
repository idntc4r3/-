"""Computation-graph utilities for the scalar autograd engine.

This module provides graph-traversal helpers that operate on ``Value``
nodes.  The primary export is ``topo_sort``, which returns nodes in
*reverse-topological* order (leaves first, output last) ready for the
backward pass.

Mathematical background
-----------------------
A computation graph is a Directed Acyclic Graph (DAG) where:

* **Nodes** are ``Value`` objects.
* **Edges** point from a node to its *inputs* (its ``_prev`` set).

Topological ordering guarantees that every node appears **after** all the
nodes it depends on.  During the backward pass we process nodes in
*reverse* topological order — from the output back to the leaves — so
that upstream gradients (``out.grad``) are fully accumulated before a
node's ``_backward`` closure is called.

Algorithm (iterative post-order DFS)
-------------------------------------
We use a **recursive** DFS with an explicit ``visited`` set to handle
shared nodes (diamonds) correctly::

    procedure topo_sort(root):
        order = []
        visited = set()

        def dfs(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:   # recurse into inputs first
                    dfs(child)
                order.append(v)         # post-order: append AFTER children

        dfs(root)
        return order   # leaves first, root last

Time complexity:  O(V + E)   where V = nodes, E = edges.
Space complexity: O(V)       for the visited set and call stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanotorch.scalar.value import Value

__all__ = ["topo_sort"]


def topo_sort(root: Value) -> list[Value]:
    """Return all nodes reachable from ``root`` in topological order.

    The returned list is ordered so that **every node appears before all
    nodes that depend on it**:

    * Leaf nodes (inputs with no predecessors) appear **first**.
    * ``root`` (the output / loss node) appears **last**.

    This ordering is exactly what ``Value.backward()`` (Day 10) needs:
    it iterates the list *in reverse* and calls each node's ``_backward``
    closure, ensuring upstream gradients are ready before they are
    consumed.

    Algorithm:
        Recursive post-order DFS.  Each node is added to ``order`` only
        after all its inputs (``_prev``) have been processed.  An explicit
        ``visited`` set (keyed by ``id``) prevents revisiting shared nodes
        in diamond-shaped DAGs.

    Args:
        root: The output ``Value`` node to start traversal from.

    Returns:
        A ``list[Value]`` in topological order (leaves first, root last).
        The list contains every node reachable from ``root`` exactly once.

    Example::

        a = Value(2.0, label='a')
        b = Value(3.0, label='b')
        c = a + b           # c._prev == {a, b}
        d = c * Value(4.0)  # d._prev == {c, Value(4.0)}

        order = topo_sort(d)
        # order[-1] is d  (root)
        # order[0]  is a leaf (a, b, or Value(4.0))
        # All of {a, b, c, Value(4.0)} appear before d

    Notes:
        * Nodes are deduplicated by object identity (``id``), which is
          consistent with ``Value.__hash__`` and ``Value.__eq__``.
        * The traversal visits ``_prev`` in iteration order of the
          underlying ``frozenset``, so the relative ordering of
          *sibling* leaves is implementation-defined — only the
          parent/child constraint is guaranteed.
    """
    order: list[Value] = []
    visited: set[int] = set()

    def _dfs(node: Value) -> None:
        if id(node) not in visited:
            visited.add(id(node))
            for child in node._prev:
                _dfs(child)
            order.append(node)  # post-order: node after all its inputs

    _dfs(root)
    return order
