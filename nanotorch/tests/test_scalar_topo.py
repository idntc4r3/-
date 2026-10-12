"""Unit tests for topo_sort — topological sort of the scalar DAG (Day 9).

Properties verified:
1. **Completeness** – every reachable node appears in the result exactly once.
2. **Root-last** – the root node is always the last element.
3. **Parent-after-children** – for every node N and every direct child C,
   C appears before N in the sorted list.
4. **Leaf-nodes** – nodes with no predecessors are present and appear before
   any node that directly or transitively depends on them.
5. **Shared nodes / diamonds** – a node used multiple times is included
   exactly once (no duplicates) and satisfies the ordering constraint w.r.t.
   all its parents.
6. **Single-node graph** – edge case: root with no children.
7. **Deep linear chain** – correctness at depth N without stack overflow
   for moderate N.
"""

from __future__ import annotations

import pytest
from nanotorch.scalar import Value, topo_sort


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ids(nodes: list[Value]) -> list[int]:
    return [id(n) for n in nodes]


def _index_of(node: Value, order: list[Value]) -> int:
    for i, n in enumerate(order):
        if n is node:
            return i
    raise AssertionError(f"Node {node!r} not found in topo_sort result")


def _check_parent_after_children(order: list[Value]) -> None:
    """Assert every node appears after all its direct children."""
    pos = {id(n): i for i, n in enumerate(order)}
    for node in order:
        node_pos = pos[id(node)]
        for child in node._prev:
            assert pos[id(child)] < node_pos, (
                f"Child {child!r} (pos {pos[id(child)]}) appears after "
                f"parent {node!r} (pos {node_pos})"
            )


# ---------------------------------------------------------------------------
# 1. Single-node graph
# ---------------------------------------------------------------------------

class TestSingleNode:
    """topo_sort on a lone leaf returns [leaf]."""

    def test_single_leaf(self) -> None:
        v = Value(1.0)
        order = topo_sort(v)
        assert len(order) == 1
        assert order[0] is v

    def test_root_is_last_for_single(self) -> None:
        v = Value(2.0)
        assert topo_sort(v)[-1] is v


# ---------------------------------------------------------------------------
# 2. Simple two-node chain: c = a + b
# ---------------------------------------------------------------------------

class TestTwoNodeAddition:
    """a + b produces a 3-node graph: {a, b, c}."""

    def setup_method(self) -> None:
        self.a = Value(2.0, label="a")
        self.b = Value(3.0, label="b")
        self.c = self.a + self.b
        self.order = topo_sort(self.c)

    def test_length(self) -> None:
        assert len(self.order) == 3

    def test_root_is_last(self) -> None:
        assert self.order[-1] is self.c

    def test_contains_all_nodes(self) -> None:
        node_ids = _ids(self.order)
        assert id(self.a) in node_ids
        assert id(self.b) in node_ids
        assert id(self.c) in node_ids

    def test_parent_after_children(self) -> None:
        _check_parent_after_children(self.order)

    def test_leaves_before_root(self) -> None:
        ia = _index_of(self.a, self.order)
        ib = _index_of(self.b, self.order)
        ic = _index_of(self.c, self.order)
        assert ia < ic
        assert ib < ic

    def test_no_duplicates(self) -> None:
        node_ids = _ids(self.order)
        assert len(node_ids) == len(set(node_ids))


# ---------------------------------------------------------------------------
# 3. Linear chain: d = ((a + b) * c)
# ---------------------------------------------------------------------------

class TestLinearChain:
    """Three-level linear chain."""

    def setup_method(self) -> None:
        self.a = Value(1.0, label="a")
        self.b = Value(2.0, label="b")
        self.c = Value(3.0, label="c")
        self.ab = self.a + self.b
        self.d = self.ab * self.c
        self.order = topo_sort(self.d)

    def test_length(self) -> None:
        # nodes: a, b, c, ab, d  = 5
        assert len(self.order) == 5

    def test_root_is_last(self) -> None:
        assert self.order[-1] is self.d

    def test_all_present(self) -> None:
        ids = _ids(self.order)
        for node in [self.a, self.b, self.c, self.ab, self.d]:
            assert id(node) in ids

    def test_parent_after_children(self) -> None:
        _check_parent_after_children(self.order)

    def test_no_duplicates(self) -> None:
        ids = _ids(self.order)
        assert len(ids) == len(set(ids))

    def test_intermediate_after_leaves(self) -> None:
        i_ab = _index_of(self.ab, self.order)
        i_a = _index_of(self.a, self.order)
        i_b = _index_of(self.b, self.order)
        assert i_a < i_ab
        assert i_b < i_ab


# ---------------------------------------------------------------------------
# 4. Diamond DAG: shared node
# ---------------------------------------------------------------------------

class TestDiamondDag:
    """c = a*b; d = a*b; L = c + d  — 'a' and 'b' shared across two paths."""

    def setup_method(self) -> None:
        self.a = Value(2.0, label="a")
        self.b = Value(3.0, label="b")
        self.c = self.a * self.b
        self.d = self.a * self.b
        self.L = self.c + self.d
        self.order = topo_sort(self.L)

    def test_a_appears_exactly_once(self) -> None:
        count = sum(1 for n in self.order if n is self.a)
        assert count == 1

    def test_b_appears_exactly_once(self) -> None:
        count = sum(1 for n in self.order if n is self.b)
        assert count == 1

    def test_root_is_last(self) -> None:
        assert self.order[-1] is self.L

    def test_no_duplicates(self) -> None:
        ids = _ids(self.order)
        assert len(ids) == len(set(ids))

    def test_parent_after_children(self) -> None:
        _check_parent_after_children(self.order)

    def test_a_before_c_and_d(self) -> None:
        ia = _index_of(self.a, self.order)
        ic = _index_of(self.c, self.order)
        id_ = _index_of(self.d, self.order)
        assert ia < ic
        assert ia < id_


# ---------------------------------------------------------------------------
# 5. Fan-out: single leaf used in many branches
# ---------------------------------------------------------------------------

class TestFanOut:
    """a used in N separate additions: L = (a+b1) + (a+b2) + … + (a+bN)."""

    @pytest.mark.parametrize("n_branches", [2, 5, 10])
    def test_a_once(self, n_branches: int) -> None:
        va = Value(1.0, label="a")
        vbs = [Value(float(i), label=f"b{i}") for i in range(n_branches)]
        terms = [va + vb for vb in vbs]
        L: Value = terms[0]
        for t in terms[1:]:
            L = L + t
        order = topo_sort(L)

        count_a = sum(1 for n in order if n is va)
        assert count_a == 1, f"Expected 'a' once, got {count_a}"

    @pytest.mark.parametrize("n_branches", [2, 5, 10])
    def test_ordering(self, n_branches: int) -> None:
        va = Value(1.0, label="a")
        vbs = [Value(float(i), label=f"b{i}") for i in range(n_branches)]
        terms = [va + vb for vb in vbs]
        L: Value = terms[0]
        for t in terms[1:]:
            L = L + t
        order = topo_sort(L)

        _check_parent_after_children(order)

    @pytest.mark.parametrize("n_branches", [2, 5, 10])
    def test_root_last(self, n_branches: int) -> None:
        va = Value(1.0)
        vbs = [Value(float(i)) for i in range(n_branches)]
        terms = [va + vb for vb in vbs]
        L: Value = terms[0]
        for t in terms[1:]:
            L = L + t
        assert topo_sort(L)[-1] is L

    @pytest.mark.parametrize("n_branches", [2, 5, 10])
    def test_no_duplicates(self, n_branches: int) -> None:
        va = Value(1.0)
        vbs = [Value(float(i)) for i in range(n_branches)]
        terms = [va + vb for vb in vbs]
        L: Value = terms[0]
        for t in terms[1:]:
            L = L + t
        ids = _ids(topo_sort(L))
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# 6. Deep linear chain (stress test)
# ---------------------------------------------------------------------------

class TestDeepChain:
    """Chain of depth N: x0 + x1 + x2 + … + xN."""

    @pytest.mark.parametrize("depth", [10, 50, 100])
    def test_completeness(self, depth: int) -> None:
        leaves = [Value(float(i)) for i in range(depth + 1)]
        acc: Value = leaves[0]
        for v in leaves[1:]:
            acc = acc + v
        order = topo_sort(acc)

        # Each intermediate + node is a unique Value, plus all leaves
        # total nodes = (depth + 1) leaves + depth intermediate = 2*depth + 1
        assert len(order) == 2 * depth + 1

    @pytest.mark.parametrize("depth", [10, 50, 100])
    def test_root_last(self, depth: int) -> None:
        leaves = [Value(float(i)) for i in range(depth + 1)]
        acc: Value = leaves[0]
        for v in leaves[1:]:
            acc = acc + v
        assert topo_sort(acc)[-1] is acc

    @pytest.mark.parametrize("depth", [10, 50, 100])
    def test_ordering_constraint(self, depth: int) -> None:
        leaves = [Value(float(i)) for i in range(depth + 1)]
        acc: Value = leaves[0]
        for v in leaves[1:]:
            acc = acc + v
        _check_parent_after_children(topo_sort(acc))


# ---------------------------------------------------------------------------
# 7. Complex mixed expression
# ---------------------------------------------------------------------------

class TestComplexExpression:
    """L = (a**2 + b**2) ** 0.5  — Euclidean norm."""

    def setup_method(self) -> None:
        self.a = Value(3.0, label="a")
        self.b = Value(4.0, label="b")
        self.a2 = self.a ** 2
        self.b2 = self.b ** 2
        self.s = self.a2 + self.b2
        self.L = self.s ** 0.5
        self.order = topo_sort(self.L)

    def test_all_nodes_present(self) -> None:
        ids = _ids(self.order)
        for node in [self.a, self.b, self.a2, self.b2, self.s, self.L]:
            assert id(node) in ids, f"{node!r} missing from topo_sort"

    def test_root_last(self) -> None:
        assert self.order[-1] is self.L

    def test_no_duplicates(self) -> None:
        ids = _ids(self.order)
        assert len(ids) == len(set(ids))

    def test_parent_after_children(self) -> None:
        _check_parent_after_children(self.order)

    def test_leaves_first_relative_to_parents(self) -> None:
        ia = _index_of(self.a, self.order)
        ia2 = _index_of(self.a2, self.order)
        is_ = _index_of(self.s, self.order)
        iL = _index_of(self.L, self.order)
        assert ia < ia2 < is_ < iL

    def test_forward_value_unchanged(self) -> None:
        """topo_sort must not mutate any node's data or grad."""
        assert self.a.data == 3.0
        assert self.b.data == 4.0
        assert self.L.data == pytest.approx(5.0)
        assert self.a.grad == 0.0
        assert self.b.grad == 0.0


# ---------------------------------------------------------------------------
# 8. topo_sort does not mutate gradients
# ---------------------------------------------------------------------------

class TestNoSideEffects:
    """topo_sort is a pure read-only traversal."""

    def test_grads_unchanged(self) -> None:
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        # manually set some grads
        a.grad = 5.0
        b.grad = 7.0
        topo_sort(c)
        assert a.grad == 5.0
        assert b.grad == 7.0

    def test_data_unchanged(self) -> None:
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        topo_sort(c)
        assert a.data == 2.0
        assert b.data == 3.0
        assert c.data == 6.0
