"""Scalar computation-graph sub-package.

Public API
----------
Value
    A single node in the scalar autograd DAG.
topo_sort
    Return all nodes reachable from a root Value in topological order
    (leaves first, root last) via post-order DFS.

Example::

    from nanotorch.scalar import Value, topo_sort

    x = Value(2.0, label='x')
    y = Value(3.0, label='y')
    z = x + y
    order = topo_sort(z)   # [x, y, z]  (or [y, x, z])
"""

from nanotorch.scalar.graph import topo_sort
from nanotorch.scalar.value import Value

__all__ = ["Value", "topo_sort"]
