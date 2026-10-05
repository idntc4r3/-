"""Scalar computation-graph sub-package.

Public API
----------
Value
    A single node in the scalar autograd DAG.

Example::

    from nanotorch.scalar import Value

    x = Value(2.0, label='x')
    y = Value(3.0, label='y')
    # arithmetic operations (Days 3–8) will be added here
"""

from nanotorch.scalar.value import Value

__all__ = ["Value"]
