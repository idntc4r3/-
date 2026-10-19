"""Multi-dimensional tensor autograd core."""

from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np

ArrayLike = int | float | list[Any] | Sequence[Any] | np.ndarray


class Tensor:
    """A multi-dimensional array with autograd tracking."""

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _children: tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        if isinstance(data, np.ndarray):
            self.data: np.ndarray = data.astype(np.float64, copy=False)
        else:
            self.data = np.array(data, dtype=np.float64)

        self.requires_grad: bool = requires_grad
        self.grad: np.ndarray | None = None
        self._prev: frozenset[Tensor] = frozenset(_children)
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def __repr__(self) -> str:
        grad_info = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        return f"Tensor({self.data.tolist()}{grad_info})"

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data.item())

    @classmethod
    def zeros(cls, *shape: int, requires_grad: bool = False) -> Tensor:
        return cls(np.zeros(shape, dtype=np.float64), requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape: int, requires_grad: bool = False) -> Tensor:
        return cls(np.ones(shape, dtype=np.float64), requires_grad=requires_grad)

    @classmethod
    def randn(cls, *shape: int, requires_grad: bool = False) -> Tensor:
        return cls(np.random.randn(*shape), requires_grad=requires_grad)
