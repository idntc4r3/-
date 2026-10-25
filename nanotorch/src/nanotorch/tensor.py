"""Multi-dimensional tensor autograd core."""

from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np

ArrayLike = int | float | list[Any] | Sequence[Any] | np.ndarray


def unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Sum out broadcasted dimensions in grad to match target_shape.

    Mathematical rule:
        If dimension was expanded from 1 -> N or prepended, upstream gradient
        must be summed across those axes.
    """
    if grad.shape == target_shape:
        return grad

    # Sum out extra leading dimensions
    ndim_diff = grad.ndim - len(target_shape)
    for _ in range(ndim_diff):
        grad = grad.sum(axis=0)

    # Sum along axes where target has size 1
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


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


    def __add__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(self.data + other_t.data, requires_grad=req_grad, _children=(self, other_t), _op="+")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = unbroadcast(out.grad, self.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self
            if other_t.requires_grad:
                grad_other = unbroadcast(out.grad, other_t.shape)
                other_t.grad = grad_other if other_t.grad is None else other_t.grad + grad_other

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike) -> Tensor:
        return self.__add__(other)


    def __neg__(self) -> Tensor:
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op="neg")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            g = -out.grad
            self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def __sub__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return self.__add__(-other_t)

    def __rsub__(self, other: ArrayLike) -> Tensor:
        other_t = Tensor(other)
        return other_t.__sub__(self)


    def __mul__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(self.data * other_t.data, requires_grad=req_grad, _children=(self, other_t), _op="*")

        self_data = self.data
        other_data = other_t.data

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = unbroadcast(out.grad * other_data, self.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self
            if other_t.requires_grad:
                grad_other = unbroadcast(out.grad * self_data, other_t.shape)
                other_t.grad = grad_other if other_t.grad is None else other_t.grad + grad_other

        out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike) -> Tensor:
        return self.__mul__(other)


    def __pow__(self, exponent: int | float) -> Tensor:
        if not isinstance(exponent, (int, float)):
            raise TypeError(f"Tensor exponent must be int or float, got {type(exponent).__name__}")
        out = Tensor(self.data ** exponent, requires_grad=self.requires_grad, _children=(self,), _op=f"**{exponent}")

        self_data = self.data

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            local_g = out.grad * exponent * (self_data ** (exponent - 1))
            g = unbroadcast(local_g, self.shape)
            self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def __truediv__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(self.data / other_t.data, requires_grad=req_grad, _children=(self, other_t), _op="/")

        self_data = self.data
        other_data = other_t.data

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                g_self = unbroadcast(out.grad / other_data, self.shape)
                self.grad = g_self if self.grad is None else self.grad + g_self
            if other_t.requires_grad:
                g_other = unbroadcast(-out.grad * self_data / (other_data ** 2), other_t.shape)
                other_t.grad = g_other if other_t.grad is None else other_t.grad + g_other

        out._backward = _backward
        return out

    def __rtruediv__(self, other: ArrayLike) -> Tensor:
        other_t = Tensor(other)
        return other_t.__truediv__(self)


    def matmul(self, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)
        req_grad = self.requires_grad or other.requires_grad
        out = Tensor(np.matmul(self.data, other.data), requires_grad=req_grad, _children=(self, other), _op="@")
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.matmul(other)
