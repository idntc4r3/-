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


    def log_softmax(self, dim: int = -1) -> Tensor:
        """Compute log-softmax along one axis using a shifted log-sum-exp.

        For upstream g, the gradient is g - softmax(x) * sum(g, axis=dim).
        Inputs must have at least one axis.
        """
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("dim must be an integer")
        if self.ndim == 0 or not -self.ndim <= dim < self.ndim:
            raise IndexError("log_softmax dimension out of range")
        axis = dim % self.ndim
        if self.data.size == 0:
            values = np.empty_like(self.data)
            probabilities = np.empty_like(self.data)
        else:
            shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
            exponents = np.exp(shifted)
            total = exponents.sum(axis=axis, keepdims=True)
            values = shifted - np.log(total)
            probabilities = exponents / total
        out = Tensor(values, requires_grad=self.requires_grad, _children=(self,), _op="log_softmax")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad - probabilities * np.sum(out.grad, axis=axis, keepdims=True)
            self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def softmax(self, dim: int = -1) -> Tensor:
        """Normalize along one axis using a stable exponential.

        For output y and upstream gradient g the derivative is
        y * (g - sum(g * y, axis=dim)). Inputs must have at least one axis.
        """
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("dim must be an integer")
        if self.ndim == 0 or not -self.ndim <= dim < self.ndim:
            raise IndexError("softmax dimension out of range")
        axis = dim % self.ndim
        if self.data.size == 0:
            values = np.empty_like(self.data)
        else:
            shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
            exponent = np.exp(shifted)
            values = exponent / exponent.sum(axis=axis, keepdims=True)
        out = Tensor(values, requires_grad=self.requires_grad, _children=(self,), _op="softmax")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = values * (out.grad - np.sum(out.grad * values, axis=axis, keepdims=True))
            self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        """Apply hyperbolic tangent elementwise; its derivative is 1 - tanh(x)^2."""
        data = np.tanh(self.data)
        out = Tensor(data, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad * (1.0 - data ** 2)
            self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        """Apply the logistic function elementwise with a stable exponent.

        For y = sigmoid(x), the local derivative is y * (1 - y).
        """
        z = np.exp(-np.abs(self.data))
        data = np.where(self.data >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
        out = Tensor(data, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad * data * (1.0 - data)
            self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        """Apply ReLU elementwise, using zero as the derivative at zero."""
        data = self.data
        out = Tensor(np.maximum(data, 0), requires_grad=self.requires_grad, _children=(self,), _op="relu")
        # PyTorch passes the upstream gradient through for NaN values.
        mask = ~(data <= 0)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = np.where(mask, out.grad, 0.0)
            self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def exp(self) -> Tensor:
        """Apply the exponential elementwise and propagate its gradient.

        For each element, the local derivative of exp(x) is exp(x).
        """
        result = np.exp(self.data)
        out = Tensor(result, requires_grad=self.requires_grad, _children=(self,), _op="exp")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad * result
            self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        """Apply the natural logarithm elementwise.

        The local derivative is 1 / x. Nonpositive inputs follow NumPy's
        floating-point behavior: log(0) is -inf and log(negative) is NaN.
        """
        data = self.data
        out = Tensor(np.log(data), requires_grad=self.requires_grad, _children=(self,), _op="log")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad / data
            self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: int | float) -> Tensor:
        if not isinstance(exponent, (int, float)):
            raise TypeError(f"Tensor exponent must be int or float, got {type(exponent).__name__}")
        out = Tensor(self.data ** exponent, requires_grad=self.requires_grad, _children=(self,), _op=f"**{exponent}")

        self_data = self.data

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            if exponent == 0:
                local_g = np.zeros_like(self_data)
            else:
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

        self_data = self.data
        other_data = other.data

        def _backward() -> None:
            if out.grad is None:
                return
            a = self_data[np.newaxis, :] if self_data.ndim == 1 else self_data
            b = other_data[..., np.newaxis] if other_data.ndim == 1 else other_data
            grad = out.grad
            if self_data.ndim == 1:
                grad = np.expand_dims(grad, -2 if other_data.ndim > 1 else 0)
            if other_data.ndim == 1:
                grad = np.expand_dims(grad, -1)
            if self.requires_grad:
                grad_self = np.matmul(grad, np.swapaxes(b, -1, -2))
                if self_data.ndim == 1:
                    grad_self = np.squeeze(grad_self, axis=-2)
                grad_self = unbroadcast(grad_self, self.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self
            if other.requires_grad:
                grad_other = np.matmul(np.swapaxes(a, -1, -2), grad)
                if other_data.ndim == 1:
                    grad_other = np.squeeze(grad_other, axis=-1)
                grad_other = unbroadcast(grad_other, other.shape)
                other.grad = grad_other if other.grad is None else other.grad + grad_other

        out._backward = _backward
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.matmul(other)

    @property
    def T(self) -> Tensor:
        return self.transpose()
    
    def transpose(self, *axes: int) -> Tensor:
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        out = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad, _children=(self,), _op="T")
        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                # NumPy accepts negative axis indices; normalize them before
                # calculating the inverse permutation.
                inv_axes = np.argsort(tuple(axis % self.ndim for axis in axes))
                g = np.transpose(out.grad, inv_axes)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> Tensor:
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, _children=(self,), _op="reshape")
        orig_shape = self.shape
        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                g = out.grad.reshape(orig_shape)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,), _op="sum")
        orig_shape = self.shape
        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            g = out.grad
            if not keepdims and axis is not None:
                # restore reduced dimensions for broadcast
                axes = (axis,) if isinstance(axis, int) else axis
                shape = list(orig_shape)
                for a in axes:
                    shape[a] = 1
                g = g.reshape(shape)
            g = np.broadcast_to(g, orig_shape)
            self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        s = self.sum(axis=axis, keepdims=keepdims)
        factor = self.size if axis is None else (np.prod([self.shape[a] for a in ((axis,) if isinstance(axis, int) else axis)]))
        return s / float(factor)

    def __getitem__(self, idx: Any) -> Tensor:
        out = Tensor(self.data[idx], requires_grad=self.requires_grad, _children=(self,), _op="slice")
        orig_shape = self.shape
        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                g = np.zeros(orig_shape, dtype=np.float64)
                np.add.at(g, idx, out.grad)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def backward(self, gradient: ArrayLike | None = None) -> None:
        """Propagate a seed gradient through the computation graph.

        The seed must have the same shape as this tensor. If omitted, an
        array of ones is used, preserving the existing backward behaviour.
        """
        seed = np.ones_like(self.data) if gradient is None else np.asarray(gradient, dtype=np.float64)
        if seed.shape != self.shape:
            raise ValueError(f"Gradient shape {seed.shape} does not match tensor shape {self.shape}")
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif id(node) not in visited:
                visited.add(id(node))
                stack.append((node, True))
                stack.extend((child, False) for child in node._prev)
        # Intermediate gradients belong to this traversal; keeping them would
        # propagate old gradients again on subsequent backward calls.
        for node in order:
            if node._prev:
                node.grad = None
        if not self._prev and self.grad is not None:
            self.grad = self.grad + seed
        else:
            self.grad = seed
        for node in reversed(order):
            node._backward()

