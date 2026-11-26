"""Shape-changing layers for neural networks."""

import math

from nanotorch.tensor import Tensor


class Flatten:
    """Merge a contiguous range of input dimensions into a single dimension.

    By default the first dimension is retained, allowing batches of inputs to
    be passed to a Linear layer. Negative dimensions count from the end.
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        if any(not isinstance(d, int) or isinstance(d, bool) for d in (start_dim, end_dim)):
            raise TypeError("dimensions must be integers")
        self.start_dim, self.end_dim = start_dim, end_dim

    def forward(self, x: Tensor) -> Tensor:
        """Return the reshaped tensor while retaining its autograd graph."""
        if not isinstance(x, Tensor):
            raise TypeError("Flatten expects a Tensor input")
        ndim = max(x.ndim, 1)
        if not (-ndim <= self.start_dim < ndim and -ndim <= self.end_dim < ndim):
            raise IndexError("flatten dimension out of range")
        start, end = self.start_dim % ndim, self.end_dim % ndim
        if start > end:
            raise ValueError("start_dim must precede or equal end_dim")
        if x.ndim == 0:
            return x.reshape(1)
        size = math.prod(x.shape[start:end + 1])
        shape = x.shape[:start] + (size,) + x.shape[end + 1:]
        return x.reshape(*shape)

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the layer."""
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """Return an empty list because Flatten has no trainable parameters."""
        return []
