"""Normalization layers using NanoTorch's tensor operations."""

from nanotorch.tensor import Tensor


class LayerNorm:
    """Normalize the final dimensions using population variance.

    Optional trainable weight and bias have ``normalized_shape`` and
    broadcast over all leading dimensions. Epsilon stabilizes the denominator.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        if not shape or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in shape):
            raise ValueError("normalized_shape must contain positive integers")
        if eps < 0:
            raise ValueError("eps must be nonnegative")
        self.normalized_shape = shape
        self.eps = eps
        self.weight = Tensor.ones(*shape, requires_grad=True) if elementwise_affine else None
        self.bias = Tensor.zeros(*shape, requires_grad=True) if elementwise_affine else None

    def forward(self, x: Tensor) -> Tensor:
        """Normalize input and apply optional elementwise affine parameters."""
        if not isinstance(x, Tensor):
            raise TypeError("LayerNorm expects a Tensor input")
        n = len(self.normalized_shape)
        if x.ndim < n or x.shape[-n:] != self.normalized_shape:
            raise ValueError(f"Expected final input dimensions {self.normalized_shape}, got {x.shape}")
        axes = tuple(range(x.ndim - n, x.ndim))
        centered = x - x.mean(axis=axes, keepdims=True)
        variance = (centered * centered).mean(axis=axes, keepdims=True)
        out = centered * (variance + self.eps) ** -0.5
        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias
        return out

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the layer."""
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """Return the affine weight and bias when enabled."""
        if self.weight is None or self.bias is None:
            return []
        return [self.weight, self.bias]
