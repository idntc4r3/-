"""Activation layers for NanoTorch."""

from nanotorch.tensor import Tensor


class ReLU:
    """Apply the rectified linear activation to each input element."""

    def forward(self, x: Tensor) -> Tensor:
        """Return max(0, x), preserving the computation graph."""
        if not isinstance(x, Tensor):
            raise TypeError("ReLU expects a Tensor input")
        return x.relu()

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the activation."""
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """Return an empty list because ReLU has no trainable parameters."""
        return []
