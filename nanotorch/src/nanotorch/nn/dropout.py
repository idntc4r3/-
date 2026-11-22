"""Dropout layer with inverted scaling during training."""

import numpy as np

from nanotorch.tensor import Tensor


class Dropout:
    """Randomly zero elements in training; evaluation is the identity.

    Each retained element is scaled by 1/(1-p), preserving its expectation.
    NumPy's global random state controls masks and can be seeded by callers.
    """

    def __init__(self, p: float = 0.5) -> None:
        if not 0 <= p <= 1:
            raise ValueError("dropout probability must lie in [0, 1]")
        self.p = p
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        """Apply a fresh mask, keeping the graph for the backward pass."""
        if not isinstance(x, Tensor):
            raise TypeError("Dropout expects a Tensor input")
        if not self.training or self.p == 0:
            return x
        if self.p == 1:
            return x * np.zeros_like(x.data)
        mask = (np.random.random(x.shape) >= self.p).astype(float) / (1 - self.p)
        return x * mask

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the layer."""
        return self.forward(x)

    def train(self, mode: bool = True) -> "Dropout":
        """Set training mode and return this layer."""
        if not isinstance(mode, bool):
            raise TypeError("mode must be a bool")
        self.training = mode
        return self

    def eval(self) -> "Dropout":
        """Switch to evaluation mode."""
        return self.train(False)

    def parameters(self) -> list[Tensor]:
        """Return an empty list; Dropout has no trainable parameters."""
        return []
