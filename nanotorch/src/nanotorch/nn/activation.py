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


class Sigmoid:
    """Apply the logistic activation to each input element."""

    def forward(self, x: Tensor) -> Tensor:
        """Return sigmoid(x), preserving the computation graph."""
        if not isinstance(x, Tensor):
            raise TypeError("Sigmoid expects a Tensor input")
        return x.sigmoid()

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the activation."""
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """Return an empty list because Sigmoid has no trainable parameters."""
        return []


class Tanh:
    """Apply hyperbolic tangent to each input element."""

    def forward(self, x: Tensor) -> Tensor:
        """Return tanh(x), preserving the computation graph."""
        if not isinstance(x, Tensor):
            raise TypeError("Tanh expects a Tensor input")
        return x.tanh()

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the activation."""
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """Return an empty list because Tanh has no trainable parameters."""
        return []
