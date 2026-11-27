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


class LeakyReLU:
    """Apply x when positive and negative_slope * x otherwise."""

    def __init__(self, negative_slope: float = 0.01) -> None:
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the activation and preserve the computation graph."""
        if not isinstance(x, Tensor):
            raise TypeError("LeakyReLU expects a Tensor input")
        return x.leaky_relu(self.negative_slope)

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the layer."""
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """Return an empty list because the layer has no trainable parameters."""
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


class Softmax:
    """Normalize values along a specified axis to a probability distribution."""

    def __init__(self, dim: int = -1) -> None:
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """Compute softmax while preserving the computation graph."""
        if not isinstance(x, Tensor):
            raise TypeError("Softmax expects a Tensor input")
        return x.softmax(self.dim)

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the activation."""
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """Return an empty list because Softmax has no trainable parameters."""
        return []


class LogSoftmax:
    """Compute log probabilities along one axis."""

    def __init__(self, dim: int = -1) -> None:
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """Compute log-softmax while preserving the computation graph."""
        if not isinstance(x, Tensor):
            raise TypeError("LogSoftmax expects a Tensor input")
        return x.log_softmax(self.dim)

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the activation."""
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """Return an empty list because LogSoftmax has no parameters."""
        return []
