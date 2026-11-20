"""A container that applies neural-network layers in order."""

from typing import Protocol

from nanotorch.tensor import Tensor


class Layer(Protocol):
    """Interface required of layers held by Sequential."""

    def __call__(self, x: Tensor) -> Tensor: ...

    def parameters(self) -> list[Tensor]: ...


class Sequential:
    """Apply layers in order and expose their unique trainable tensors.

    Nested Sequential containers and reused layers are supported. An empty
    container returns its input unchanged.
    """

    def __init__(self, *layers: Layer) -> None:
        for layer in layers:
            if not callable(layer) or not callable(getattr(layer, "parameters", None)):
                raise TypeError("Each layer must be callable and provide parameters()")
        self.layers = tuple(layers)

    def forward(self, x: Tensor) -> Tensor:
        """Pass the input through each layer, keeping the autograd graph."""
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the composed network."""
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """Return parameters in layer order, omitting repeated objects."""
        result: list[Tensor] = []
        seen: set[int] = set()
        for layer in self.layers:
            for parameter in layer.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    result.append(parameter)
        return result
