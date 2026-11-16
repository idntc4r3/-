"""Fully connected affine transformation."""

from __future__ import annotations

import math

import numpy as np

from nanotorch.tensor import Tensor


class Linear:
    """Apply ``y = x @ weight.T + bias`` along the final input dimension.

    Weights have shape ``(out_features, in_features)``. Parameters are
    initialized uniformly in [-1/sqrt(in_features), 1/sqrt(in_features)].
    Inputs can have any number of batch dimensions, or be a single vector.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        bound = 1.0 / math.sqrt(in_features)
        self.weight = Tensor(
            np.random.uniform(-bound, bound, (out_features, in_features)), requires_grad=True
        )
        self.bias = (
            Tensor(np.random.uniform(-bound, bound, out_features), requires_grad=True)
            if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return the affine transformation, retaining its computation graph."""
        if not isinstance(x, Tensor):
            raise TypeError("Linear expects a Tensor input")
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError(f"Expected last input dimension to be {self.in_features}")
        result = x @ self.weight.T
        return result + self.bias if self.bias is not None else result

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the layer on an input tensor."""
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """Return trainable weight and optional bias tensors."""
        return [self.weight, self.bias] if self.bias is not None else [self.weight]
