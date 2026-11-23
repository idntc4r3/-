"""Trainable embedding lookups for integer indices."""

from typing import Any

import numpy as np

from nanotorch.tensor import Tensor


class Embedding:
    """Map integer indices to rows of a learnable embedding matrix.

    Values initialize independently from a standard normal distribution.
    Indices may be an integer, a list of integers, or an integer NumPy array;
    the output has shape ``indices.shape + (embedding_dim,)``. Repeated
    indices accumulate gradients into the same weight row.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        if (not isinstance(num_embeddings, int) or isinstance(num_embeddings, bool)
                or not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool)
                or num_embeddings <= 0 or embedding_dim <= 0):
            raise ValueError("num_embeddings and embedding_dim must be positive integers")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim), requires_grad=True)

    def forward(self, indices: Any) -> Tensor:
        """Look up rows and preserve a snapshot of indices for backpropagation."""
        array = np.asarray(indices)
        if not np.issubdtype(array.dtype, np.integer):
            if isinstance(indices, list) and not indices:
                array = np.array([], dtype=np.int64)
            else:
                raise TypeError("Embedding indices must be integers")
        if np.any(array < 0) or np.any(array >= self.num_embeddings):
            raise IndexError("Embedding index out of range")
        return self.weight[array.astype(np.intp, copy=True)]

    def __call__(self, indices: Any) -> Tensor:
        """Evaluate the lookup."""
        return self.forward(indices)

    def parameters(self) -> list[Tensor]:
        """Return the learnable embedding matrix."""
        return [self.weight]
