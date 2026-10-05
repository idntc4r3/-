"""NanoTorch — a lightweight autograd / ML framework built on NumPy.

Packages
--------
nanotorch.scalar
    Scalar-level computation graph (Value nodes, Days 2–15).
nanotorch.tensor  (Days 16–35)
    Multi-dimensional tensor engine with broadcasting.
nanotorch.nn      (Days 36–50)
    Neural-network layers (Linear, LayerNorm, Dropout, Embedding …).
nanotorch.optim   (Days 56–60)
    Gradient-descent optimizers (SGD, Adam, AdamW).
nanotorch.data    (Days 61–63)
    Dataset / DataLoader utilities.
nanotorch.loss    (Days 51–55)
    Loss functions (MSE, CrossEntropy …).
"""

__version__: str = "0.1.0"
__all__: list[str] = ["__version__"]
