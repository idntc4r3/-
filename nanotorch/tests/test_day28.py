import numpy as np
from nanotorch.tensor import Tensor

def test_sum_forward():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = Tensor(data)
    assert t.sum().item() == 10.0
    assert np.allclose(t.sum(axis=0).data, [4.0, 6.0])