from parallel_statistics import SparseArray
import numpy as np


def test_sparse_array():
    # TODO loads more tests
    s = SparseArray(10)
    t = SparseArray(10)

    s[2] = 3
    s[1] = 3
    t[1] = 3
    t[2] = 4

    assert np.allclose(s == t, np.array([1]))
    assert np.allclose(s == 3, np.array([2, 1]))
    assert np.allclose(t == 3, np.array([1]))
