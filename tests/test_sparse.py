from parallel_statistics import SparseArray
import numpy as np


def test_sparse_eq():
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


def test_sparse_set():
    s = SparseArray(10)
    s[0] = 1
    s[np.array([1, 2])] = 2
    s[np.array([2, 3])] = 3

    d = s.to_dense()
    assert d[0] == 1
    assert d[1] == 2
    assert d[2] == 3
    assert d[3] == 3
    assert np.all(d[4:] == 0)
