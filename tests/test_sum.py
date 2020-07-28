from mock_mpi import mock_mpiexec
import numpy as np
from parallel_statistics import ParallelSum

def run_sums(comm):
    # test of the paralle sum facility - only non-trivial
    # in parallel, so skip a serial test
    if comm is None:
        return

    # 10 bins to sum into
    s = ParallelSum(10)

    if comm.rank > 0:
        for i in range(10):
            s.add_data(i, [2.0])

    count, sums = s.collect(comm)

    if comm is None or comm.rank == 0:
        if comm is not None:
            print("size = ", comm.size)
        print("count = ", count)
        print("sums = ", sums)
        assert np.allclose(count, comm.size - 1)
        assert np.allclose(sums, 2 * (comm.size - 1))


def run_sums_sparse(comm):
    # as above but using a sparse sum - expect
    # zeros for unhit pixels
    if comm is None:
        return
    s = ParallelSum(5, sparse=True)

    s.add_data(0, [1.0])
    s.add_data(1, [2.0])
    s.add_data(2, [3.0])

    count, sums = s.collect(comm)

    if comm is None or comm.rank == 0:
        assert count[0] == comm.size
        assert count[1] == comm.size
        assert count[2] == comm.size
        assert count[3] == 0
        assert sums[0] == comm.size
        assert sums[1] == 2.0 * comm.size
        assert sums[2] == 3.0 * comm.size
        assert sums[3] == 0.0



def test_sums():
    run_sums(None)
    mock_mpiexec(2, run_sums)
    mock_mpiexec(3, run_sums)


def test_sparse_sums():
    run_sums_sparse(None)
    mock_mpiexec(2, run_sums_sparse)
    mock_mpiexec(3, run_sums_sparse)