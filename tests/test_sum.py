from parallel_statistics import ParallelSum
from mockmpi import mock_mpiexec
import numpy as np
import pytest


def run_sum(comm, nbin, ndata, mode):
    nproc = 1 if comm is None else comm.size
    rank = 0 if comm is None else comm.rank

    data = np.random.uniform(size=(nproc, ndata))
    bins = np.random.randint(0, nbin, size=(nproc, ndata))

    # make one empty bin
    if nbin > 10:
        bins[bins==10] = 9

    # and one bin empty on one process
    if (nbin > 10) and (nproc > 3):
        bins[3, bins[3]==10] = 11

    comm.Bcast(data)
    comm.Bcast(bins)

    my_data = data[rank]
    my_bins = bins[rank]


    calc = ParallelSum(size=nbin)
    for i in range(ndata):
        calc.add_datum(my_bins[i], my_data[i])

    print("local counts", rank, calc._weight)
    counts, sums = calc.collect(comm, mode=mode)

    if (rank == 0) or (mode == 'allgather'):
        true_counts = [(bins==i).sum() for i in range(nbin)]
        true_sums = [data[bins==i].sum() for i in range(nbin)]
        print("true counts: ", true_counts)
        print("true sums:", true_sums)
        print("est counts: ", counts)
        print("est sums:", sums)
        assert np.allclose(counts, true_counts, equal_nan=True)
        assert np.allclose(sums, true_sums, equal_nan=True)


def run_sum_weights(comm, nbin, ndata, mode, sparse):
    nproc = 1 if comm is None else comm.size
    rank = 0 if comm is None else comm.rank

    data = np.random.uniform(size=(nproc, ndata))
    bins = np.random.randint(0, nbin, size=(nproc, ndata))
    weights = np.random.uniform(size=(nproc, ndata))

    # make one proc where all has zero weight
    if nproc > 3:
        weights[3] = 0

    # make one empty bin
    if nbin > 10:
        weights[bins==10] = 0.0

    comm.Bcast(data)
    comm.Bcast(bins)
    comm.Bcast(weights)

    my_data = data[rank]
    my_bins = bins[rank]
    my_weights = weights[rank]


    calc = ParallelSum(size=nbin, sparse=sparse)
    for i in range(ndata):
        calc.add_datum(my_bins[i], my_data[i], my_weights[i])

    counts, sums = calc.collect(comm, mode=mode)

    if (rank == 0) or (mode == 'allgather'):
        if sparse:
            p, v = counts.to_arrays()
            counts = np.zeros(nbin)
            counts[p] = v
            p, v = sums.to_arrays()
            sums = np.zeros(nbin)
            sums[p] = v

        true_counts = [weights[(bins==i)].sum() for i in range(nbin)]
        true_sums = [
            np.dot(data[bins==i], weights[bins==i]) 
            for i in range(nbin)]
        print("true counts: ", true_counts)
        print("true sums:", true_sums)
        print("est counts: ", counts)
        print("est sums:", sums)
        assert np.allclose(counts, true_counts, equal_nan=True)
        assert np.allclose(sums, true_sums, equal_nan=True)

@pytest.mark.parametrize("nbin", [1, 10, 50])
@pytest.mark.parametrize("ndata", [1, 10, 100])
@pytest.mark.parametrize("nproc", [1, 2, 5])
@pytest.mark.parametrize("mode", ['gather', 'allgather'])
def test_sum(nbin, ndata, nproc, mode):
    mock_mpiexec(nproc, run_sum, nbin, ndata, mode)

@pytest.mark.parametrize("nbin", [1, 10, 50])
@pytest.mark.parametrize("ndata", [1, 10, 100])
@pytest.mark.parametrize("nproc", [1, 2, 5])
@pytest.mark.parametrize("mode", ['gather', 'allgather'])
@pytest.mark.parametrize("sparse", [True, False])
def test_sum_weights(nbin, ndata, nproc, mode, sparse):
    mock_mpiexec(nproc, run_sum_weights, nbin, ndata, mode, sparse)
