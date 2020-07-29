from parallel_statistics import ParallelMean
from mock_mpi import mock_mpiexec
import numpy as np
import pytest


def run_mean(comm, nbin, ndata, mode):
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


    calc = ParallelMean(size=nbin)
    for i in range(ndata):
        calc.add_datum(my_bins[i], my_data[i])

    print("local counts", rank, calc._weight)
    counts, means = calc.collect(comm, mode=mode)

    if (rank == 0) or (mode == 'allgather'):
        true_counts = [(bins==i).sum() for i in range(nbin)]
        true_means = [data[bins==i].mean() for i in range(nbin)]
        print("true counts: ", true_counts)
        print("true means:", true_means)
        print("est counts: ", counts)
        print("est means:", means)
        assert np.allclose(counts, true_counts, equal_nan=True)
        assert np.allclose(means, true_means, equal_nan=True)


def run_mean_weights(comm, nbin, ndata, mode, sparse):
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


    calc = ParallelMean(size=nbin, sparse=sparse)
    for i in range(ndata):
        calc.add_datum(my_bins[i], my_data[i], my_weights[i])

    counts, means = calc.collect(comm, mode=mode)


    if (rank == 0) or (mode == 'allgather'):

        if sparse:
            p, v = counts.to_arrays()
            counts = np.zeros(nbin)
            counts[p] = v
            p, v = means.to_arrays()
            means = np.repeat(np.nan, nbin)
            means[p] = v

        true_counts = [weights[(bins==i)].sum() for i in range(nbin)]
        true_means = []
        for i in range(nbin):
            w = weights[bins==i]
            if w.sum() == 0:
                mu = np.nan
            else:
                mu = np.average(data[bins==i], weights=w)
            true_means.append(mu)

        print("true counts: ", true_counts)
        print("true means:", true_means)
        print("est counts: ", counts)
        print("est means:", means)
        assert np.allclose(counts, true_counts, equal_nan=True)
        assert np.allclose(means, true_means, equal_nan=True)

@pytest.mark.parametrize("nbin", [1, 10, 50])
@pytest.mark.parametrize("ndata", [1, 10, 100])
@pytest.mark.parametrize("nproc", [1, 2, 5])
@pytest.mark.parametrize("mode", ['gather', 'allgather'])
def test_mean(nbin, ndata, nproc, mode):
    mock_mpiexec(nproc, run_mean, args=[nbin, ndata, mode])

@pytest.mark.parametrize("nbin", [1, 10, 50])
@pytest.mark.parametrize("ndata", [1, 10, 100])
@pytest.mark.parametrize("nproc", [1, 2, 5])
@pytest.mark.parametrize("mode", ['gather', 'allgather'])
@pytest.mark.parametrize("sparse", [True, False])
def test_mean_weights(nbin, ndata, nproc, mode, sparse):
    mock_mpiexec(nproc, run_mean_weights, args=[nbin, ndata, mode, sparse])
