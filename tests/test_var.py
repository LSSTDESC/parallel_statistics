from parallel_statistics import ParallelMeanVariance
from mockmpi import mock_mpiexec
import numpy as np
import pytest


def run_mean_variance(comm, nbin, ndata, mode):
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


    calc = ParallelMeanVariance(size=nbin)
    for i in range(ndata):
        calc.add_datum(my_bins[i], my_data[i])

    print("local counts", rank, calc._weight)
    counts, means, vars = calc.collect(comm, mode=mode)

    if (rank == 0) or (mode == 'allgather'):
        true_counts = np.array([(bins==i).sum() for i in range(nbin)])
        true_means =  np.array([data[bins==i].mean() for i in range(nbin)])
        true_vars =   np.array([data[bins==i].var() for i in range(nbin)])
        # numpy returns zero for the variance of a single value.
        # we prefer to do np.nan
        true_means[true_counts<1] = np.nan
        true_vars[true_counts<1] = np.nan
        print("true counts: ", true_counts)
        print("true means:", true_means)
        print("true vars:", true_vars)
        print("est counts: ", counts)
        print("est means:", means)
        print("est vars:", vars)
        assert np.allclose(counts, true_counts, equal_nan=True)
        assert np.allclose(means, true_means, equal_nan=True)
        assert np.allclose(vars, true_vars, equal_nan=True)

def weighted_var(values, weights):
    """
    Return the weighted average and standard deviation - naive formula

    values, weights -- Numpy ndarrays with the same shape.

    From 
    https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return variance

def run_mean_variance_weights(comm, nbin, ndata, mode, sparse):
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


    calc = ParallelMeanVariance(size=nbin, sparse=sparse)
    for i in range(ndata):
        calc.add_datum(my_bins[i], my_data[i], my_weights[i])

    counts, means, vars = calc.collect(comm, mode=mode)


    if (rank == 0) or (mode == 'allgather'):

        if sparse:
            p, v = counts.to_arrays()
            counts = np.zeros(nbin)
            counts[p] = v
            p, v = means.to_arrays()
            means = np.repeat(np.nan, nbin)
            means[p] = v
            p, v = vars.to_arrays()
            vars = np.repeat(np.nan, nbin)
            vars[p] = v

        true_counts = [weights[(bins==i)].sum() for i in range(nbin)]
        true_means = []
        true_vars = []
        for i in range(nbin):
            w = weights[bins==i]
            if w.sum() == 0:
                mu = np.nan
                v = np.nan
            else:
                mu = np.average(data[bins==i], weights=w)
                v = weighted_var(data[bins==i], w)
            true_means.append(mu)
            true_vars.append(v)

        print("true counts: ", true_counts)
        print("true means:", true_means)
        print("true vars:", true_vars)
        print("est counts: ", counts)
        print("est means:", means)
        print("est vars:", vars)
        assert np.allclose(counts, true_counts, equal_nan=True)
        assert np.allclose(means, true_means, equal_nan=True)
        assert np.allclose(vars, true_vars, equal_nan=True)

@pytest.mark.parametrize("nbin", [1, 10, 50])
@pytest.mark.parametrize("ndata", [1, 10, 100])
@pytest.mark.parametrize("nproc", [1, 2, 5])
@pytest.mark.parametrize("mode", ['gather', 'allgather'])
def test_mean_variance(nbin, ndata, nproc, mode):
    mock_mpiexec(nproc, run_mean_variance, nbin, ndata, mode)

@pytest.mark.parametrize("nbin", [1, 10, 50])
@pytest.mark.parametrize("ndata", [1, 10, 100])
@pytest.mark.parametrize("nproc", [1, 2, 5])
@pytest.mark.parametrize("mode", ['gather', 'allgather'])
@pytest.mark.parametrize("sparse", [True, False])
def test_mean_variance_weights(nbin, ndata, nproc, mode, sparse):
    mock_mpiexec(nproc, run_mean_variance_weights, nbin, ndata, mode, sparse)
