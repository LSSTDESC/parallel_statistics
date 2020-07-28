# coding: utf-8

import numpy as np
from .sparse import SparseArray


class ParallelMeanVariance:
    """ParallelStatsCalculator is a parallel, on-line calculator for mean
    and variance statistics.  "On-line" means that it does not need
    to read the entire data set at once, and requires only a single
    pass through the data.

    The calculator is designed for maps and similar systems, so 
    assumes that it is calculating statistics in a number of different bins
    (e.g. pixels).

    The usual life-cycle of this class is to create it,
    repeatedly call add_data on chunks, and then call
    collect to finalize. You can also call the calculate
    method with an iterator to combine these.

    If only a few indices in the data are expected to be used, the sparse
    option can be set to change how data is represented and returned to 
    a sparse form which will use less memory and be faster below a certain
    size.

    The algorithm here is basd on Schubert & Gertz 2018,
    Numerically Stable Parallel Computation of (Co-)Variance

    Attributes
    ----------
    size: int
        number of pixels or bins
    sparse: bool
        whether to use sparse representations of arrays
    
    Methods
    -------

    calculate(values_iterator, comm=None)
        Main public method - run through iterator returning bin, [values] and calculate stats
    add_data(bin, values)
        For manual usage - add another set of values for the given bin
    finalize()
        For manual usage - after all data is passed in, return counts, means, variances
    collect(counts, means, variances)
        For manual usage - combine sequences of the statistics from different processors

    """

    def __init__(self, size, sparse=False, weighted=False):
        """Create a parallel, on-line mean and variance calcuator.

        Note that unlike other calculators you must specify in advance
        whether this is weighted or not, since we have to pre-allocate
        an additional array.
        
        Parameters
        ----------
        size: int
            The number of bins (or pixels) in which statistics will be calculated
        sparse: bool, optional
            Whether to use a sparse representation of the arrays, internally and returned.
        weighted: bool, optional
            Whether to expect weights along with the data and produce weighted stats
        """
        self.size = size
        self.sparse = sparse
        self.weighted = weighted

        if sparse:
            t = SparseArray
        else:
            t = np.zeros

        self._mean = t(size)
        self._weight = t(size)
        self._M2 = t(size)

        if self.weighted:
            self._W2 = t(size)

    def add_datum(self, bin, value, weight=None):
        """Add a single data point to the sum.

        Parameters
        ----------
        bin: int
            Index of bin or pixel these value apply to
        value: float
            Value for this bin to accumulate
        weight: float
            Optional, default=1, a weight for this data point
        """
        if self.weighted:
            if weight is None:
                raise ValueError("Weights expected in ParallelStatsCalculator")

            if weight == 0:
                return

            self._weight[bin] += weight
            delta = value - self._mean[bin]
            self._mean[bin] += (weight / self._weight[bin]) * delta
            delta2 = value - self._mean[bin]
            self._M2[bin] += weight * delta * delta2
            self._W2[bin] += weight * weight

        else:
            if weight is not None:
                raise ValueError("No weights expected in ParallelStatsCalculator")

            self._weight[bin] += 1
            delta = value - self._mean[bin]
            self._mean[bin] += delta / self._weight[bin]
            delta2 = value - self._mean[bin]
            self._M2[bin] += delta * delta2


    def add_data(self, bin, values, weights=None):
        """Add a sequence of values associated with one pixel.

        Add a set of values assinged to a given bin or pixel.
        Weights must be supplied only if you set "weighted=True"
        on creation and cannot be otherwise.

        Parameters
        ----------
        bin: int
            The bin or pixel for these values
        values: sequence
            A sequence (e.g. array or list) of values assigned to this bin
        weights: sequence, optional
            A sequence (e.g. array or list) of weights per value
        """
        if self.weighted:
            if weights is None:
                raise ValueError("Weights expected in ParallelStatsCalculator")

            for value, w in zip(values, weights):
                if w == 0:
                    continue
                self._weight[bin] += w
                delta = value - self._mean[bin]
                self._mean[bin] += (w / self._weight[bin]) * delta
                delta2 = value - self._mean[bin]
                self._M2[bin] += w * delta * delta2
                self._W2[bin] += w * w
        else:
            if weights is not None:
                raise ValueError("No weights expected in ParallelStatsCalculator")
            for value in values:
                self._weight[bin] += 1
                delta = value - self._mean[bin]
                self._mean[bin] += delta / self._weight[bin]
                delta2 = value - self._mean[bin]
                self._M2[bin] += delta * delta2

    @np.errstate(divide='ignore', invalid='ignore')
    def collect(self, comm=None, mode="gather"):
        """Finalize the statistics calculation, collecting togther results
        from multiple processes.

        If mode is set to "allgather" then every calling process will return
        the same data.  Otherwise the non-root processes will return None
        for all the values.

        You can only call this once, when you've finished calling add_data.
        After that internal data is deleted.

        Parameters
        ----------
        comm: MPI Communicator or None

        mode: string
            'gather', or 'allgather'

        Returns
        -------
        weight: array or SparseArray
            The total weight or count in each bin
        mean: array or SparseArray
            An array of the computed mean for each bin
        variance: array or SparseArray
            An array of the computed variance for each bin

        """
        # Serial version - just take the values from this processor,
        # set the values where the weight is zero, and return
        if comm is None:
            results = self._weight, self._mean, self._get_variance()
            very_bad = self._weight == 0
            # Deal with pixels that have been hit, but only with
            # zero weight
            if self.sparse:
                for i in very_bad:
                    self._mean[i] = np.nan
            else:
                self._mean[very_bad] = np.nan

            del self._M2
            del self._weight
            del self._mean
            return results

        # Otherwise we do this in parallel.  The general approach is
        # a little crude because the reduction operation here is not
        # that simple (we can't just sum things, because we also need
        # the variance and combining those is slightly more complicated).
        rank = comm.Get_rank()
        size = comm.Get_size()

        if mode not in ["gather", "allgather"]:
            raise ValueError(
                "mode for ParallelStatsCalculator.collect must be"
                "'gather' or 'allgather'"
            )

        # The send command differs depending whether we are sending
        # a sparse object (which is pickled) or an array.
        if self.sparse:
            send = lambda x: comm.send(x, dest=0)
        else:
            send = lambda x: comm.Send(x, dest=0)

        # If we are not the root process we send our results
        # to the root one by one.  Then delete them to save space,
        # since for the mapping case this can get quite large.
        if rank > 0:
            send(self._weight)
            del self._weight
            send(self._mean)
            del self._mean
            send(self._M2)
            del self._M2

            # If we are running allgather and need dense arrays
            # then we make a buffer for them now and will send
            # them below
            if mode == "allgather" and not self.sparse:
                weight = np.empty(self.size)
                mean = np.empty(self.size)
                variance = np.empty(self.size)
            else:
                weight = None
                mean = None
                variance = None
        # Otherwise this is the root node, which accumulates the
        # results
        else:
            # start with our own results
            weight = self._weight
            mean = self._mean
            sq = self._M2
            if not self.sparse:
                # In the sparse case MPI4PY unpickles and creates a new variable.
                # In the dense case we have to pre-allocate it.
                w = np.empty(self.size)
                m = np.empty(self.size)
                s = np.empty(self.size)

            # Now received each processes's data chunk in turn
            # at root.
            for i in range(1, size):
                if self.sparse:
                    w = comm.recv(source=i)
                    m = comm.recv(source=i)
                    s = comm.recv(source=i)
                else:
                    comm.Recv(w, source=i)
                    comm.Recv(m, source=i)
                    comm.Recv(s, source=i)

                # Add this to the overall sample.  This is very similar
                # to what's done in add_data except it combines all the
                # pixels/bins at once.
                weight, mean, sq = self._accumulate(weight, mean, sq, w, m, s)

            # get the population variance from the squared deviations
            # and set the mean to nan where we can't estimate it.
            variance = sq / weight

        if mode == "allgather":
            if self.sparse:
                weight, mean, variance = comm.bcast([weight, mean, variance])
            else:
                comm.Bcast(weight)
                comm.Bcast(mean)
                comm.Bcast(variance)

        return weight, mean, variance

    def run(self, iterator, comm=None, mode="gather"):
        """Run the whole life cycle on an iterator returning data chunks.

        This is equivalent to calling add_data repeatedly and then collect.

        Parameters
        ----------
        iterator: iterator
            Iterator yieding (bin, values) or (bin, values, weights)
        comm: MPI comm or None
            The comm, or None for serial
        mode: str
            "gather" or "allgather"

        Returns
        -------
        weight: array or SparseArray
            The total weight or count in each bin
        mean: array or SparseArray
            An array of the computed mean for each bin
        variance: array or SparseArray
            An array of the computed variance for each bin
        """
        for values in iterator:
            self.add_data(*values)
        return self.collect(comm=comm, mode=mode)

    def _get_variance(self):
        # Compute the variance from the previously
        # computed squared deviations. 
        variance = self._M2 / self._weight
        if not self.sparse:
            if self.weighted:
                neff = self._weight ** 2 / self._W2
                bad = neff < 1.000001
            else:
                bad = self._weight < 2
            variance[bad] = np.nan

        return variance


    def _accumulate(self, weight, mean, sq, w, m, s):
        # Algorithm from Shubert and Gertz.
        weight = weight + w
        delta = m - mean
        mean = mean + (w / weight) * delta
        delta2 = m - mean
        sq = sq + s + w * delta * delta2

        return weight, mean, sq


