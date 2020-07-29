import numpy as np
from .sparse import SparseArray
from .tools import AllOne, in_place_reduce

class ParallelSum:
    """``ParallelMean`` is a parallel and incremental calculator for sums.
    "Incremental" means that it does not need
    to read the entire data set at once, and requires only a single
    pass through the data.

    The calculator is designed to work on data in a collection of different bins,
    for example a map (where the bins are pixels).
    The usual life-cycle of this class is:

    * create an instance of the class (on each process if in parallel)

    * repeatedly call ``add_data`` or ``add_datum`` on it to add new data points

    * call ``collect``, (supplying in MPI communicator if in parallel)

    You can also call the ``run`` method with an iterator to combine these.

    If only a few indices in the data are expected to be used, the sparse
    option can be set to change how data is represented and returned to 
    a sparse form which will use less memory and be faster below a certain
    size.

    Bins which have no objects in will be given weight=0 and sum=0.
    """
    def __init__(self, size, sparse=False):
        """Create the calculator

        Parameters
        ----------
        size: int
            The maximum number of bins or pixels
        sparse: bool, optional
            If True, use sparse arrays to minimize memory usage
        """
        self.size = size
        self.sparse = sparse

        if sparse:
            t = SparseArray
        else:
            t = np.zeros

        self._sum = t(size)
        self._weight = t(size)

    def add_datum(self, bin, value, weight=None):
        """Add a single data point to the sum.

        Parameters
        ----------
        bin: int
            Index of bin or pixel these value apply to
        value: float
            Value for this bin to accumulate
        """
        if weight is None:
            weight = 1

        self._weight[bin] += weight
        self._sum[bin] += value * weight

    def add_data(self, bin, values, weights=None):
        """Add a chunk of data in the same bin to the sum.

        Parameters
        ----------
        bin: int
            Index of bin or pixel these value apply to
        values: sequence
            Values for this bin to accumulate
        weights: sequence
            Optional, weights per value
        """
        if weights is None:
            weights = AllOne()

        for i, value in enumerate(values):
            self._weight[bin] += weights[i]
            self._sum[bin] += value * weights[i]

    def collect(self, comm=None, mode="gather"):
        """Finalize the sum and return the counts and the sums.

        The "mode" decides whether all processes receive the results
        or just the root.

        Parameters
        ----------
        comm: mpi communicator or None
            If in parallel, supply this
        mode: str, optional
            "gather" or "allgather"

        Returns
        -------
        count: array or SparseArray
            The number of values hitting each pixel
        sum: array or SparseArray
            The total of values hitting each pixel
        """
        if comm is None:
            return self._weight, self._sum

        if self.sparse:
            if mode == "allgather":
                self._weight = comm.allreduce(self._weight)
                self._sum = comm.allreduce(self._sum)
            else:
                self._weight = comm.reduce(self._weight)
                self._sum = comm.reduce(self._sum)
        else:
            in_place_reduce(self._weight, comm, allreduce=(mode == "allgather"))
            in_place_reduce(self._sum, comm, allreduce=(mode == "allgather"))

        return self._weight, self._sum

    def run(self, iterator, comm=None, mode="gather"):
        """Run the whole life cycle on an iterator returning data chunks.

        This is equivalent to calling add_data repeatedly and then collect.

        Parameters
        ----------
        iterator: iterator
            Iterator yielding (pixel, values) pairs
        comm: MPI comm or None
            The comm, or None for serial

        Returns
        -------
        count: array or SparseArray
            The number of values hitting each pixel
        sum: array or SparseArray
            The total of values hitting each pixel
        """        
        for values in iterator:
            self.add_data(*values)
        return self.collect(comm=comm, mode=mode)
