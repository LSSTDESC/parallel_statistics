# coding: utf-8
from .sum import ParallelSum

class ParallelMean(ParallelSum):
    """``ParallelMean`` is a parallel and incremental calculator for mean
    statistics.  "Incremental" means that it does not need
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

    Bins which have no objects in will be given weight=0 and mean=nan.
    """
    def collect(self, comm=None, mode="gather"):
        """Finalize the sum and return the counts and the means.

        The ``mode`` decides whether all processes receive the results
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
        mean: array or SparseArray
            The mean of values hitting each pixel
        """
        weights, sums = super().collect(comm=comm, mode=mode)
        if weights is not None:
            means = sums / weights
        else:
            means = None
        return weights, means
