# coding: utf-8
from .sum import ParallelSum

class ParallelMean(ParallelSum):
    """Find mean values in pixels in parallel, on-line.

    The usual life-cycle of this class is to create it,
    repeatedly call add_data on chunks, and then call
    collect to finalize. You can also call the "run"
    method with an iterator to combine these.
    """
    def collect(self, comm=None, mode="gather"):
        """Finalize the sum and return the counts and the means.

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
        mean: array or SparseArray
            The mean of values hitting each pixel
        """
        weights, sums = super().collect(comm=comm, mode=mode)
        means = sums / weights
        return weights, means
