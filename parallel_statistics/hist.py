from .tools import AllOne
import numpy as np

class ParallelHistogram:
    """Make a histogram in parallel.

    Bin edges must be pre-defined and values
    outside them will be ignored.

    The usual life-cycle of this class is to create it,
    repeatedly call add_data on chunks, and then call
    collect to finalize. You can also call the "run"
    method with an iterator to combine these.
    """
    def __init__(self, edges):
        """Create the histogram.
        
        Parameters
        ----------
        edges: sequence
            histogram bin edges
        """
        self.edges = edges
        self.size = len(edges) - 1
        self.counts = np.zeros(self.size)

    def add_data(self, data, weights=None):
        """Add a chunk of data to the histogram.
        
        Parameters
        ----------
        data: sequence
            Values to be histogrammed
        weights: sequence, optional
            Weights per value.
        """
        b = np.digitize(data, self.edges) - 1

        if weights is None:
            weights = AllOne()

        n = self.size
        for b_i, w_i in zip(b, weights):
            if b_i >= 0 and b_i < n:
                self.counts[b_i] += w_i

    def collect(self, comm=None):
        """Finalize and collect together histogram values

        Parameters
        ----------
        comm: MPI comm or None
            The comm, or None for serial

        Returns
        -------
        counts: array
            Total counts/weights per bin
        """
        counts = self.counts.copy()

        if comm is None:
            return counts

        import mpi4py.MPI
        if comm.rank == 0:
            comm.Reduce(mpi4py.MPI.IN_PLACE, counts)
            return counts
        else:
            comm.Reduce(counts, None)
            return None

    def run(self, iterator, comm=None):
        """Run the whole life cycle on an iterator returning data chunks.

        This is equivalent to calling add_data repeatedly and then collect.

        Parameters
        ----------
        iterator: iterator
            Iterator yieding values or (values, weights) pairs
        comm: MPI comm or None
            The comm, or None for serial

        Returns
        --------
        counts: array
            Total counts/weights per bin
        """
        for values in iterator:
            self.add_data(*values)
        return self.collect(comm)
