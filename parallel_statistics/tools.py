import sys
from itertools import repeat

class AllOne:
    """
    A utility to return one for every requested index
    """
    def __getitem__(self, index):
        return 1

    def __iter__(self):
        return repeat(1)

def in_place_reduce(data, comm, allreduce=False, root=0):
    """
    Sum arrays across an MPI communicator, in-place.

    If allreduce is True, then all processors receive the
    summed array; otherwise only the root processor does.
    """
    # This awkward phrasing allows us to use either mpi4py
    # or the test mpi_mock class
    in_place = sys.modules[comm.__class__.__module__].IN_PLACE

    if allreduce:
        comm.Allreduce(in_place, data)
    else:
        if comm.Get_rank() == root:
            comm.Reduce(in_place, data, root=root)
        else:
            comm.Reduce(data, None, root=root)
