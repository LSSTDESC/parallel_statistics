Example
=======

This complete example shows how the use the ``ParallelMeanVariance`` calculator
on chunks of data loaded from an HDF5 file.

.. code-block:: python

    import mpi4py.MPI
    import h5py
    import parallel_statistics
    import numpy as np

    # This data file is available at
    # https://portal.nersc.gov/project/lsst/txpipe/tomo_challenge_data/ugrizy/mini_training.hdf5
    f = h5py.File("mini_training.hdf5", "r")
    comm = mpi4py.MPI.COMM_WORLD

    # We must divide up the data between the processes
    # Choose the chunk sizes to use here
    chunk_size = 1000
    total_size = f['redshift_true'].size
    nchunk = total_size // chunk_size
    if nchunk * chunk_size < total_size:
        nchunk += 1

    # Choose the binning in which to put values
    nbin = 20
    dz = 0.2

    # Make our calculator
    calc = parallel_statistics.ParallelMeanVariance(size=nbin)

    # Loop through the data
    for i in range(nchunk):
        # Each process only reads its assigned chunks,
        # otherwise, skip this chunk
        if i % comm.size != comm.rank:
            continue
        # work out the data range to read
        start = i * chunk_size
        end = start + chunk_size

        # read in the input data
        z = f['redshift_true'][start:end]
        r = f['r_mag'][start:end]

        # Work out which bins to use for it
        b = (z / dz).astype(int)

        # add add each one
        for j in range(z.size):
            # skip inf, nan, and sentinel values
            if not r[j] < 30:
                continue
            # add each data point
            calc.add_datum(b[j], r[j])

    # Finally, collect the results together
    weight, mean, variance = calc.collect(comm)

    # Print out results - only the root process gets the data, unless you pass
    # mode=allreduce to collect.  Will print out NaNs for bins with no objects in.
    if comm.rank == 0:
        for i in range(nbin):
            print(f"z = [{ dz * i :.1f} .. { dz * (i+1) :.1f}]    r = { mean[i] :.2f} ± { variance[i] :.2f}")
