import random
import numpy as np
cimport numpy as np

cimport cython
from cython.view cimport array as cvarray

# Performs the lagged update of x by g.
cdef inline lagged_update(long k, double[:] x, double[:] g, unsigned long[:] lag,
                          long[:] yindices, int ylen, double[:] lag_scaling, double a):

    cdef unsigned int i
    cdef long ind
    cdef unsigned long lagged_amount = 0

    for i in range(ylen):
        ind = yindices[i]
        lagged_amount = k-lag[ind]
        lag[ind] = k
        x[ind] += lag_scaling[lagged_amount]*(a*g[ind])

# Performs x += a*y, where x is dense and y is sparse.
cdef inline add_weighted(double[:] x, double[:] ydata , long[:] yindices, int ylen, double a):
    cdef unsigned int i

    for i in range(ylen):
        x[yindices[i]] += a*ydata[i]

# Dot product of a dense vector with a sparse vector
cdef inline spdot(double[:] x, double[:] ydata , long[:] yindices, int ylen):
    cdef unsigned int i
    cdef double v = 0.0

    for i in range(ylen):
        v += ydata[i] * x[yindices[i]]

    return v

def saga_lstsq(A, double[:] b, unsigned int maxiter, props):

    # temporaries
    cdef double[:] ydata
    cdef long[:] yindices
    cdef unsigned int i, j, epoch, lagged_amount
    cdef long indstart, indend, ylen, ind
    cdef double cnew, Aix, cchange, gscaling

    # Data points are stored in columns in CSC format.
    cdef double[:] data = A.data
    cdef long[:] indices = A.indices
    cdef long[:] indptr = A.indptr

    cdef unsigned int m = A.shape[0] # dimensions
    cdef unsigned int n = A.shape[1] # datapoints

    cdef double[:] xk = np.zeros(m)
    cdef double[:] gk = np.zeros(m)

    cdef double eta = props['eta'] # Inverse step size = 1/gamma
    cdef double reg = props.get('reg', 0.0) # Default 0
    cdef double betak = 1.0 # Scaling factor for xk.

    # Tracks for each entry of x, what iteration it was last updated at.
    cdef unsigned long[:] lag = np.zeros(m, dtype=np.uint64)

    # Initialize gradients
    cdef double gd = -1.0/n
    for i in range(n):
        indstart = indptr[i]
        indend = indptr[i+1]
        ydata = data[indstart:indend]
        yindices = indices[indstart:indend]
        ylen = indend-indstart
        add_weighted(gk, ydata, yindices, ylen, gd*b[i])

    # This is just a table of the sum the geometric series (1-reg/eta)
    # It is used to correctly do the just-in-time updating when
    # L2 regularisation is used.
    cdef double[:] lag_scaling = np.zeros(n*maxiter+1)
    lag_scaling[0] = 0.0
    lag_scaling[1] = 1.0
    cdef double geosum = 1.0
    cdef double mult = 1.0-reg/eta
    for i in range(2,n*maxiter+1):
        geosum *= mult
        lag_scaling[i] = lag_scaling[i-1] + geosum

    # For least-squares, we only need to store a single
    # double for each data point, rather than a full gradient vector.
    # The value stored is the A_i*betak*x product
    cdef double[:] c = np.zeros(n)

    cdef unsigned long k = 0 # Current iteration number

    for epoch in range(maxiter):

        for j in range(n):
            if epoch == 0:
                i = j
            else:
                i = np.random.randint(0, n)

            # Selects the (sparse) column of the data matrix containing datapoint i.
            indstart = indptr[i]
            indend = indptr[i+1]
            ydata = data[indstart:indend]
            yindices = indices[indstart:indend]
            ylen = indend-indstart

            # Apply the missed updates to xk just-in-time
            lagged_update(k, xk, gk, lag, yindices, ylen, lag_scaling,-1.0/(eta*betak))
            Aix = betak*spdot(xk, ydata, yindices, ylen)

            cnew = Aix
            cchange = cnew-c[i]
            c[i] = cnew
            betak *= 1.0-reg/eta

            # Update xk with sparse step bit (with betak scaling)
            add_weighted(xk, ydata, yindices, ylen,-cchange/(eta*betak))

            k += 1

            # Perform the gradient-average part of the step
            lagged_update(k, xk, gk, lag, yindices, ylen, lag_scaling,-1.0/(eta*betak))

            # update the gradient average
            add_weighted(gk, ydata, yindices, ylen, cchange/n)

    # Perform the just in time updates for the whole xk vector, so that all entries are up-to-date.
    gscaling =-1.0/(eta*betak)
    for ind in range(m):
        lagged_amount = k-lag[ind]
        lag[ind] = k
        xk[ind] += lag_scaling[lagged_amount]*gscaling*gk[ind]
    return betak * np.asarray(xk)
