import cython
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
def normalize_product(double [::1, :] XTY, int [::1] X_size, int [::1] Y_size, int a, int b):
    '''
    this is a utility for computing the generalized jaccard kernel, given by

    k(x,y) = a*<x,y>/(|x| + |y| - b*<x,y>)

    this function takes in the product <x,y> (which can be efficiently computed using usual numpy/scipy sparse operations)
    and arrays containing |x|, |y|, and normalizes the product on the fly. this avoids the memory overhead of computing
    the full matrix of M_{ij} = |x_i| + |x_j| - b*<x_i,x_j>
    '''
    cdef Py_ssize_t i, j
    for i in prange(XTY.shape[0], nogil=True):
        for j in range(XTY.shape[1]):
            XTY[i,j] /= ((X_size[i] + Y_size[j] - b*XTY[i,j])/a)

@cython.boundscheck(False)
@cython.wraparound(False)
def normalize_product_tril(double [::1, :] XTY, int [::1] X_size, int [::1] Y_size, int a, int b):
    '''
    this is a utility for computing the generalized jaccard kernel, given by

    k(x,y) = a*<x,y>/(|x| + |y| - b*<x,y>)

    this function takes in the product <x,y> (which can be efficiently computed using usual numpy/scipy sparse operations)
    and arrays containing |x|, |y|, and normalizes the product on the fly. this avoids the memory overhead of computing
    the full matrix of M_{ij} = |x_i| + |x_j| - b*<x_i,x_j>
    '''
    cdef Py_ssize_t i, j
    for i in prange(XTY.shape[0], nogil=True):
        for j in range(i):
            XTY[i,j] /= ((X_size[i] + Y_size[j] - b*XTY[i,j])/a)

@cython.boundscheck(False)
@cython.wraparound(False)
def _jaccard(double [::1, :] XTY, int [::1] X_size, int [::1] Y_size, int a, int b):
    '''
    this is a utility for computing the generalized jaccard kernel, given by

    k(x,y) = a*<x,y>/(|x| + |y| - b*<x,y>)

    this function takes in the product <x,y> (which can be efficiently computed using usual numpy/scipy sparse operations)
    and arrays containing |x|, |y|, and normalizes the product on the fly. this avoids the memory overhead of computing
    the full matrix of M_{ij} = |x_i| + |x_j| - b*<x_i,x_j>
    '''
    cdef Py_ssize_t i, j
    for i in prange(XTY.shape[0], nogil=True):
        for j in range(XTY.shape[1]):
            XTY[i,j] /= ((X_size[i] + Y_size[j] - b*XTY[i,j])/a)