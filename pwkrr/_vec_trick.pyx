"""
This code is ported from https://github.com/aatapa/RLScore/blob/master/rlscore/utilities/_sampled_kronecker_products.pyx
"""
import cython
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_mat_from_left(double [::1, :] dst, double [:] sparse_matrix, double [::1, :] dense_matrix, int [::1] row_inds, int [::1] col_inds, int entry_count):
    
    cdef Py_ssize_t innerind, outerind
    for innerind in prange(dst.shape[1], nogil=True):
    #for innerind in range(dst.shape[1]):
        for outerind in range(entry_count):
            dst[row_inds[outerind], innerind] += sparse_matrix[outerind] * dense_matrix[col_inds[outerind], innerind]

@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_mat_from_right(double [:, ::1] dst, double [:, ::1] dense_matrix, double [:] sparse_matrix, int [::1] row_inds, int [::1] col_inds, int entry_count):
    
    cdef Py_ssize_t innerind, outerind
    #for innerind in range(dst.shape[0]):
    for innerind in prange(dst.shape[0], nogil=True):
        for outerind in range(entry_count):
            dst[innerind, col_inds[outerind]] += dense_matrix[innerind, row_inds[outerind]] * sparse_matrix[outerind]


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_subset_of_matprod_entries(double [::1] dst, double [:, ::1] matrix_left, double [::1, :] matrix_right, int [::1] row_inds, int [::1] col_inds, int subsetlen):
    
    cdef Py_ssize_t i, j
    cdef Py_ssize_t innerind, outerind
    
    #for outerind in range(subsetlen):
    for outerind in prange(subsetlen, nogil=True):
        for innerind in range(matrix_left.shape[1]):
            dst[outerind] += matrix_left[row_inds[outerind], innerind] * matrix_right[innerind, col_inds[outerind]]