import numpy as np
import pyximport
pyximport.install()
from . import _vec_trick


# u  <- R * (M x N) * C * v
def sampled_vec_trick(v, M, N, row_inds_M, row_inds_N, col_inds_M, col_inds_N, temp = None, x_after = None):
    
    assert len(v.shape) == 1
    assert len(col_inds_N) == v.shape[0]
    assert row_inds_N is not None
    assert col_inds_N is not None
    rc_m, cc_m = M.shape
    rc_n, cc_n = N.shape
    u_len = len(row_inds_N)
    v_len = len(col_inds_N)
    
    if x_after is None: x_after = np.zeros((u_len))
    else: x_after.fill(0)
    if rc_m * v_len + cc_n * u_len < rc_n * v_len + cc_m * u_len:
        if temp is None: temp = np.zeros((cc_n, rc_m), order = 'F')
        else: temp.fill(0)
        _vec_trick.sparse_mat_from_left(temp, v, M.T, col_inds_N, col_inds_M, v_len)
        _vec_trick.compute_subset_of_matprod_entries(x_after, N, temp, row_inds_N, row_inds_M, u_len)
    else:
        if temp is None: temp = np.zeros((rc_n, cc_m), order = 'C')
        else: temp.fill(0)
        _vec_trick.sparse_mat_from_right(temp, N, v, col_inds_N, col_inds_M, v_len)
        _vec_trick.compute_subset_of_matprod_entries(x_after, temp, M.T, row_inds_N, row_inds_M, u_len)
    return x_after