import numpy as np
from skbio.alignment import StripedSmithWaterman
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
import scipy.sparse as sp
from . import _generalized_jaccard


def generalized_jaccard(X, Y=None, a=1, b=1):
    """
    generalized jaccard index that can be used to compute
    the tanimoto similarity (a=1,b=1) as well as the
    dice similarity (a=2, b=0)
    """
    if Y is None:
        Y = X
    assert type(Y) is type(
        X
    ), "computing the similarity is highly inefficient if the two objects are not the same type"
    assert sp.issparse(X)
    prod = (
        (X * Y.T).toarray().copy(order="F").astype("double")
    )  # np.asarray(np.dot(X, Y.T), order="F").astype("double") if not sp.issparse(X) else
    size_X = np.atleast_1d(np.squeeze(np.asarray(X.sum(axis=1), dtype=np.int32)))
    size_Y = np.atleast_1d(np.squeeze(np.asarray(Y.sum(axis=1), dtype=np.int32)))
    _generalized_jaccard.normalize_product(prod, size_X, size_Y, a, b)
    # print(f"ligand kernel size = {round(prod.nbytes*1e-9,3)} GB")
    return prod


def generalized_jaccard_old(X, Y=None, a=1, b=1):
    """
    generalized jaccard index that can be used to compute
    the tanimoto similarity (a=1,b=1) as well as the
    dice similarity (a=2, b=0)
    """
    if Y is None:
        Y = X
    assert type(Y) is type(
        X
    ), "computing the similarity is highly inefficient if the two objects are not the same type"
    prod = np.dot(X, Y.T) if not sp.issparse(X) else (X * Y.T).toarray()
    norm_X = X.sum(axis=1).reshape(-1, 1)
    norm_Y = Y.sum(axis=1).reshape(-1, 1)
    # the broadcasting that happens in the denominator here accounts
    # for a large memory overhead that we can avoid by normalizing
    # on the fly, which is implemented in the _generalized_jaccard cython module
    return a * prod / (norm_X + norm_Y.T - b * prod)


def tanimoto(X, Y=None):
    """
    computes the tanimoto kernel
    """
    return generalized_jaccard(X, Y, a=1, b=1)


def dice(X, Y=None):
    """
    computes the dice similarity
    """
    return generalized_jaccard(X, Y, a=2, b=0)


def identity(X, Y=None):
    """
    computes the kernel k(x, y) = 1(x == y)
    """
    if Y is None:
        return np.eye(len(X))
    else:
        K = np.zeros((len(X), len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                K[i, j] = int(X[i] == Y[j])
        return K


def normalized_ssw(X, Y=None):
    """
    computes the normalized striped smith-waterman kernel
    """
    X_align = []
    for s in X:
        query = StripedSmithWaterman(s)
        X_align.append(query(s)["optimal_alignment_score"])

    if Y is None:
        Y = X
        Y_align = X_align
    else:
        Y_align = []
        for s in Y:
            query = StripedSmithWaterman(s)
            Y_align.append(query(s)["optimal_alignment_score"])

    K = np.empty((len(X), len(Y)))
    for i, si in enumerate(X):
        query = StripedSmithWaterman(si)
        for j, sj in enumerate(Y):
            K[i, j] = query(sj)["optimal_alignment_score"]
            K[i, j] /= np.sqrt(X_align[i] * Y_align[j])
    return K


KERNEL_FACTORY = {
    "tanimoto": tanimoto,
    "dice": dice,
    "identity": identity,
    "normalized_ssw": normalized_ssw,
    "rbf": rbf_kernel,
    "cosine": cosine_similarity,
}
