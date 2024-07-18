from scipy.sparse.linalg import LinearOperator
from .vec_trick import sampled_vec_trick
import numpy as np


class PairwiseKernelOperator(LinearOperator):
    """
    This is an operator which performs the operation B(Kl x Kp)B^T v for a protein kernel Kp, ligand kernel Kl
    and sparse indicator matrix B
    """

    def __init__(
        self,
        Kp,
        Kl,
        train_protein_ixs,
        train_ligand_ixs,
        test_protein_ixs=None,
        test_ligand_ixs=None,
        sample_weight=None,
    ):
        """
        Arguments:
            Kp (array): protein kernel of size (np ,np) (train) or (np ,np_test) (test)
            Kl (array): ligand kernel of size (nl, nl) (train) or (nl, nl_test) (test)
            train_protein_ixs (array): list of indices in {0,...,np-1} of size (n,)
            train_ligand_ixs (array): list of indices in {0,...,nl-1} of size (n,)
            test_protein_ixs (array, optional): list of indices in {0,...,np-1} of size (m,)
            test_ligand_ixs (array, optional): list of indices in {0,...,nl-1} of size (m,)
            sample_weight (array or None, optional): array containing per-sample weights
        """
        self.Kp = Kp.astype(np.double).copy(order="C")
        self.Kl = Kl.astype(np.double).copy(order="C")
        self.train_protein_ixs = train_protein_ixs
        self.train_ligand_ixs = train_ligand_ixs
        self.sample_weight = sample_weight

        if test_protein_ixs is None or test_ligand_ixs is None:
            assert self.Kp.shape[0] == self.Kp.shape[1]
            assert self.Kl.shape[0] == self.Kl.shape[1]
            self.test_protein_ixs = self.train_protein_ixs
            self.test_ligand_ixs = self.train_ligand_ixs
        else:
            self.test_protein_ixs = test_protein_ixs
            self.test_ligand_ixs = test_ligand_ixs

        # datatypes need to be converted for cython module to work
        self.train_protein_ixs = np.atleast_1d(
            np.squeeze(np.asarray(self.train_protein_ixs, dtype=np.int32))
        )
        self.train_ligand_ixs = np.atleast_1d(
            np.squeeze(np.asarray(self.train_ligand_ixs, dtype=np.int32))
        )
        self.test_protein_ixs = np.atleast_1d(
            np.squeeze(np.asarray(self.test_protein_ixs, dtype=np.int32))
        )
        self.test_ligand_ixs = np.atleast_1d(
            np.squeeze(np.asarray(self.test_ligand_ixs, dtype=np.int32))
        )

        self.shape = (len(self.test_ligand_ixs), len(self.train_ligand_ixs))
        self.dtype = np.dtype(float)

    def _matvec(self, v):
        """
        performs the matrix-vector product
        """
        v = v.astype(np.double)
        if self.sample_weight is not None:
            try:
                self.sample_weight = self.sample_weight.reshape(v.shape)
            except:
                raise Exception(
                    "sample weight cannot be properly reshaped, make sure it has the same dimensions as y"
                )
            v *= np.sqrt(self.sample_weight)

        v = sampled_vec_trick(
            v,
            self.Kp,
            self.Kl,
            self.test_protein_ixs,
            self.test_ligand_ixs,
            self.train_protein_ixs,
            self.train_ligand_ixs,
        )

        if self.sample_weight is not None:
            v *= np.sqrt(self.sample_weight)

        return v
