from .pairwise_kernel_operator import PairwiseKernelOperator
from .kernels import KERNEL_FACTORY
import scipy.sparse.linalg as sla
import numpy as np
import pickle
import os


class PairwiseKernelRidge:
    def __init__(
        self,
        alpha: float = 1e-5,
        protein_kernel="normalized_ssw",
        ligand_kernel="tanimoto",
        protein_kernel_power=1.0,
        ligand_kernel_power=1.0,
    ):
        self.alpha = alpha
        self.protein_kernel = protein_kernel
        self.ligand_kernel = ligand_kernel
        self.protein_kernel_power = protein_kernel_power
        self.ligand_kernel_power = ligand_kernel_power
        self.is_fitted = False

    def _get_protein_kernel(self, X, Y=None):
        """
        compute protein kernel

        Arguments:
            X (array): protein features
            Y (array, optional): protein features
        """
        if self.protein_kernel == "rbf":
            return KERNEL_FACTORY[self.protein_kernel](
                X, Y, gamma=self.protein_kernel_power
            )
        else:
            return KERNEL_FACTORY[self.protein_kernel](X, Y)

    def _get_ligand_kernel(self, X, Y=None):
        """
        compute ligand kernel

        Arguments:
            X (array): ligand features
            Y (array, optional): ligand features
        """
        return KERNEL_FACTORY[self.ligand_kernel](X, Y)

    def fit(
        self,
        proteins,
        ligands,
        y,
        maxiter=None,
        sample_weight=None,
        store_eigs=False,
        debug=False,
    ):
        """
        fit the pairwise kernel ridge model using the conjugate gradient method

        Arguments:
            proteins (array): (n, d_p) array of protein features, e.g. AA sequences
            ligands (array): (n, d_l) array of ligand features, e.g. fingerprints
            y (array): (n,) or (n,1) array of responses, e.g. pIC50 values
            max_iter (int, optional): integer specifying the maximum number of iterations for CG
            sample_weight (array, optional): array specifying sample weights
            store_eigs (bool): whether to compute and store the top k=500 eigenvalues/vectors
                    --> used for estimating prediction variance at inference time

        Returns:
            self
        """

        self.train_proteins, self.train_protein_ixs = proteins
        self.train_ligands, self.train_ligand_ixs = ligands

        # get kernel matrices
        Kp = self._get_protein_kernel(self.train_proteins)
        Kl = np.asarray(self._get_ligand_kernel(self.train_ligands))

        if sample_weight is not None:
            raise ValueError("Sample weighting is not yet supported")

        # define pairwise kernel operator
        pwk_op = PairwiseKernelOperator(
            Kp=Kp,
            Kl=Kl,
            train_protein_ixs=self.train_protein_ixs,
            train_ligand_ixs=self.train_ligand_ixs,
            sample_weight=sample_weight,
        )

        # fit model using conjugate gradient method
        self.params, run_info = sla.minres(
            pwk_op, y, shift=-self.alpha, maxiter=maxiter
        )
        print(f"fitting exited with status {run_info}")
        self.is_fitted = True

        # compute top eigenvalues/vectors if desired
        # (this is computationally intensive)
        if store_eigs:
            self.eigenvalues, self.eigenvectors = sla.eigsh(pwk_op, k=min(500, len(self.train_ligand_ixs)))
        else:
            self.eigenvalues, self.eigenvectors = None, None

        return self

    def predict(self, proteins, ligands, return_variance=False):
        """
        make predictions on new protein/ligand pairs

        Arguments:
            proteins (array): (m, d_p) array of protein features, e.g. AA sequences
            ligands (array): (m, d_l) array of ligand features, e.g. fingerprints

        Returns:
            predictions (array)
        """
        test_proteins, test_protein_ixs = proteins
        test_ligands, test_ligand_ixs = ligands

        # get kernel matrices (note need to convert train ligands back from sparse)
        Kp = self._get_protein_kernel(test_proteins, self.train_proteins)
        Kl = self._get_ligand_kernel(test_ligands, self.train_ligands)

        # define pairwise kernel operator
        pwk_op = PairwiseKernelOperator(
            Kp=Kp,
            Kl=Kl,
            train_protein_ixs=self.train_protein_ixs,
            train_ligand_ixs=self.train_ligand_ixs,
            test_protein_ixs=test_protein_ixs,
            test_ligand_ixs=test_ligand_ixs,
        )

        pred = pwk_op._matvec(self.params)
        if return_variance:
            if self.eigenvectors is None:
                print(
                    "No EVD available for this model's training kernel, not returning variance"
                )
                return pred, None
            else:
                projections = np.array([pwk_op._matvec(v) for v in self.eigenvectors.T])
                var = np.diag(
                    projections.T
                    @ np.diag(1.0 / (self.eigenvalues + self.alpha))
                    @ projections
                )
                var = np.ones_like(var) - var
                return pred, var
        else:
            return pred

    def save(self, filepath, meta=None):
        """
        save model to filepath with optional meta data
        """
        assert self.is_fitted, "need to fit model before saving"

        model_meta = {
            "alpha": self.alpha,
            "train_protein_ixs": self.train_protein_ixs,
            "train_ligand_ixs": self.train_ligand_ixs,
            "train_proteins": self.train_proteins,
            "train_ligands": self.train_ligands,
            "protein_kernel": self.protein_kernel,
            "protein_kernel_power": self.protein_kernel_power,
            "ligand_kernel": self.ligand_kernel,
            "ligand_kernel_power": self.ligand_kernel_power,
            "eigenvalues": self.eigenvalues,
            "eigenvectors": self.eigenvectors,
            "params": self.params,
        }

        if meta is not None:
            model_meta["meta"] = meta

        with open(filepath, "wb") as fp:
            pickle.dump(model_meta, fp, protocol=pickle.HIGHEST_PROTOCOL)

        return model_meta

    @classmethod
    def load(cls, data):
        """
        load model from data
        """
        try:
            obj = cls()
            obj.alpha = data["alpha"]
            obj.train_protein_ixs = data["train_protein_ixs"]
            obj.train_ligand_ixs = data["train_ligand_ixs"]
            obj.train_proteins = data["train_proteins"]
            obj.train_ligands = data["train_ligands"]
            obj.protein_kernel = data.get("protein_kernel", "normalized_ssw")
            obj.protein_kernel_power = data.get("protein_kernel_power", 1.0)
            obj.ligand_kernel = data.get("ligand_kernel", "tanimoto")
            obj.ligand_kernel_power = data.get("ligand_kernel_power", 1.0)
            obj.eigenvalues = data.get("eigenvalues")
            obj.eigenvectors = data.get("eigenvectors")
            obj.params = data["params"]
            obj.is_fitted = True
            obj.timestamp = data.get("timestamp")

            if "meta" in data.keys():
                obj.meta = data["meta"]

            return obj
        except:
            raise Exception("error loading saved model")

    @classmethod
    def load_from_file(cls, filepath):
        """
        load model from filepath
        """

        if os.path.isfile(filepath):
            with open(filepath, "rb") as fp:
                data = pickle.load(fp)
            return cls.load(data)
        else:
            raise ValueError(f"no model found at filepath {filepath}")
