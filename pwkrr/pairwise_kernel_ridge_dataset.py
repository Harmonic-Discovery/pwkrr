import scipy.sparse as sp
import numpy as np
import pandas as pd
import os
import json

from ._utils import FP_FACTORY

_base_dir = os.path.split(__file__)[0]


class PairwiseKernelRidgeDataset:
    """
    Utility class that is used for ML models that use fingerprints as features
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        smiles_column: str = "canonical_smiles",
        prot_id_column: str = "uniprot_id",
        labels_column: str = "summarized_activity_value",
        n_workers: int = 1,
        verbose: bool = True,
        ligand_features: list = [("morgan", {"radius": 2, "n_bits": 1024})],
    ):
        """
        Arguments:
            dataset (DataFrame)
            smiles_column (str): name of column containing smiles
            prot_id_column (str): name of column containing uniprot ids
            labels_column (str): the column where the labels are stored (if any)
            n_workers (int): number of processes to spawn when generating the fingerprints
            ligand_features (dict): list of tuples defining fingerprints to be used for ligands,
                e.g. [("morgan", {"radius": 2, "n_bits": 1024})]

        Attributes:
            ligands (tuple): tuple containing (ligands, ligand_ixs), where
                ligands contain fingerprints of distinct ligands, and ligand_ixs
                are the corresponding indices of the ligands in the dataset
            proteins (tuple): tuple containing (proteins, protein_ixs), where
                proteins contain active site sequences of distinct proteins, and
                protein_ixs are the corresponding indices of the proteins in the dataset
            labels (array): array containing labels
            smiles (list): list containing smiles strings extracted from the dataset
        """

        assert (smiles_column in dataset.columns) and (
            prot_id_column in dataset.columns
        ), "dataset needs to contain smiles_column and prot_id_column"

        self.dataset = dataset
        self.smiles_column = smiles_column
        self.prot_id_column = prot_id_column
        self.labels_column = labels_column
        self.verbose = verbose

        # get protein features
        with open(os.path.join(_base_dir, "kinase_active_site_sequences.json")) as f:
            protein_features = json.load(f)

        self.dataset = self.dataset[
            self.dataset[self.prot_id_column].isin(protein_features.keys())
        ]
        self.available_uniprot_ids = protein_features.keys()
        # extract the indices of the proteins and ligands for each observation in the dataset
        self.protein_ixs = self.dataset.groupby(self.prot_id_column).ngroup().to_numpy()

        if self.verbose:
            print(f"extracted {len(self.protein_ixs)} distinct protein/ligand pairs")

        self.uniprot_ids = self.dataset[self.prot_id_column].tolist()
        self.proteins = [
            protein_features[id]
            for id in self.dataset.groupby(self.prot_id_column)
            .first()
            .reset_index()[self.prot_id_column]
            .tolist()
        ]

        # get ligand features
        self.ligand_ixs = self.dataset.groupby(self.smiles_column).ngroup().to_numpy()

        self.smiles = (
            self.dataset.groupby(self.smiles_column)
            .first()
            .reset_index()[self.smiles_column]
            .to_numpy()
        )

        # extract the different fingerprint types
        self.ligands = []
        for fp_type, fp_params in ligand_features:
            curr, ix = FP_FACTORY[fp_type](
                self.smiles, n_workers=n_workers, **fp_params
            )
            self.ligands.append(curr)
        self.ligands = sp.hstack(self.ligands, format="csr")

        # get labels
        if self.labels_column in self.dataset.columns:
            self.labels = self.dataset[self.labels_column].to_numpy()
        else:
            self.labels = np.array([None] * len(self.ligand_ixs))
            if self.verbose:
                print("No labels found in dataset")

        # remove obs that don't have valid fps
        self.labels = self.labels[np.isin(self.ligand_ixs, ix)]
        self.protein_ixs = self.protein_ixs[np.isin(self.ligand_ixs, ix)]
        self.ligand_ixs = self.ligand_ixs[np.isin(self.ligand_ixs, ix)]

        assert len(self.protein_ixs) == len(self.ligand_ixs)

        self.ligands = (self.ligands, self.ligand_ixs)
        self.proteins = (self.proteins, self.protein_ixs)

        if verbose:
            print(
                f"fingerprints successfully extracted for [{len(ix)}/{len(self.smiles)}] of the ligands"
            )

    def __len__(self):
        return len(self.ligand_ixs)
