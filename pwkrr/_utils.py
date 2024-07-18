from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from joblib import Parallel, delayed
import json

# from IPython import embed


def gobbi_fingerprints_from_smiles(smiles, n_workers=1, **kwargs):
    """
    computes Gobbi fingerprints from smiles

    Arguments:
        smiles (list): list of smiles strings
        as_csr (bool): whether to store the fingerprints in csr format

    Returns:
        fingerprints (array): array of numpy fingerprints
        indices (list): list of indices of valid original smiles. some smiles may
        cause errors when obtaining fingerprints, so we store which ones are successfully
        computed
    """
    smiles = np.array(smiles)

    def get_fp(smi, ind):
        ixs = []
        row_ind, col_ind = [], []
        for ix, smile in enumerate(smi):
            try:
                m = Chem.MolFromSmiles(smile)
                fp = Generate.Gen2DFingerprint(m, Gobbi_Pharm2D.factory)
                on_bits = list(fp.GetOnBits())
                col_ind.extend(on_bits)
                row_ind.extend([ix] * len(on_bits))
                ixs.append(ix)
            except:
                continue

        data = [1.0] * len(col_ind)
        fingerprints = csr_matrix((data, (row_ind, col_ind)), shape=(len(ixs), 39972))

        if not kwargs.get("as_csr", True):
            fingerprints = fingerprints.toarray()

        return fingerprints, ind[ixs]

    indices = np.arange(len(smiles))
    tmp = Parallel(n_jobs=n_workers, max_nbytes=None)(
        delayed(get_fp)(smiles[jx], jx) for jx in np.array_split(indices, n_workers)
    )
    fingerprints = sp.vstack([l for l, _ in tmp], format="csr")
    # ix contain the indices of the ligands that we were able to extract fps for
    ix = np.concatenate([i for _, i in tmp], axis=None)

    return fingerprints, ix


def rdk_fingerprints_from_smiles(smiles, n_workers=1, **kwargs):
    """
    computes RDKit fingerprints from smiles

    Arguments:
        smiles (list): list of smiles strings
        as_csr (bool): whether to store the fingerprints in csr format
        n_bits (int): number of bits for fingerprints

    Returns:
        fingerprints (array): array of numpy fingerprints
        indices (list): list of indices of valid original smiles. some smiles may
        cause errors when obtaining fingerprints, so we store which ones are successfully
        computed
    """
    smiles = np.array(smiles)

    def get_fp(smi, ind):
        ixs = []
        fingerprints = []
        for ix, s in enumerate(smi):
            try:
                m = Chem.MolFromSmiles(s)
                fp = list(Chem.RDKFingerprint(m, fpSize=kwargs.get("n_bits", 1024)))
                fingerprints.append(fp)
                ixs.append(ix)
            except:
                continue
        fingerprints = np.array(fingerprints)

        if kwargs.get("as_csr", True):
            fingerprints = csr_matrix(fingerprints)

        return fingerprints, ind[ixs]

    indices = np.arange(len(smiles))
    tmp = Parallel(n_jobs=n_workers, max_nbytes=None)(
        delayed(get_fp)(smiles[jx], jx) for jx in np.array_split(indices, n_workers)
    )
    fingerprints = sp.vstack([l for l, _ in tmp], format="csr")
    # ix contain the indices of the ligands that we were able to extract fps for
    ix = np.concatenate([i for _, i in tmp], axis=None)

    return fingerprints, ix


def morgan_fingerprints_from_smiles(smiles, n_workers=1, **kwargs):
    """
    Generates morgan fingerprints from a list of smiles strings

    Arguments:
        smiles (list): list of smiles strings
        as_csr (bool): whether to store the fingerprints in csr format
        radius (int): radius of morgan fingerprints
        n_bits (int): number of bits for morgan fingerprints
        n_workers (int): number of workers for extracting fingerprints in parallel

    Returns:
        fingerprints (array): array of numpy fingerprints
        indices (list): list of indices of valid original smiles. some smiles may
        cause errors when obtaining fingerprints, so we store which ones are successfully
        computed
    """
    smiles = np.array(smiles)

    def get_fp(smi, ind):
        ixs = []
        fingerprints = []
        for ix, s in enumerate(smi):
            try:
                m = Chem.MolFromSmiles(s)
                fp = list(
                    AllChem.GetMorganFingerprintAsBitVect(
                        m,
                        radius=kwargs.get("radius", 2),
                        nBits=kwargs.get("n_bits", 1024),
                        useChirality=kwargs.get("use_chirality", False),
                    )
                )
                fingerprints.append(fp)
                ixs.append(ix)
            except:
                continue
        fingerprints = np.array(fingerprints)

        if kwargs.get("as_csr", True):
            fingerprints = csr_matrix(fingerprints)

        return fingerprints, ind[ixs]

    indices = np.arange(len(smiles))
    tmp = Parallel(n_jobs=n_workers, max_nbytes=None)(
        delayed(get_fp)(smiles[jx], jx) for jx in np.array_split(indices, n_workers)
    )
    fingerprints = sp.vstack([l for l, _ in tmp], format="csr")
    # ix contain the indices of the ligands that we were able to extract fps for
    ix = np.concatenate([i for _, i in tmp], axis=None)

    return fingerprints, ix


# this is a simple dictionary (usually called a "factory") which can specify a fingerprint generation method
# by keyword
FP_FACTORY = {
    "gobbi": gobbi_fingerprints_from_smiles,
    "rdkit": rdk_fingerprints_from_smiles,
    "morgan": morgan_fingerprints_from_smiles,
}
