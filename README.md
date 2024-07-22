# Pairwise Kernel Ridge Regression

This repo implements pairwise kernel ridge regression ("pwkrr") for compound-kinase bioactivity prediction. For a set of training protein-ligand pairs $S = \{(p_i, l_i)\}_{i=1}^n$, the prediction for a new pair $(p,l)$ is defined as

$$
f(p, l) = \sum_i \alpha_i k((p,l), (p_i, l_i)) 
$$

where 

$$
k((p, l), (p', l')) = k_P(p, p')\cdot k_L(l, l')
$$

is the product kernel. Here $k_P$ is taken to be the normalized Striped Smith-Waterman alignment, and $k_L$ is the Tanimoto kernel on molecular fingerprints. The parameters $\alpha_i$ are fit using the conjugate gradient method. 

Our implementation also includes support for uncertainty estimatation, based on the interpretation of a kernel ridge regression model as a Gaussian process. The uncertainty is computed as

$$
s^2(p,l) = 1 - k_S(p, l)^\top(K_S + \lambda I)^{-1}k_S(p, l)
$$

where $k_S(p, l) \in \mathbb{R}^n$ is the vector whose $i$th entry is $k((p,l), (p_i, l_i))$ and $K_S$ is the $n\times n$ training kernel. Since this operation is very expensive for large $n$, our implementation approximates this uncertainty via a rank $m \leq n$ approximation of $K_S$.

## Environment setup

We first need to create a conda environment to run the model in. To do this, run

```bash
conda env create -f env.yml
```

Now activate the environment, and from this directory install the `pwkrr` module.

```bash
conda activate pwkrr 
pip install .
```

## Training a model

An example script to train the model is included in `train_pwkrr.py`. This script requires a config file to run; an example is provided in `example_config.json`, which points to an example train/val split (based on a clustering of the compounds) in the `data/` directory.

To test that the environment is setup and runnning correctly, we can run the training script in test mode with

```bash
python train_pwkrr.py --config example_config.json --test
```

You should see output similar to the following:

```
extracted 96 distinct protein/ligand pairs
fingerprints successfully extracted for [96/96] of the ligands
extracted 93 distinct protein/ligand pairs
fingerprints successfully extracted for [91/91] of the ligands
Training...
fitting exited with status 0
Training completed
Validation completed
RMSE: 1.901
Pearson: 0.245
Spearman: 0.226
```

This samples a tiny subset of the data to use for training. For a full training run, simply omit the `--test` flag. 

## Loading a trained model

After training, the final model will be saved to `checkpoints/<your_run_name>/`. It can be loaded as follows:

```python
from pwkrr import PairwiseKernelRidge

model = PairwiseKernelRidge.load_from_file("checkpoints/<your_run_name>/model.pkl")
```

## References

The cython implementation for the vec trick is based on the implementation in [RLScore](https://github.com/aatapa/RLScore/blob/master/rlscore/utilities/_sampled_kronecker_products.pyx).