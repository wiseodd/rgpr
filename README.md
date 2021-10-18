# ReLU-GP Residual (RGPR)

This repository contains code for reproducing [the following NeurIPS 2021 paper](https://arxiv.org/abs/2010.02709):

```
@inproceedings{kristiadi2021infinite,
  title={An infinite-feature extension for {B}ayesian {ReLU} nets that fixes their asymptotic overconfidence},
  author={Kristiadi, Agustinus and Hein, Matthias and Hennig, Philipp},
  booktitle={NeurIPS},
  year={2021}
}
```

## Contents

**General**

* `eval_*.py` scripts are for running experiments.
* `aggregate_*.py` scripts are for processing experiment results, to make them paper-ready.

**RGPR-specific code**

* The implementation of the double-sided cubic spline (DSCS) kernel is in `rgpr/kernel.py`.
* To apply RGPR on a BNN, see the respective prediction code (no retraining required):
  * The `predict` function in `laplace/llla.py` for last-layer Laplace.
  * The `predict` function in `laplace/util.py` for general BNNs (with Monte Carlo sampling).
* To do hyperparameter search for RGPR, see the `get_best_kernel_var` function in `eval_OOD.py`.
* To generate mean and standard deviation of an NN's activations (required by the non-asymptotic extension of RGPR), use `compute_meanstd.py`.

## Running the code:

1. Install dependencies (check `requirements.txt`). We use Python 3.7.
2. Install BackPACK: <https://f-dangel.github.io/backpack/>.
3. In `util/dataloaders.py`, change `path = ...` to your liking.

Pre-trained models are in <https://nc.mlcloud.uni-tuebingen.de/index.php/s/NknifES7G9MBrAB>
