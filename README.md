# PUbiasedN

PyTorch implementation to reproduce experiments in the paper
[Classification from Positive, Unlabeled and Biased Negative Data](https://arxiv.org/abs/1810.00846).

## Requirements
1. PyTorch >= 0.4.0, scikit-learn, NumPy
2. yaml to load parameters
3. nltk, allennlp, h5py if you need to prepare the 20newsgroups ELMO embedding

## Disclaimer

This repository is still in progress.
Though the code provided here should allow one to reproduce the main experiments of the paper,
the documentation is not yet complete and the parameters of the experiments have not all been updated.
The uci directory is deprecated and may not work with the current codes.

## Usage

To reproduce the MNIST experiments shown in Table 1 of the paper:

```
python pu_biased_n.py --dataset mnist --params-path [parameter-path] --random-seed [random-seed]
```

where `parameter-path` is a `yml` file containing the parameters that are used for
some experiment of the paper.
For MNIST, the parameter files can be found in the `mnist/params` directory.
