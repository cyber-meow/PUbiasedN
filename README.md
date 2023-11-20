# PUbiasedN - Forked from https://github.com/cyber-meow/PUbiasedN for class project

PyTorch implementation for experiments in the paper
[Classification from Positive, Unlabeled and Biased Negative Data](https://arxiv.org/abs/1810.00846).

[//]: # (## Citation)

[//]: # (If you find this repository useful, please cite our paper)

[//]: # (```)
[//]: # (@inproceedings{hsieh2018classification,)
[//]: # (  title={Classification from Positive, Unlabeled and Biased Negative Data},)
[//]: # (  author={Hsieh, Yu-Guan and Niu, Gang and Sugiyama, Masashi},)
[//]: # (  booktitle = {International Conference on Machine Learning ICML})
[//]: # (  pages     = {4864--4873},)
[//]: # (  year      = {2019},)
[//]: # (})
[//]: # (``` )

## Requirements
1. Python >= 3.6
2. PyTorch >= 0.4.0, scikit-learn, NumPy
3. yaml to load parameters
4. nltk, allennlp, h5py to prepare the 20newsgroups ELMO embedding

## Usage

The file `pu_biased_n.py` allows to reproduce most of the experimental results
described in the paper:

```
python(3) pu_biased_n.py --dataset [dataset] --params-path [parameter-path] --random-seed [random-seed]
```

where `dataset` is either `mnist`, `cifar10` or `newsgroups` and
`parameter-path` is a `yml` file containing the hyperparameters of the experiment.
The hyperparameter files used for the results shown in Table 1 can be found under
the `params/` directory.

## 20newgroups preprocessing

To prepare the ELMO embedding of the 20newsgroups dataset. Please download the ELMO 5.5B pre-trained model from https://allennlp.org/elmo (elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights) and put it under `data/20newsgroups/`; then run the two files `train_elmo_prepare.py` and `test_elmo_prepare.py` located in this same directory.
