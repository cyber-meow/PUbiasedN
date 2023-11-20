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

## Implement on Google Collab
1. Download the above Project.ipnyb file and add it to a directory on Google Drive
2. Change the path in Cell 3 as indicated to the directory in step 1.

where `dataset` is either `mnist` and `fashion_mnist` and
`parameter-path` is a `yml` file containing the hyperparameters of the experiment.
The hyperparameter files used for the results shown in Table 1 can be found under
the `params/` directory.
