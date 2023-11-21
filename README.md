# PUbiasedN - Forked from https://github.com/cyber-meow/PUbiasedN

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

where `dataset` is either `mnist` and `fashion_mnist` (Only these two will be rerun) and
`parameter-path` is a `yml` file containing the hyperparameters of the experiment.
The hyperparameter files used for the results shown in Table 1 can be found under
the `params/` directory.

## Code Reuse
1. The only changes made in the code was the addition of the Fashion MNIST Experiment

## Desired Direction
1. Implement the same PUbN algorithm on a non-benchmark dataset preferably a medical dataset
2. Estimate Class Prior using either of the TiCE (Estimation of Class Prior using Decision Tree Induction )

## Attempts (These were not included in the paper due to time constraints)
1. Integrating Tice method to estimate class prior on UCI Breast Cancer Dataset but was not able to run PUbN experiments successfully. Kept giving series of error which could not be resolved in time.
2. Added the Food101 dataset but pre-processing kept failing. 
