import h5py
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

import settings


num_classes = 20
num_input = 300

p_num = 1000
n_num = 1000
sn_num = 1000
u_num = 8000

pv_num = 200
nv_num = 200
snv_num = 200
uv_num = 1600

u_cut = 8000

pi = 0.44
rho = 0.3

positive_classes = [i for i in range(10)]

neg_ps = [0] * 10 + [0.1] * 10


non_pu_fraction = 0.5
balanced = False

u_per = 0.5
adjust_p = True
adjust_sn = True

cls_training_epochs = 50
convex_epochs = 50

p_batch_size = 10
n_batch_size = 10
sn_batch_size = 10
u_batch_size = 120

learning_rate_cls = 1e-3
weight_decay = 1e-4
validation_momentum = 0

start_validation_epoch = 0

non_negative = True
nn_threshold = 0
nn_rate = 1

settings.validation_interval = 50

pu_prob_est = False
use_true_post = False

partial_n = False
hard_label = False

iwpn = False
pu = True
pnu = False
unbiased_pn = False

random_seed = 2


params = OrderedDict([
    ('num_classes', num_classes),
    ('\np_num', p_num),
    ('n_num', n_num),
    ('sn_num', sn_num),
    ('u_num', u_num),
    ('\npv_num', pv_num),
    ('nv_num', nv_num),
    ('snv_num', snv_num),
    ('uv_num', uv_num),
    ('\nu_cut', u_cut),
    ('\npi', pi),
    ('rho', rho),
    ('\npositive_classes', positive_classes),
    ('neg_ps', neg_ps),
    ('\nnon_pu_fraction', non_pu_fraction),
    ('balanced', balanced),
    ('\nu_per', u_per),
    ('adjust_p', adjust_p),
    ('adjust_sn', adjust_sn),
    ('\ncls_training_epochs', cls_training_epochs),
    ('convex_epochs', convex_epochs),
    ('\np_batch_size', p_batch_size),
    ('n_batch_size', n_batch_size),
    ('sn_batch_size', sn_batch_size),
    ('u_batch_size', u_batch_size),
    ('\nlearning_rate_cls', learning_rate_cls),
    ('weight_decay', weight_decay),
    ('validation_momentum', validation_momentum),
    ('\nstart_validation_epoch', start_validation_epoch),
    ('\nnon_negative', non_negative),
    ('nn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
    ('\npu_prob_est', pu_prob_est),
    ('use_true_post', use_true_post),
    ('\npartial_n', partial_n),
    ('hard_label', hard_label),
    ('\niwpn', iwpn),
    ('pu', pu),
    ('pnu', pnu),
    ('unbiased_pn', unbiased_pn),
    ('\nrandom_seed', random_seed),
])


vectorizer = TfidfVectorizer(min_df=0.001)

glove_train_f = h5py.File(
    'data/20newsgroups/20newsgroups_glove_mmm_train.hdf5', 'r')
elmo_train_f = h5py.File(
    'data/20newsgroups/20newsgroups_elmo_train.hdf5', 'r')

train_data = np.concatenate(
                [glove_train_f['data'][:], elmo_train_f['data'][:]],
                axis=1)
train_data = elmo_train_f['data'][:]
train_data = preprocessing.scale(train_data)
newsgroups_train = fetch_20newsgroups(subset='train')
# train_data = vectorizer.fit_transform(newsgroups_train.data).todense()
train_labels = newsgroups_train.target

glove_test_f = h5py.File(
    'data/20newsgroups/20newsgroups_glove_mmm_test.hdf5', 'r')
elmo_test_f = h5py.File(
    'data/20newsgroups/20newsgroups_elmo_test.hdf5', 'r')

test_data = np.concatenate(
                [glove_test_f['data'][:], elmo_test_f['data'][:]],
                axis=1)
test_data = elmo_test_f['data'][:]
test_data = preprocessing.scale(test_data)
newsgroups_test = fetch_20newsgroups(subset='test')
# test_data = vectorizer.transform(newsgroups_test.data).todense()
test_labels = newsgroups_test.target


priors = []
for i in range(num_classes):
    priors.append(
        (np.sum(train_labels == i).item() + np.sum(test_labels == i).item())
        / (len(train_labels) + len(test_labels)))
params['\npriors'] = priors

train_data = torch.tensor(train_data)
train_labels = torch.tensor(train_labels)

test_data = torch.tensor(test_data)
test_labels = torch.tensor(test_labels)


class Net(nn.Module):

    def __init__(self, num_classes=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3072, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, num_classes)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
