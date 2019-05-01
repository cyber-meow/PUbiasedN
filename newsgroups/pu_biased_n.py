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


extracted = 'elmo'
# extracted = 'glove+elmo'
# extracted = 'tfidf'
linear_model = False

cbs_feature = False
cbs_feature_later = False
n_select_features = 2400
cbs_alpha = 6
cbs_beta = 4
svm = False
svm_C = 1

num_classes = 20
# num_input = 10000
num_input = 9216

p_num = 500
n_num = 500
sn_num = 500
u_num = 6000

pv_num = 100
nv_num = 100
snv_num = 100
uv_num = 1200

u_cut = 8000
# u_cut = 11000

pi = 0.56
# true_rho = 0.17
# rho = 0.21
rho = 0.17
# rho = 0.1
true_rho = rho

positive_classes = [i for i in range(11)]

# neg_ps = [0] * 11 + [0.25] * 4 + [0] * 5
neg_ps = [0] * 16 + [0.28, 0.29, 0.24, 0.19]
# neg_ps = [0] * 11 + [0.025] * 4 + [0.5] + [0.112, 0.116, 0.096, 0.076]

non_pu_fraction = 0.5
balanced = False

u_per = 0.7
adjust_p = True
adjust_sn = True

cls_training_epochs = 50
convex_epochs = 50

p_batch_size = 10
n_batch_size = 10
sn_batch_size = 10
u_batch_size = 120

learning_rate_ppe = 1e-3
learning_rate_cls = 5e-3
weight_decay = 1e-4

milestones = [200]
milestones_ppe = [200]
lr_d = 0.1

non_negative = True
nn_threshold = 0
nn_rate = 1

settings.validation_interval = 50

pu_prob_est = True
use_true_post = False

partial_n = True
hard_label = False

iwpn = False
pu = False
pnu = False
unbiased_pn = False

random_seed = 0

ppe_save_name = None
# ppe_save_name = 'weights/newsgroups/imbN/500-100-400-200_lr1e-3_3'
ppe_load_name = None


params = OrderedDict([
    ('extracted', extracted),
    ('linear_model', linear_model),
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
    ('true_rho', true_rho),
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
    ('learning_rate_ppe', learning_rate_ppe),
    ('weight_decay', weight_decay),
    ('milestones', milestones),
    ('milestones_ppe', milestones_ppe),
    ('lr_d', lr_d),
    ('\nnon_negative', non_negative),
    ('nn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
    ('\ncbs_feature', cbs_feature),
    ('cbs_feature_later', cbs_feature_later),
    ('cbs_alpha', cbs_alpha),
    ('cbs_beta', cbs_beta),
    ('n_select_features', n_select_features),
    ('svm', svm),
    ('svm_C', svm_C),
    ('\npu_prob_est', pu_prob_est),
    ('use_true_post', use_true_post),
    ('\npartial_n', partial_n),
    ('hard_label', hard_label),
    ('\niwpn', iwpn),
    ('pu', pu),
    ('pnu', pnu),
    ('unbiased_pn', unbiased_pn),
    ('\nrandom_seed', random_seed),
    ('\nppe_save_name', ppe_save_name),
    ('ppe_load_name', ppe_load_name),
])


newsgroups_train = fetch_20newsgroups(subset='train')
train_labels = newsgroups_train.target
newsgroups_test = fetch_20newsgroups(subset='test')
test_labels = newsgroups_test.target


glove_train_f = h5py.File(
    'data/20newsgroups/20newsgroups_glove_mmm_train.hdf5', 'r')
elmo_train_f = h5py.File(
    'data/20newsgroups/20newsgroups_elmo_mmm_train.hdf5', 'r')

glove_test_f = h5py.File(
    'data/20newsgroups/20newsgroups_glove_mmm_test.hdf5', 'r')
elmo_test_f = h5py.File(
    'data/20newsgroups/20newsgroups_elmo_mmm_test.hdf5', 'r')


if extracted == 'elmo':
    train_data = elmo_train_f['data'][:]
    test_data = elmo_test_f['data'][:]
    train_data = preprocessing.scale(train_data)
    test_data = preprocessing.scale(test_data)


if extracted == 'glove+elmo':
    train_data = np.concatenate(
                    [glove_train_f['data'][:], elmo_train_f['data'][:]],
                    axis=1)
    test_data = np.concatenate(
                    [glove_test_f['data'][:], elmo_test_f['data'][:]],
                    axis=1)
    train_data = preprocessing.scale(train_data)
    test_data = preprocessing.scale(test_data)


if extracted == 'tfidf':
    vectorizer = TfidfVectorizer(max_features=num_input,
                                 max_df=0.95, stop_words='english')
    # vectorizer2 = TfidfVectorizer(max_features=num_input,
    #                               ngram_range=(2, 2),
    #                               max_df=0.95, stop_words='english')
    # vectorizer3 = TfidfVectorizer(max_features=num_input,
    #                               ngram_range=(3, 3),
    #                               max_df=0.95, stop_words='english')

    train_data = vectorizer.fit_transform(newsgroups_train.data).toarray()
    # train_data2 = vectorizer2.fit_transform(newsgroups_train.data).toarray()
    # train_data3 = vectorizer3.fit_transform(newsgroups_train.data).toarray()
    # train_data = np.concatenate([
    #     np.expand_dims(train_data, 1),
    #     np.expand_dims(train_data2, 1),
    #     np.expand_dims(train_data3, 1)], axis=1)

    test_data = vectorizer.transform(newsgroups_test.data).toarray()
    # test_data2 = vectorizer2.transform(newsgroups_test.data).toarray()
    # test_data3 = vectorizer3.transform(newsgroups_test.data).toarray()
    # test_data = np.concatenate([
    #     np.expand_dims(test_data, 1),
    #     np.expand_dims(test_data2, 1),
    #     np.expand_dims(test_data3, 1)], axis=1)


priors = []
for i in range(num_classes):
    priors.append(
        (np.sum(train_labels == i).item() + np.sum(test_labels == i).item())
        / (len(train_labels) + len(test_labels)))
params['\npriors'] = priors

if not cbs_feature:
    train_data = torch.tensor(train_data)
train_labels = torch.tensor(train_labels)

if not cbs_feature:
    test_data = torch.tensor(test_data)
test_labels = torch.tensor(test_labels)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        if linear_model:
            self.fc1 = nn.Linear(num_input, 1)
        else:
            self.fc1 = nn.Linear(num_input, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x):
        if linear_model:
            return self.fc1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NetCBS(nn.Module):

    def __init__(self):
        super(NetCBS, self).__init__()
        self.fc1 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x
