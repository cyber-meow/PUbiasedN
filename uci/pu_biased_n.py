from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import settings


dataset_path = 'data/UCI/abalone.ord'
train_num = 2177
test_num = 2000

print('train_num', train_num)
print('test_num', test_num)
print('')


num_classes = 10


p_num = 50
n_num = 50
sn_num = 50
u_num = 500

pv_num = 50
nv_num = 50
snv_num = 50
uv_num = 500

u_cut = 1500

pi = 0.5
rho = 0.3

positive_classes = [0, 1, 2]

neg_ps = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

non_pu_fraction = 0.5
balanced = False

sep_value = 0.5
adjust_p = True
adjust_sn = True

cls_training_epochs = 100
convex_epochs = 100

p_batch_size = 20
n_batch_size = 25
sn_batch_size = 25
u_batch_size = 250

learning_rate_cls = 1e-2
weight_decay = 1e-4
validation_momentum = 0

non_negative = True
nn_threshold = 0
nn_rate = 1/3

pu_prob_est = True
use_true_post = False

partial_n = True
hard_label = False

iwpn = False
pu = False
pnu = False
unbiased_pn = False

random_seed = 1

sets_save_name = None
sets_load_name = None

ppe_save_name = None
ppe_load_name = None

settings.validation_interval = 1


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
    ('\nsep_value', sep_value),
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
    ('\nsets_save_name', sets_save_name),
    ('sets_load_name', sets_load_name),
    ('ppe_save_name', ppe_save_name),
    ('ppe_load_name', ppe_load_name),
])


data = np.loadtxt(dataset_path)

labels = data[:, -1] - 1
data = data[:, :-1]

priors = []
for i in range(num_classes):
    priors.append(np.sum(labels == i).item()/len(labels))
params['\npriors'] = priors

np.random.seed(random_seed)
print('random_seed', random_seed)
print('')

idxs = np.random.permutation(len(data))

train_data = torch.tensor(data[idxs][:train_num])
train_labels = torch.tensor(labels[idxs][:train_num])

# for i in range(num_classes):
#     print(torch.sum(train_labels == i))

test_data = torch.tensor(data[idxs][train_num:])
test_labels = torch.tensor(labels[idxs][train_num:])


class Net(nn.Module):

    def __init__(self, num_input=10, num_classes=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_input, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
