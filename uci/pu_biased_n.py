import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing

import settings


parser = argparse.ArgumentParser(description='UCI')

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--rho', type=float, default=0)
parser.add_argument('--u_per', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--adjust_p', default=True)
parser.add_argument('--algo', default=0)

args = parser.parse_args()
args.algo = int(args.algo)

if args.adjust_p == 'False':
    args.adjust_p = False


dataset_path = 'data/UCI/bank32nh.ord'
train_num = 4192
test_num = 4000

num_classes = 5
num_input = 32

p_num = 100
n_num = 100
sn_num = 100
u_num = 400

pv_num = 100
nv_num = 100
snv_num = 100
uv_num = 400

u_cut = 2100

pi = 0.4
rho = args.rho

positive_classes = [3, 4]

neg_ps = [0, 0, 1, 0, 0]


non_pu_fraction = args.gamma
balanced = False

u_per = args.u_per
adjust_p = args.adjust_p
adjust_sn = True

cls_training_epochs = 50
convex_epochs = 50

p_batch_size = 10
n_batch_size = 10
sn_batch_size = 10
u_batch_size = 40

learning_rate_cls = args.learning_rate
weight_decay = args.weight_decay
validation_momentum = 0

start_validation_epoch = 0

non_negative = True
nn_threshold = 0
nn_rate = 1

settings.validation_interval = 10

pu_prob_est = args.algo == 0
use_true_post = False

partial_n = args.algo == 0
hard_label = False

iwpn = False
pu = args.algo == 1
pnu = args.algo == 2
unbiased_pn = False

random_seed = args.random_seed

sets_save_name = None
sets_load_name = None

ppe_save_name = None
ppe_load_name = None


print('train_num', train_num)
print('test_num', test_num)
print('')


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
    ('\nsets_save_name', sets_save_name),
    ('sets_load_name', sets_load_name),
    ('ppe_save_name', ppe_save_name),
    ('ppe_load_name', ppe_load_name),
])


data = np.loadtxt(dataset_path)

labels = data[:, -1] - 1
data = data[:, :-1]
data = preprocessing.scale(data)

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

    def __init__(self, num_classes=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_input, num_input*3)
        self.fc2 = nn.Linear(num_input*3, num_input)
        self.fc3 = nn.Linear(num_input, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net2(nn.Module):

    def __init__(self, num_classes=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_input, num_input*2)
        self.fc2 = nn.Linear(num_input*2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
