from collections import OrderedDict

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

import settings
from cifar10.nets import PreActResNet18


num_classes = 10

p_num = 1000
sn_num = 1000
u_num = 10000

pv_num = 200
snv_num = 200
uv_num = 2000

u_cut = 40000

pi = 0.4
true_rho = 0.3
rho = 0.3

# positive_classes = [3, 4, 5, 7]
# positive_classes = [1]
positive_classes = [0, 1, 8, 9]

# neg_ps = [0, 0, 1/2, 0, 0, 0, 1/2, 0, 0, 0]
# neg_ps = [0, 1/2, 0, 0, 0, 0, 0, 0, 0, 1/2]
neg_ps = [0, 0, 0, 1/3, 0, 1/3, 0, 1/3, 0, 0]
# neg_ps = [0, 0, 0.1, 0.02, 0.2, 0.08, 0.2, 0.4, 0, 0]
# neg_ps = [0, 0, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0, 0]


non_pu_fraction = 0.5
balanced = False

u_per = 0.7
adjust_p = True
adjust_sn = True

cls_training_epochs = 200
convex_epochs = 200

p_batch_size = 10
sn_batch_size = 10
u_batch_size = 100

learning_rate_ppe = 1e-3
learning_rate_cls = 1e-3
weight_decay = 1e-4
validation_momentum = 0

start_validation_epoch = 0
milestones = [80, 120]
lr_d = 0.1

non_negative = True
nn_threshold = 0
nn_rate = 1

pu_prob_est = True
use_true_post = False

partial_n = True
hard_label = False

pu_then_pn = False
iwpn = False
pu = False
pnu = False

vat = True

random_seed = 0

settings.test_batch_size = 500
settings.validation_interval = 100


params = OrderedDict([
    ('num_classes', num_classes),
    ('\np_num', p_num),
    ('sn_num', sn_num),
    ('u_num', u_num),
    ('\npv_num', pv_num),
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
    ('sn_batch_size', sn_batch_size),
    ('u_batch_size', u_batch_size),
    ('\nlearning_rate_cls', learning_rate_cls),
    ('learning_rate_ppe', learning_rate_ppe),
    ('weight_decay', weight_decay),
    ('validation_momentum', validation_momentum),
    ('\nstart_validation_epoch', start_validation_epoch),
    ('milestones', milestones),
    ('lr_d', lr_d),
    ('\nnon_negative', non_negative),
    ('nn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
    ('\npu_prob_est', pu_prob_est),
    ('use_true_post', use_true_post),
    ('\npartial_n', partial_n),
    ('hard_label', hard_label),
    ('\niwpn', iwpn),
    ('pu_then_pn', pu_then_pn),
    ('pu', pu),
    ('pnu', pnu),
    ('\nvat', vat),
    ('\nrandom_seed', random_seed),
])


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Load and transform data
cifar10 = torchvision.datasets.CIFAR10(
    './data/CIFAR10', train=True, download=True, transform=transform)

cifar10_test = torchvision.datasets.CIFAR10(
    './data/CIFAR10', train=False, download=True, transform=transform)


train_data = torch.zeros(cifar10.train_data.shape)
train_data = train_data.permute(0, 3, 1, 2)
# must use one dimensional vector
train_labels = torch.tensor(cifar10.train_labels)

for i, (image, _) in enumerate(cifar10):
    train_data[i] = image

test_data = torch.zeros(cifar10_test.test_data.shape)
test_data = test_data.permute(0, 3, 1, 2)
test_labels = torch.tensor(cifar10_test.test_labels)

for i, (image, _) in enumerate(cifar10_test):
    test_data[i] = image

Net = PreActResNet18
