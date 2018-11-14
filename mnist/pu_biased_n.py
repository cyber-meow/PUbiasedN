from collections import OrderedDict

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import settings


num_classes = 10

p_num = 500
sn_num = 500
n_num = 500
u_num = 6000

pv_num = 100
nv_num = 100
snv_num = 100
uv_num = 1200

u_cut = 40000

pi = 0.49
# pi = 0.097
true_rho = 0.25
rho = 0.25

positive_classes = [0, 2, 4, 6]
negative_classes = [1, 3, 5, 7]

# neg_ps = [0, 0.03, 0, 0.15, 0, 0.3, 0, 0.02, 0, 0.5]
# neg_ps = [0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2]
# neg_ps = [0, 1/3, 0, 1/3, 0, 1/3, 0, 0, 0, 0]
# neg_ps = [1/6, 1/6, 1/6, 1/6, 0, 1/6, 1/6, 0, 0, 0]
neg_ps = [0, 0, 0, 0, 0, 1/4, 0, 1/4, 1/4, 1/4]

non_pu_fraction = 0.7
balanced = False

u_per = 0.7
adjust_p = True
adjust_sn = True

cls_training_epochs = 100
convex_epochs = 100

p_batch_size = 10
n_batch_size = 10
sn_batch_size = 10
u_batch_size = 120

learning_rate_ppe = 1e-3
learning_rate_cls = 1e-3
weight_decay = 1e-4
validation_momentum = 0

start_validation_epoch = 0
milestones = [200]

non_negative = True
nn_threshold = 0
nn_rate = 1

settings.validation_interval = 50

pu_prob_est = True
use_true_post = False

partial_n = True
hard_label = False

pn_then_pu = False
pu_then_pn = False
iwpn = False
pu = False
pnu = False
unbiased_pn = False

vat = False
ent = False

alpha = 1
beta = 0.8

random_seed = 10

sets_save_name = None
sets_load_name = None

ppe_save_name = None
# ppe_load_name = 'weights/MNIST/135N/500P+500N_135N_1e-3_1'
ppe_load_name = None


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
    ('true_rho', true_rho),
    ('\npositive_classes', positive_classes),
    ('negative_classes', negative_classes),
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
    ('validation_momentum', validation_momentum),
    ('\nstart_validation_epoch', start_validation_epoch),
    ('milestones', milestones),
    ('\nnon_negative', non_negative),
    ('nn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
    ('\npu_prob_est', pu_prob_est),
    ('use_true_post', use_true_post),
    ('\npartial_n', partial_n),
    ('hard_label', hard_label),
    ('\niwpn', iwpn),
    ('pn_then_pu', pn_then_pu),
    ('pu_then_pn', pu_then_pn),
    ('pu', pu),
    ('pnu', pnu),
    ('unbiased_pn', unbiased_pn),
    ('\nvat', vat),
    ('ent', ent),
    ('alpha', alpha),
    ('beta', beta),
    ('\nrandom_seed', random_seed),
    ('\nsets_save_name', sets_save_name),
    ('sets_load_name', sets_load_name),
    ('ppe_save_name', ppe_save_name),
    ('ppe_load_name', ppe_load_name),
])


# torchvision.datasets.MNIST outputs a set of PIL images
# We transform them to tensors
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load and transform data
mnist = torchvision.datasets.MNIST(
    './data/MNIST', train=True, download=True, transform=transform)

mnist_test = torchvision.datasets.MNIST(
    './data/MNIST', train=False, download=True, transform=transform)


train_data = torch.zeros(mnist.train_data.size())

for i, (image, _) in enumerate(mnist):
    train_data[i] = image

train_data = train_data.unsqueeze(1)
train_labels = mnist.train_labels

test_data = torch.zeros(mnist_test.test_data.size())

for i, (image, _) in enumerate(mnist_test):
    test_data[i] = image

test_data = test_data.unsqueeze(1)
test_labels = mnist_test.test_labels

# for i in range(10):
#     print(torch.sum(train_labels == i))


class Net(nn.Module):

    def __init__(self, num_classes=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, 1)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 10, 5, 1)
        self.bn2 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(4*4*10, 40)
        self.fc2 = nn.Linear(40, num_classes)

    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*10)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def call_bn(bn, x, update_batch_stats=True):
    if bn.training is False:
        return bn(x)
    elif not update_batch_stats:
        return F.batch_norm(x, None, None, bn.weight, bn.bias, True,
                            bn.momentum, bn.eps)
    else:
        return bn(x)


# Following this repository
# https://github.com/musyoku/vat
class FcNet(nn.Module):
    def __init__(self, n_class=1, n_ch=1, res=28):
        super(FcNet, self).__init__()
        self.input_len = n_ch * res * res
        self.fc1 = nn.Linear(self.input_len, 1200)
        self.fc2 = nn.Linear(1200, 600)
        self.fc3 = nn.Linear(600, n_class)

        self.bn_fc1 = nn.BatchNorm1d(1200)
        self.bn_fc2 = nn.BatchNorm1d(600)

    def __call__(self, x, update_batch_stats=True):
        h = F.relu(call_bn(self.bn_fc1, self.fc1(x.view(-1, self.input_len)),
                           update_batch_stats))
        h = F.relu(call_bn(self.bn_fc2, self.fc2(h), update_batch_stats))
        return self.fc3(h)
