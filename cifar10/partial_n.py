import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import training
import settings
from nets import PreActResNet18


p_num = 16000
n_num = 24000
sn_num = 1000
u_num = 40000

pv_num = 3000
nv_num = 4500
snv_num = 100
uv_num = 4000

u_cut = 42000

pi = 0.4
rho = 0.15

dre_training_epochs = 60
cls_training_epochs = 100

p_batch_size = 100
n_batch_size = 100
sn_batch_size = 100
u_batch_size = 1000

learning_rate_dre = 1e-4
learning_rate_cls = 1e-3
weight_decay = 1e-4
validation_momentum = 0.5


params = OrderedDict([
    ('p_num', p_num),
    ('n_num', n_num),
    ('sn_num', sn_num),
    ('u_num', u_num),
    ('\npv_num', pv_num),
    ('snv_num', snv_num),
    ('nv_num', nv_num),
    ('uv_num', uv_num),
    ('\npi', pi),
    ('rho', rho),
    ('\ndre_training_epochs', dre_training_epochs),
    ('cls_training_epochs', cls_training_epochs),
    ('\np_batch_size', p_batch_size),
    ('sn_batch_size', sn_batch_size),
    ('n_batch_size', n_batch_size),
    ('u_batch_size', u_batch_size),
    ('\nlearning_rate_dre', learning_rate_dre),
    ('learning_rate_cls', learning_rate_cls),
    ('weight_decay', weight_decay),
    ('validation_momentum', validation_momentum),
])


for key, value in params.items():
    print('{}: {}'.format(key, value))
print('')


parser = argparse.ArgumentParser(description='CIFAR10')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

settings.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Load and transform data
cifar10 = torchvision.datasets.CIFAR10(
    './data/CIFAR10', train=True, download=True, transform=transform)

cifar10_test = torchvision.datasets.CIFAR10(
    './data/CIFAR10', train=False, download=True, transform=transform)


def pick_p_data(data, labels, n):
    p_idxs = np.argwhere(
        np.logical_or(labels < 2, labels > 7)).reshape(-1)
    selected_p = np.random.choice(p_idxs, n, replace=False)
    return data[selected_p]


def pick_n_data(data, labels, n):
    n_idxs = np.argwhere(
        np.logical_and(labels >= 2, labels <= 7)).reshape(-1)
    selected_n = np.random.choice(n_idxs, n, replace=False)
    return data[selected_n]


def pick_u_data(data, n):
    selected_u = np.random.choice(len(data), n, replace=False)
    return data[selected_u]


train_data = torch.zeros(cifar10.train_data.shape)
train_data = train_data.permute(0, 3, 1, 2)
train_labels = torch.tensor(cifar10.train_labels)

for i, (image, _) in enumerate(cifar10):
    train_data[i] = image

idxs = np.random.permutation(len(train_data))

valid_data = train_data[idxs][u_cut:]
valid_labels = train_labels[idxs][u_cut:]
train_data = train_data[idxs][:u_cut]
train_labels = train_labels[idxs][:u_cut]


p_set = torch.utils.data.TensorDataset(
    pick_p_data(train_data, train_labels, p_num))

n_set = torch.utils.data.TensorDataset(
    pick_n_data(train_data, train_labels, n_num))

u_set = torch.utils.data.TensorDataset(pick_u_data(train_data, u_num))


p_validation = pick_p_data(valid_data, valid_labels, pv_num)
n_validation = pick_n_data(valid_data, valid_labels, nv_num)
u_validation = pick_u_data(valid_data, uv_num)


test_data = torch.zeros(cifar10_test.test_data.shape)
test_data = test_data.permute(0, 3, 1, 2)
test_labels = torch.tensor(cifar10_test.test_labels)

for i, (image, _) in enumerate(cifar10_test):
    test_data[i] = image

test_labels[test_labels < 2] = 1
test_labels[test_labels > 7] = 1
test_labels[test_labels != 1] = -1

test_set = torch.utils.data.TensorDataset(
    test_data, test_labels.unsqueeze(1))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x


print('')
# model = PreActResNet18().cuda() if args.cuda else PreActResNet18()
model = Net().cuda() if args.cuda else Net()
cls = training.PNClassifier(
        model, pi=pi, lr=learning_rate_cls, weight_decay=weight_decay)
cls.train(p_set, n_set, test_set, p_batch_size, n_batch_size,
          p_validation, n_validation, cls_training_epochs)
