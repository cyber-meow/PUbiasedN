import argparse
import numpy as np
import pickle
from collections import OrderedDict

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import training
import settings


p_num = 1000
sn_num = 5000
u_num = 30000

pv_num = 100
snv_num = 500
uv_num = 3000

u_cut = 40000

pi = 0.49
rho = 0.15
sn_prob = 1/2
non_pu_fraction = 0.8
sep_value = 0.3

dre_training_epochs = 50
cls_training_epochs = 100

p_batch_size = 100
sn_batch_size = 500
u_batch_size = 3000

learning_rate_dre = 1e-4
learning_rate_cls = 1e-3
weight_decay = 1e-4
validation_momentum = 0.5

nn_threshold = 0
nn_rate = 1/3

partial_n = True
pu_partial_n = False
partial_n_kai = False
pu_partial_n_kai = False

pu = False
pn = False
pnu = False

sets_save_name = None
sets_load_name = 'pickle/sets_n_rho015.p'

dre_save_name = None
dre_load_name = 'pickle/dre_model_n_rho015.p'


params = OrderedDict([
    ('p_num', p_num),
    ('sn_num', sn_num),
    ('u_num', u_num),
    ('\npv_num', pv_num),
    ('snv_num', snv_num),
    ('uv_num', uv_num),
    ('\npi', pi),
    ('rho', rho),
    ('non_pu_fraction', non_pu_fraction),
    ('sep_value', sep_value),
    ('\ndre_training_epochs', dre_training_epochs),
    ('cls_training_epochs', cls_training_epochs),
    ('\np_batch_size', p_batch_size),
    ('sn_batch_size', p_batch_size),
    ('u_batch_size', u_batch_size),
    ('\nlearning_rate_dre', learning_rate_dre),
    ('learning_rate_cls', learning_rate_cls),
    ('weight_decay', weight_decay),
    ('validation_momentum', validation_momentum),
    ('\nnn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
    ('\npartial_n', partial_n),
    ('pu_partial_n', pu_partial_n),
    ('partial_n_kai', partial_n_kai),
    ('pu_partial_n_kai', pu_partial_n_kai),
    ('\npu', pu),
    ('pn', pn),
    ('pnu', pnu),
    ('\nsets_save_name', sets_save_name),
    ('sets_load_name', sets_load_name),
    ('dre_save_name', dre_save_name),
    ('dre_load_name', dre_load_name),
])

for key, value in params.items():
    print('{}: {}'.format(key, value))
print('')


parser = argparse.ArgumentParser(description='MNIST partial n')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

settings.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor


# torchvision.datasets.MNIST outputs a set of PIL images
# We transform them to tensors
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

# Load and transform data
mnist = torchvision.datasets.MNIST(
    './data/MNIST', train=True, download=True, transform=transform)

mnist_test = torchvision.datasets.MNIST(
    './data/MNIST', train=False, download=True, transform=transform)


def pick_p_data(data, labels, n):
    p_idxs = np.argwhere(labels % 2 == 0).reshape(-1)
    selected_p = np.random.choice(p_idxs, n, replace=False)
    return data[selected_p]


def pick_sn_data(data, labels, n):
    # sn_idxs = np.argwhere(
    #     np.logical_and(labels % 2 == 1, labels < 6)).reshape(-1)
    sn_idxs = np.argwhere(labels % 2 == 1).reshape(-1)
    selected_sn = np.random.choice(sn_idxs, n, replace=False)
    return data[selected_sn]


def pick_u_data(data, n):
    selected_u = np.random.choice(len(data), n, replace=False)
    return data[selected_u]


train_data = mnist.train_data
train_labels = mnist.train_labels
idxs = np.random.permutation(len(train_data))

valid_data = train_data[idxs][u_cut:]
valid_labels = train_labels[idxs][u_cut:]
train_data = train_data[idxs][:u_cut]
train_labels = train_labels[idxs][:u_cut]


if sets_load_name is None:
    p_set = torch.utils.data.TensorDataset(
        pick_p_data(train_data, train_labels, p_num).unsqueeze(1))

    sn_set = torch.utils.data.TensorDataset(
        pick_sn_data(train_data, train_labels, sn_num).unsqueeze(1))

    selected_u = np.random.choice(len(train_data), u_num, replace=False)
    u_set = torch.utils.data.TensorDataset(
        train_data[selected_u].unsqueeze(1))
    u_set_labels = torch.utils.data.TensorDataset(
        train_data[selected_u].unsqueeze(1),
        train_labels[selected_u].unsqueeze(1))

    u_set = torch.utils.data.TensorDataset(
        pick_u_data(train_data, u_num).unsqueeze(1))

    if sets_save_name is not None:
        pickle.dump((p_set, sn_set, u_set), open(sets_save_name, 'wb'))

if sets_load_name is not None:
    p_set, sn_set, u_set = pickle.load(open('sets_n.p', 'rb'))


p_validation = pick_p_data(valid_data, valid_labels, pv_num).unsqueeze(1)
sn_validation = pick_sn_data(valid_data, valid_labels, snv_num).unsqueeze(1)
u_validation = pick_u_data(valid_data, uv_num).unsqueeze(1)


test_data = mnist_test.test_data
test_labels = mnist_test.test_labels

test_posteriors = torch.zeros(test_labels.size())
test_posteriors[test_labels % 2 == 0] = 1
# test_posteriors[test_labels % 2 == 1] = -1
test_posteriors[test_labels == 1] = sn_prob
test_posteriors[test_labels == 3] = sn_prob
test_posteriors[test_labels == 5] = sn_prob
test_posteriors[test_labels == 7] = 0
test_posteriors[test_labels == 9] = 0

test_set_dre = torch.utils.data.TensorDataset(
    test_data.unsqueeze(1), test_posteriors.unsqueeze(1))

t_labels = torch.zeros(test_labels.size())
t_labels[test_labels % 2 == 0] = 1
t_labels[test_labels % 2 == 1] = -1

test_set_cls = torch.utils.data.TensorDataset(
    test_data.unsqueeze(1), t_labels.unsqueeze(1))


class Net(nn.Module):

    def __init__(self, sigmoid_output=False):
        super(Net, self).__init__()
        self.sigmoid_output = sigmoid_output
        self.conv1 = nn.Conv2d(1, 5, 5, 1)
        self.conv2 = nn.Conv2d(5, 10, 5, 1)
        self.fc1 = nn.Linear(4*4*10, 40)
        self.fc2 = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*10)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


if partial_n or pu_partial_n:
    if dre_load_name is None:
        print('')
        model = Net(True).cuda() if args.cuda else Net(True)
        dre = training.PosteriorProbability(
                model, pi=pi, rho=rho,
                lr=learning_rate_dre, weight_decay=weight_decay)
        dre.train(p_set, sn_set, u_set, test_set_dre,
                  p_batch_size, sn_batch_size, u_batch_size,
                  p_validation, sn_validation, u_validation,
                  dre_training_epochs)
        if dre_save_name is not None:
            pickle.dump(dre.model, open(dre_save_name, 'wb'))

    # model = Net().cuda() if args.cuda else Net()
    # dre = training.PUClassifier3(
    #         model, pi=pi, rho=rho,
    #         lr=learning_rate_cls, weight_decay=weight_decay)
    # dre.train(p_set, sn_set, u_set, test_set_dre,
    #           p_batch_size, sn_batch_size, u_batch_size,
    #           p_validation, sn_validation, u_validation, cls_training_epochs)

    if dre_load_name is not None:
        dre_model = pickle.load(open(dre_load_name, 'rb'))

    if partial_n:
        print('')
        model = Net().cuda() if args.cuda else Net()
        cls = training.WeightedClassifier(
                model, dre_model, pi=pi, rho=rho, sep_value=sep_value,
                lr=learning_rate_cls, weight_decay=weight_decay)
        cls.train(p_set, sn_set, u_set, test_set_cls,
                  p_batch_size, sn_batch_size, u_batch_size,
                  p_validation, sn_validation, u_validation,
                  cls_training_epochs)

    if pu_partial_n:
        print('')
        model = Net().cuda() if args.cuda else Net()
        cls = training.PUWeightedClassifier(
                model, dre_model, pi=pi, rho=rho,
                lr=learning_rate_cls, weight_decay=weight_decay,
                weighted_fraction=non_pu_fraction,
                nn=True, nn_threshold=0, nn_rate=nn_rate)
        cls.train(p_set, sn_set, u_set, test_set_cls,
                  p_batch_size, sn_batch_size, u_batch_size,
                  p_validation, sn_validation, u_validation,
                  cls_training_epochs)

if partial_n_kai or pu_partial_n_kai:
    # print('')
    # model = Net(True).cuda() if args.cuda else Net(True)
    # dre = training.PosteriorProbability2(
    #         model, pi=rho,
    #         lr=learning_rate_dre, weight_decay=weight_decay)
    # dre.train(sn_set, u_set, test_set_dre, sn_batch_size, u_batch_size,
    #           sn_validation, u_validation, dre_training_epochs)
    # pickle.dump(dre.model, open('dre_model.p', 'wb'))

    dre_model = pickle.load(open('dre_model.p', 'rb'))

    if partial_n_kai:
        print('')
        model = Net().cuda() if args.cuda else Net()
        cls = training.WeightedClassifier2(
                model, dre_model, pi=pi, rho=rho,
                lr=learning_rate_cls, weight_decay=weight_decay,
                nn=True, nn_threshold=nn_threshold, nn_rate=nn_rate)
        cls.train(p_set, sn_set, u_set, test_set_cls,
                  p_batch_size, sn_batch_size, u_batch_size,
                  p_validation, sn_validation, u_validation,
                  cls_training_epochs)

    if pu_partial_n_kai:
        print('')
        model = Net().cuda() if args.cuda else Net()
        cls = training.PUWeightedClassifier2(
                model, dre_model, pi=pi, rho=rho,
                lr=learning_rate_cls, weight_decay=weight_decay,
                weighted_fraction=non_pu_fraction,
                nn=True, nn_threshold=nn_threshold, nn_rate=nn_rate)
        cls.train(p_set, sn_set, u_set, test_set_cls,
                  p_batch_size, sn_batch_size, u_batch_size,
                  p_validation, sn_validation, u_validation,
                  cls_training_epochs)

if pu:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PUClassifier(
            model, pi=pi, lr=learning_rate_cls, weight_decay=weight_decay,
            nn=True, nn_threshold=0, nn_rate=nn_rate)
    cls.train(p_set, u_set, test_set_cls, p_batch_size, u_batch_size,
              p_validation, u_validation, cls_training_epochs)

if pn:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNClassifier(
            model, pi=pi, lr=learning_rate_cls, weight_decay=weight_decay)
    cls.train(p_set, sn_set, test_set_cls, p_batch_size, sn_batch_size,
              p_validation, sn_validation, cls_training_epochs)

if pnu:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNUClassifier(
            model, pi=pi,
            lr=learning_rate_cls, weight_decay=weight_decay,
            pn_fraction=non_pu_fraction,
            nn=True, nn_threshold=0, nn_rate=nn_rate)
    cls.train(p_set, sn_set, u_set, test_set_cls,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation, cls_training_epochs)
