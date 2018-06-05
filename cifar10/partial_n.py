import argparse
import numpy as np
import pickle
from collections import OrderedDict

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

import training
import settings
from nets import PreActResNet18


p_num = 3000
n_num = 3000
sn_num = 3000
u_num = 30000

pv_num = 300
nv_num = 300
snv_num = 300
uv_num = 3000

u_cut = 45000

pi = 0.49
rho = 0.2

neg_ps = [0, 1/3, 0, 1/3, 0, 1/3]

non_pu_fraction = 0.6

sep_value = 0.3
adjust_p = False
adjust_sn = True

dre_training_epochs = 20
cls_training_epochs = 200

p_batch_size = 100
n_batch_size = 100
sn_batch_size = 100
u_batch_size = 1000

learning_rate_dre = 1e-4
learning_rate_cls = 1e-3
weight_decay = 1e-4
validation_momentum = 0.5

nn_threshold = 0
nn_rate = 1/3

partial_n = False
pu_then_pn = False

pu = False
unbiased_pn = False
iwpn = False
pnu = True

sets_save_name = None
sets_load_name = 'pickle/cifar10/3000_3000_30000/sets_rho015_357N.p'

dre_save_name = None
dre_load_name = 'pickle/cifar10/3000_3000_30000/dre_model_rho015_357N.p'


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
    ('neg_ps', neg_ps),
    ('non_pu_fraction', non_pu_fraction),
    ('\nsep_value', sep_value),
    ('adjust_p', adjust_p),
    ('adjust_sn', adjust_sn),
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
    ('\nnn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
    ('\npartial_n', partial_n),
    ('pu_then_pn', pu_then_pn),
    ('\npu', pu),
    ('unbiased_pn', unbiased_pn),
    ('iwpn', iwpn),
    ('pnu', pnu),
    ('\nsets_save_name', sets_save_name),
    ('sets_load_name', sets_load_name),
    ('dre_save_name', dre_save_name),
    ('dre_load_name', dre_load_name),
])


for key, value in params.items():
    print('{}: {}'.format(key, value))
print('', flush=True)


parser = argparse.ArgumentParser(description='CIFAR10')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

settings.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
settings.test_batch_size = 500
settings.validation_interval = 10


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


def pick_sn_data(data, labels, n):
    neg_nums = np.random.multinomial(n, neg_ps)
    print('numbers in each negative subclass', neg_nums)
    selected_sn = []
    for i in range(6):
        idxs = np.argwhere(labels == i+2).reshape(-1)
        selected = np.random.choice(idxs, neg_nums[i], replace=False)
        selected_sn.extend(selected)
    return data[np.array(selected_sn)]


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


if sets_load_name is None:
    p_set = torch.utils.data.TensorDataset(
        pick_p_data(train_data, train_labels, p_num))

    sn_set = torch.utils.data.TensorDataset(
        pick_sn_data(train_data, train_labels, sn_num))

    n_set = torch.utils.data.TensorDataset(
        pick_n_data(train_data, train_labels, n_num))

    u_set = torch.utils.data.TensorDataset(pick_u_data(train_data, u_num))

    p_validation = pick_p_data(valid_data, valid_labels, pv_num)
    sn_validation = pick_sn_data(valid_data, valid_labels, snv_num)
    n_validation = pick_n_data(valid_data, valid_labels, nv_num)
    u_validation = pick_u_data(valid_data, uv_num)

    if sets_save_name is not None:
        pickle.dump(
            (p_set, sn_set, n_set, u_set,
             p_validation, sn_validation, n_validation, u_validation),
            open(sets_save_name, 'wb'))

if sets_load_name is not None:
    p_set, sn_set, n_set, u_set,\
        p_validation, sn_validation, n_validation, u_validation =\
        pickle.load(open(sets_load_name, 'rb'))


test_data = torch.zeros(cifar10_test.test_data.shape)
test_data = test_data.permute(0, 3, 1, 2)
test_labels = torch.tensor(cifar10_test.test_labels)

for i, (image, _) in enumerate(cifar10_test):
    test_data[i] = image

test_posteriors = torch.zeros(test_labels.size())
test_posteriors[test_labels < 2] = 1
test_posteriors[test_labels == 2] = neg_ps[0] * rho * 10
test_posteriors[test_labels == 3] = neg_ps[1] * rho * 10
test_posteriors[test_labels == 4] = neg_ps[2] * rho * 10
test_posteriors[test_labels == 5] = neg_ps[3] * rho * 10
test_posteriors[test_labels == 6] = neg_ps[4] * rho * 10
test_posteriors[test_labels == 7] = neg_ps[5] * rho * 10
test_posteriors[test_labels > 7] = 1

test_set_dre = torch.utils.data.TensorDataset(
    test_data, test_posteriors.unsqueeze(1))

t_labels = torch.zeros(test_labels.size())
t_labels[test_posteriors > 1/2] = 1
t_labels[test_posteriors <= 1/2] = -1

test_set_pre_cls = torch.utils.data.TensorDataset(
    test_data, t_labels.unsqueeze(1))

test_labels[test_labels < 2] = 1
test_labels[test_labels > 7] = 1
test_labels[test_labels != 1] = -1

test_set_cls = torch.utils.data.TensorDataset(
    test_data, test_labels.unsqueeze(1))


if pu:
    print('')
    model = PreActResNet18().cuda() if args.cuda else PreActResNet18()
    cls = training.PUClassifier(
            model, pi=pi, lr=learning_rate_cls, weight_decay=weight_decay,
            nn=True, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, u_set, test_set_cls, p_batch_size, u_batch_size,
              p_validation, u_validation, cls_training_epochs)

if unbiased_pn:
    print('')
    model = PreActResNet18().cuda() if args.cuda else PreActResNet18()
    cls = training.PNClassifier(
            model, pi=pi, lr=learning_rate_cls, weight_decay=weight_decay)
    cls.train(p_set, n_set, test_set_cls, p_batch_size, n_batch_size,
              p_validation, n_validation, cls_training_epochs)

if pnu:
    print('')
    model = PreActResNet18().cuda() if args.cuda else PreActResNet18()
    cls = training.PNUClassifier(
            model, pi=pi,
            lr=learning_rate_cls, weight_decay=weight_decay,
            pn_fraction=non_pu_fraction,
            nn=True, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, n_set, u_set, test_set_cls,
              p_batch_size, n_batch_size, u_batch_size,
              p_validation, n_validation, u_validation,
              cls_training_epochs)

if partial_n or iwpn:
    if dre_load_name is None:
        print('')
        model = (PreActResNet18(True).cuda()
                 if args.cuda
                 else PreActResNet18(True))
        dre = training.PosteriorProbability(
                model, pi=pi, rho=rho,
                lr=learning_rate_dre, weight_decay=weight_decay)
        dre.train(p_set, sn_set, u_set, test_set_dre,
                  p_batch_size, sn_batch_size, u_batch_size,
                  p_validation, sn_validation, u_validation,
                  dre_training_epochs)
        if dre_save_name is not None:
            pickle.dump(dre.model, open(dre_save_name, 'wb'))

    if dre_load_name is not None:
        dre_model = pickle.load(open(dre_load_name, 'rb'))
    else:
        dre_model = dre.model

if partial_n:
    print('')
    model = PreActResNet18().cuda() if args.cuda else PreActResNet18()
    cls = training.WeightedClassifier(
            model, dre_model, pi=pi, rho=rho,
            sep_value=sep_value, adjust_p=adjust_p, adjust_sn=adjust_sn,
            lr=learning_rate_cls, weight_decay=weight_decay)
    cls.train(p_set, sn_set, u_set, test_set_cls,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs)

if iwpn:
    print('')
    model = PreActResNet18().cuda() if args.cuda else PreActResNet18()
    cls = training.PNClassifier(
            model, pi=pi/(pi+rho), pp_model=dre_model,
            adjust_p=adjust_p, adjust_n=adjust_sn,
            lr=learning_rate_cls, weight_decay=weight_decay)
    cls.train(p_set, sn_set, test_set_cls, p_batch_size, sn_batch_size,
              p_validation, sn_validation, cls_training_epochs)

if pu_then_pn:
    print('')
    model = PreActResNet18().cuda() if args.cuda else PreActResNet18()
    cls = training.PUClassifier3(
            model, pi=pi, rho=rho,
            lr=learning_rate_cls, weight_decay=weight_decay,
            nn=True, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, sn_set, u_set, test_set_pre_cls,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs)

    print('')
    model = PreActResNet18().cuda() if args.cuda else PreActResNet18()
    cls2 = training.PNClassifier(
            model, pi=pi/(pi+rho), pu_model=cls.model,
            lr=learning_rate_cls, weight_decay=weight_decay)
    cls2.train(p_set, sn_set, test_set_cls, p_batch_size, sn_batch_size,
               p_validation, sn_validation, cls_training_epochs)
