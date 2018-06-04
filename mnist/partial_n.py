import argparse
import numpy as np
import pickle
# import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import training
import settings


normalize = False

p_num = 1000
n_num = 1000
sn_num = 1000
u_num = 10000

pv_num = 100
nv_num = 100
snv_num = 100
uv_num = 1000

u_cut = 40000

pi = 0.49
rho = 0.15

neg_ps = [0.6, 0.15, 0.03, 0.2, 0.02]
# neg_ps = [0.2, 0.2, 0.2, 0.2, 0.2]
# neg_ps = [1/3, 1/3, 1/3, 0, 0]

non_pu_fraction = 0.8

sep_value = 0.3
adjust_p = False
adjust_sn = True

dre_training_epochs = 100
cls_training_epochs = 100
convex_epochs = 100

p_batch_size = 100
n_batch_size = 100
sn_batch_size = 100
u_batch_size = 1000

learning_rate_dre = 1e-4
learning_rate_cls = 1e-3
weight_decay = 1e-4
validation_momentum = 0.5

non_negative = True
nn_threshold = 0
nn_rate = 1/3

partial_n = False
pu_partial_n = False
partial_n_kai = False
pu_partial_n_kai = False

ls_prob_est = True
pu_prob_est = False

pu_plus_n = False
minus_n = False
pu_then_pn = False

pu = False
unbiased_pn = False
iwpn = False
pnu = False

# sets_load_name = None
sets_load_name = 'pickle/mnist/1000_1000_10000/imbN/sets_imbN.p'
sets_save_name = None

dre_load_name = None
# dre_load_name = 'pickle/1000_1000_10000/dre_model_rho015_farN.p'
dre_save_name = None


params = OrderedDict([
    ('normalize', normalize),
    ('\np_num', p_num),
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
    ('convex_epochs', convex_epochs),
    ('\np_batch_size', p_batch_size),
    ('sn_batch_size', sn_batch_size),
    ('n_batch_size', n_batch_size),
    ('u_batch_size', u_batch_size),
    ('\nlearning_rate_dre', learning_rate_dre),
    ('learning_rate_cls', learning_rate_cls),
    ('weight_decay', weight_decay),
    ('validation_momentum', validation_momentum),
    ('\nnon_negative', non_negative),
    ('nn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
    ('\npartial_n', partial_n),
    ('pu_partial_n', pu_partial_n),
    ('partial_n_kai', partial_n_kai),
    ('pu_partial_n_kai', pu_partial_n_kai),
    ('\nls_prob_est', ls_prob_est),
    ('pu_prob_est', pu_prob_est),
    ('\npu_plus_n', pu_plus_n),
    ('minus_n', minus_n),
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


parser = argparse.ArgumentParser(description='MNIST partial n')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

settings.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor


# torchvision.datasets.MNIST outputs a set of PIL images
# We transform them to tensors
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

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
    neg_nums = np.random.multinomial(n, neg_ps)
    print('numbers in each negative subclass', neg_nums)
    selected_sn = []
    for i in range(5):
        idxs = np.argwhere(labels == 2*i+1).reshape(-1)
        selected = np.random.choice(idxs, neg_nums[i], replace=False)
        selected_sn.extend(selected)
    return data[np.array(selected_sn)]


def pick_n_data(data, labels, n2):
    # sn_idxs = np.argwhere(
    #     np.logical_and(labels % 2 == 1, labels < 6)).reshape(-1)
    n_idxs = np.argwhere(labels % 2 == 1).reshape(-1)
    # selected_sn = np.random.choice(sn_idxs, n1, replace=False)
    selected_n = np.random.choice(n_idxs, n2, replace=False)
    # selected_n = np.r_[selected_n, selected_sn]
    return data[np.random.permutation(selected_n)]


def pick_u_data(data, n):
    selected_u = np.random.choice(len(data), n, replace=False)
    return data[selected_u]


if normalize:
    train_data = torch.zeros(mnist.train_data.size())
    for i, (image, _) in enumerate(mnist):
        train_data[i] = image
else:
    train_data = mnist.train_data
train_labels = mnist.train_labels

# for i in range(10):
#     print(torch.sum(train_labels == i))

idxs = np.random.permutation(len(train_data))

probs = pickle.load(open('prob_ac_pos.p', 'rb'))
probs2 = -np.ones(len(train_data))
probs2[np.array(train_labels) % 2 == 1] = probs
probs2 = probs2[idxs]
probs = probs2[probs2 >= 0]

valid_data = train_data[idxs][u_cut:]
valid_labels = train_labels[idxs][u_cut:]
train_data = train_data[idxs][:u_cut]
train_labels = train_labels[idxs][:u_cut]


n_train_data = train_data[train_labels % 2 == 1]
n_labels = train_labels[train_labels % 2 == 1]
num_n_train = len(n_train_data)
n_valid_data = valid_data[valid_labels % 2 == 1]

# probs_train = (np.maximum(probs[:num_n_train]-27000, 0))**5
probs_train = (len(probs)-probs[:num_n_train])**10
# probs_train = probs[:num_n_train]**10
probs_train = probs_train/np.sum(probs_train)
# probs_valid = (np.maximum(probs[num_n_train:]-27000, 0))**5
probs_valid = (len(probs)-probs[num_n_train:])**10
# probs_valid = probs[num_n_train:]**10
probs_valid = probs_valid/np.sum(probs_valid)

sn_train_idxs = np.random.choice(
    num_n_train, sn_num, replace=False, p=probs_train)
sn_valid_idxs = np.random.choice(
    len(n_valid_data), snv_num, replace=False, p=probs_valid)

# plt.hist(probs[sn_train_idxs])
# plt.show()


if sets_load_name is None:
    p_set = torch.utils.data.TensorDataset(
        pick_p_data(train_data, train_labels, p_num).unsqueeze(1))

    sn_set = torch.utils.data.TensorDataset(
        pick_sn_data(train_data, train_labels, sn_num).unsqueeze(1))

    # sn_set = torch.utils.data.TensorDataset(
    #     n_train_data[sn_train_idxs].unsqueeze(1))

    n_set = torch.utils.data.TensorDataset(
        pick_n_data(train_data, train_labels, n_num).unsqueeze(1))

    # selected_u = np.random.choice(len(train_data), u_num, replace=False)
    # u_set = torch.utils.data.TensorDataset(
    #     train_data[selected_u].unsqueeze(1))
    # u_set_labels = torch.utils.data.TensorDataset(
    #     train_data[selected_u].unsqueeze(1),
    #     train_labels[selected_u].unsqueeze(1))

    u_set = torch.utils.data.TensorDataset(
        pick_u_data(train_data, u_num).unsqueeze(1))

    if sets_save_name is not None:
        pickle.dump((p_set, sn_set, u_set), open(sets_save_name, 'wb'))

if sets_load_name is not None:
    p_set, sn_set, u_set = pickle.load(open(sets_load_name, 'rb'))


p_validation = pick_p_data(valid_data, valid_labels, pv_num).unsqueeze(1)
sn_validation = pick_sn_data(valid_data, valid_labels, snv_num).unsqueeze(1)
# sn_validation = n_valid_data[sn_valid_idxs].unsqueeze(1)
n_validation = pick_n_data(valid_data, valid_labels, nv_num).unsqueeze(1)
u_validation = pick_u_data(valid_data, uv_num).unsqueeze(1)


if normalize:
    test_data = torch.zeros(mnist_test.test_data.size())
    for i, (image, _) in enumerate(mnist_test):
        test_data[i] = image
else:
    test_data = mnist_test.test_data
test_labels = mnist_test.test_labels


test_posteriors = torch.zeros(test_labels.size())
test_posteriors[test_labels % 2 == 0] = 1
# test_posteriors[test_labels % 2 == 1] = -1
test_posteriors[test_labels == 1] = neg_ps[0] * rho * 10
test_posteriors[test_labels == 3] = neg_ps[1] * rho * 10
test_posteriors[test_labels == 5] = neg_ps[2] * rho * 10
test_posteriors[test_labels == 7] = neg_ps[3] * rho * 10
test_posteriors[test_labels == 9] = neg_ps[4] * rho * 10

test_set_dre = torch.utils.data.TensorDataset(
    test_data.unsqueeze(1), test_posteriors.unsqueeze(1))

t_labels = torch.zeros(test_labels.size())
t_labels[test_labels % 2 == 0] = 1
t_labels[test_labels % 2 == 1] = -1

test_set_cls = torch.utils.data.TensorDataset(
    test_data.unsqueeze(1), t_labels.unsqueeze(1))


t_labels = torch.zeros(test_labels.size())
t_labels[test_posteriors > 1/2] = 1
t_labels[test_posteriors <= 1/2] = -1

test_set_pre_cls = torch.utils.data.TensorDataset(
    test_data.unsqueeze(1), t_labels.unsqueeze(1))


class Net(nn.Module):

    def __init__(self, sigmoid_output=False):
        super(Net, self).__init__()
        self.sigmoid_output = sigmoid_output
        self.conv1 = nn.Conv2d(1, 5, 5, 1)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 10, 5, 1)
        self.bn2 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(4*4*10, 40)
        self.fc2 = nn.Linear(40, 1)

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
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


if ls_prob_est and dre_load_name is None:
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
    dre_model = dre.model


if pu_prob_est and dre_load_name is None:
    print('')
    model = Net().cuda() if args.cuda else Net()
    dre = training.PUClassifier3(
            model, pi=pi, rho=rho,
            lr=learning_rate_cls, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate,
            prob_est=True)
    dre.train(p_set, sn_set, u_set, test_set_dre,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

if dre_load_name is not None:
    dre_model = pickle.load(open(dre_load_name, 'rb'))

if partial_n:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.WeightedClassifier(
            model, dre_model, pi=pi, rho=rho,
            sep_value=sep_value, adjust_p=adjust_p, adjust_sn=adjust_sn,
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
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, sn_set, u_set, test_set_cls,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs)

if iwpn:
    print('')
    model = Net().cuda() if args.cuda else Net()
    if adjust_sn:
        pi_ = pi/(pi+rho)
    else:
        pi_ = pi
    cls = training.PNClassifier(
            model, pi=pi_, pp_model=dre_model,
            adjust_p=adjust_p, adjust_n=adjust_sn,
            lr=learning_rate_cls, weight_decay=weight_decay)
    cls.train(p_set, sn_set, test_set_cls, p_batch_size, sn_batch_size,
              p_validation, sn_validation, cls_training_epochs)

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
                nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
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
                nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
        cls.train(p_set, sn_set, u_set, test_set_cls,
                  p_batch_size, sn_batch_size, u_batch_size,
                  p_validation, sn_validation, u_validation,
                  cls_training_epochs)

if pu_plus_n:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PUClassifierPlusN(
            model, pi=pi, rho=rho,
            lr=learning_rate_cls, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate,
            minus_n=minus_n)
    cls.train(p_set, sn_set, u_set, test_set_cls,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs)


if pu_then_pn:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PUClassifier3(
            model, pi=pi, rho=rho,
            lr=learning_rate_cls, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, sn_set, u_set, test_set_pre_cls,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs)

    print('')
    model = Net().cuda() if args.cuda else Net()
    cls2 = training.PNClassifier(
            model, pi=pi/(pi+rho), pu_model=cls.model,
            lr=learning_rate_cls, weight_decay=weight_decay)
    cls2.train(p_set, sn_set, test_set_cls, p_batch_size, sn_batch_size,
               p_validation, sn_validation, cls_training_epochs)


if pu:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PUClassifier(
            model, pi=pi, lr=learning_rate_cls, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, u_set, test_set_cls, p_batch_size, u_batch_size,
              p_validation, u_validation, cls_training_epochs)

if unbiased_pn:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNClassifier(
            model, pi=pi, lr=learning_rate_cls, weight_decay=weight_decay)
    cls.train(p_set, n_set, test_set_cls, p_batch_size, n_batch_size,
              p_validation, n_validation, cls_training_epochs)

if pnu:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNUClassifier(
            model, pi=pi,
            lr=learning_rate_cls, weight_decay=weight_decay,
            pn_fraction=non_pu_fraction,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, n_set, u_set, test_set_cls,
              p_batch_size, n_batch_size, u_batch_size,
              p_validation, n_validation, u_validation, cls_training_epochs)
