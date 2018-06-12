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


normalize = True

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
# pi = 0.097
rho = 0.15

positive_classes = [0, 2, 4, 6, 8]

neg_ps = [0, 0.6, 0, 0.15, 0, 0.03, 0, 0.2, 0, 0.02]
# neg_ps = [0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2]
# neg_ps = [0, 1/3, 0, 1/3, 0, 1/3, 0, 0, 0, 0]
# neg_ps = [1/6, 1/6, 1/6, 1/6, 0, 1/6, 1/6, 0, 0, 0]

non_pu_fraction = 0.7

sep_value = 0.3
adjust_p = False
adjust_sn = True

dre_training_epochs = 100
cls_training_epochs = 100
pn_epochs = 40
convex_epochs = 50
pre_convex_epochs = 10

p_batch_size = 100
n_batch_size = 100
sn_batch_size = 100
u_batch_size = 1000

learning_rate_dre = 1e-3
learning_rate_pn = 1e-3
learning_rate_cls = 1e-3
weight_decay = 1e-4
validation_momentum = 0.5
beta = 0

non_negative = True
nn_threshold = 0
nn_rate = 1/3

ls_prob_est = False
pu_prob_est = True

use_true_prob = False

partial_n = False
hard_label = True
sampling = False
p_some_u_plus_n = True

pu_plus_n = False
minus_n = False

iwpn = False
three_class = False
prob_pred = False
pu_then_pn = False

pu = False
unbiased_pn = False
pnu = False

sets_save_name = None
# sets_load_name = 'pickle/mnist/1000_1000_10000/imbN/sets_imbN_a.p'
sets_load_name = None

dre_save_name = None
# dre_load_name = ('pickle/mnist/1000_1000_10000/imbN/'
#                  + 'ls_prob_est_rho015_imbN_a.p')
dre_load_name = None


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
    ('positive_classes', positive_classes),
    ('neg_ps', neg_ps),
    ('non_pu_fraction', non_pu_fraction),
    ('\nsep_value', sep_value),
    ('adjust_p', adjust_p),
    ('adjust_sn', adjust_sn),
    ('\ndre_training_epochs', dre_training_epochs),
    ('cls_training_epochs', cls_training_epochs),
    ('pn_epochs', pn_epochs),
    ('convex_epochs', convex_epochs),
    ('pre_convex_epochs', pre_convex_epochs),
    ('\np_batch_size', p_batch_size),
    ('sn_batch_size', sn_batch_size),
    ('n_batch_size', n_batch_size),
    ('u_batch_size', u_batch_size),
    ('\nlearning_rate_dre', learning_rate_dre),
    ('\nlearning_rate_pn', learning_rate_pn),
    ('learning_rate_cls', learning_rate_cls),
    ('weight_decay', weight_decay),
    ('validation_momentum', validation_momentum),
    ('beta', beta),
    ('\nnon_negative', non_negative),
    ('nn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
    ('\nls_prob_est', ls_prob_est),
    ('pu_prob_est', pu_prob_est),
    ('use_true_prob', use_true_prob),
    ('\npartial_n', partial_n),
    ('hard_label', hard_label),
    ('sampling', sampling),
    ('p_some_u_plus_n', p_some_u_plus_n),
    ('\npu_plus_n', pu_plus_n),
    ('minus_n', minus_n),
    ('\niwpn', iwpn),
    ('three_class', three_class),
    ('prob_pred', prob_pred),
    ('pu_then_pn', pu_then_pn),
    ('\npu', pu),
    ('unbiased_pn', unbiased_pn),
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


def posteriors(labels):
    posteriors = torch.zeros(labels.size())
    for i in range(10):
        if i in positive_classes:
            posteriors[labels == i] = 1
        else:
            posteriors[labels == i] = neg_ps[i] * rho * 10
    return posteriors.unsqueeze(1)


def pick_p_data(data, labels, n):
    p_idxs = np.zeros_like(labels)
    for i in range(10):
        if i in positive_classes:
            p_idxs[(labels == i).numpy().astype(bool)] = 1
    p_idxs = np.argwhere(p_idxs == 1).reshape(-1)
    selected_p = np.random.choice(p_idxs, n, replace=False)
    return data[selected_p], posteriors(labels[selected_p])


def pick_sn_data(data, labels, n):
    neg_nums = np.random.multinomial(n, neg_ps)
    print('numbers in each subclass', neg_nums)
    selected_sn = []
    for i in range(10):
        idxs = np.argwhere(labels == i).reshape(-1)
        selected = np.random.choice(idxs, neg_nums[i], replace=False)
        selected_sn.extend(selected)
    selected_sn = np.array(selected_sn)
    return data[selected_sn], posteriors(labels[selected_sn])


def pick_n_data(data, labels, n):
    n_idxs = np.zeros_like(labels)
    for i in range(10):
        if i not in positive_classes:
            n_idxs[(labels == i).numpy().astype(bool)] = 1
    n_idxs = np.argwhere(n_idxs == 1).reshape(-1)
    selected_n = np.random.choice(n_idxs, n, replace=False)
    return data[selected_n], posteriors(labels[selected_n])


def pick_u_data(data, labels, n):
    selected_u = np.random.choice(len(data), n, replace=False)
    return data[selected_u], posteriors(labels[selected_u])


if normalize:
    train_data = torch.zeros(mnist.train_data.size())
    for i, (image, _) in enumerate(mnist):
        train_data[i] = image
else:
    train_data = mnist.train_data
train_data = train_data.unsqueeze(1)
train_labels = mnist.train_labels

if normalize:
    test_data = torch.zeros(mnist_test.test_data.size())
    for i, (image, _) in enumerate(mnist_test):
        test_data[i] = image
else:
    test_data = mnist_test.test_data
test_data = test_data.unsqueeze(1)
test_labels = mnist_test.test_labels


# for i in range(10):
#     print(torch.sum(train_labels == i))

idxs = np.random.permutation(len(train_data))

probs = pickle.load(open('prob_ac_pos.p', 'rb'))
probs2 = -np.ones(len(train_data))
probs2[np.array(train_labels).reshape(-1) % 2 == 1] = probs
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
        *pick_p_data(train_data, train_labels, p_num))

    # sn_set = torch.utils.data.TensorDataset(n_train_data[sn_train_idxs])
    sn_set = torch.utils.data.TensorDataset(
        *pick_sn_data(train_data, train_labels, sn_num))

    n_set = torch.utils.data.TensorDataset(
        *pick_n_data(train_data, train_labels, n_num))

    u_set = torch.utils.data.TensorDataset(
        *pick_u_data(train_data, train_labels, u_num))

    p_validation = pick_p_data(valid_data, valid_labels, pv_num)
    # sn_validation = n_valid_data[sn_valid_idxs],
    sn_validation = pick_sn_data(valid_data, valid_labels, snv_num)
    n_validation = pick_n_data(valid_data, valid_labels, nv_num)
    u_validation = pick_u_data(valid_data, valid_labels, uv_num)

    if sets_save_name is not None:
        pickle.dump(
            (p_set, sn_set, n_set, u_set,
             p_validation, sn_validation, n_validation, u_validation),
            open(sets_save_name, 'wb'))

if sets_load_name is not None:
    p_set, sn_set, n_set, u_set,\
        p_validation, sn_validation, n_validation, u_validation =\
        pickle.load(open(sets_load_name, 'rb'))


t_labels = torch.zeros(test_labels.size())

for i in range(10):
    if i in positive_classes:
        t_labels[test_labels == i] = 1
    else:
        t_labels[test_labels == i] = -1

test_posteriors = posteriors(test_labels)

test_set = torch.utils.data.TensorDataset(
    test_data, t_labels.unsqueeze(1), test_posteriors)


t_labels = torch.zeros(test_labels.size())
t_labels[test_posteriors.view(-1) > 1/2] = 1
t_labels[test_posteriors.view(-1) <= 1/2] = -1

test_set_pre_cls = torch.utils.data.TensorDataset(
    test_data, t_labels.unsqueeze(1))


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


if ls_prob_est and dre_load_name is None:
    print('')
    model = Net().cuda() if args.cuda else Net()
    dre = training.PosteriorProbability(
            model, pi=pi, rho=rho, beta=beta,
            lr=learning_rate_dre, weight_decay=weight_decay)
    dre.train(p_set, sn_set, u_set, test_set,
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
    dre.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)
    if dre_save_name is not None:
        pickle.dump(dre.model, open(dre_save_name, 'wb'))
    dre_model = dre.model

if dre_load_name is not None:
    dre_model = pickle.load(open(dre_load_name, 'rb'))
    # dre_model.eval()
    # px = dre_model(p_validation.type(settings.dtype))
    # snx = dre_model(sn_validation.type(settings.dtype))
    # ux = dre_model(u_validation.type(settings.dtype))
    # nl = nn.LogSigmoid()
    # print((torch.mean(-nl(ux))
    #       - rho * torch.mean(-nl(-snx))
    #       - pi * torch.mean(-nl(-px))).item())
    # plt.figure()
    # plt.title('p')
    # # plt.hist(np.log(1+np.exp(px.detach().cpu().numpy())))
    # plt.hist(px.detach().cpu().numpy())
    # plt.figure()
    # plt.title('sn')
    # # plt.hist(np.log(1+np.exp(snx.detach().cpu().numpy())))
    # plt.hist(snx.detach().cpu().numpy())
    # plt.figure()
    # plt.title('u')
    # # plt.hist(np.log(1+np.exp(ux.detach().cpu().numpy())))
    # plt.hist(ux.detach().cpu().numpy())
    # plt.show()

if use_true_prob:
    dre_model = None

if partial_n:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.WeightedClassifier(
            model, dre_model, pi=pi, rho=rho,
            sep_value=sep_value,
            adjust_p=adjust_p, adjust_sn=adjust_sn, hard_label=hard_label,
            lr=learning_rate_cls, weight_decay=weight_decay)
    cls.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

if sampling:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.SamplingClassifier(
            model, dre_model, pi=pi, rho=rho,
            lr=learning_rate_cls, weight_decay=weight_decay)
    cls.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

if p_some_u_plus_n:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PSomeUPlusN(
            model, pi=pi, rho=rho,
            lr=learning_rate_cls, weight_decay=weight_decay,
            pn_lr=learning_rate_pn,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, pn_epochs,
              convex_epochs=convex_epochs)

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
    cls.train(p_set, sn_set, test_set, p_batch_size, sn_batch_size,
              p_validation, sn_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

if pu_plus_n:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PUClassifierPlusN(
            model, pi=pi, rho=rho,
            lr=learning_rate_cls, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate,
            minus_n=minus_n)
    cls.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)


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
              cls_training_epochs, convex_epochs=pre_convex_epochs)

    print('')
    model = Net().cuda() if args.cuda else Net()
    cls2 = training.PNClassifier(
            model, pi=pi/(pi+rho), pu_model=cls.model,
            lr=learning_rate_cls, weight_decay=weight_decay)
    cls2.train(p_set, sn_set, test_set, p_batch_size, sn_batch_size,
               p_validation, sn_validation,
               cls_training_epochs, convex_epochs=convex_epochs)

if three_class:
    print('')
    model = Net(num_classes=3).cuda() if args.cuda else Net(num_classes=3)
    cls = training.ThreeClassifier(
            model, pi=pi, rho=rho, prob_pred=prob_pred,
            lr=learning_rate_cls, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation, cls_training_epochs)

if pu:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PUClassifier(
            model, pi=pi, lr=learning_rate_cls, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, u_set, test_set, p_batch_size, u_batch_size,
              p_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

if unbiased_pn:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNClassifier(
            model, pi=pi, lr=learning_rate_cls, weight_decay=weight_decay)
    cls.train(p_set, n_set, test_set, p_batch_size, n_batch_size,
              p_validation, n_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

if pnu:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNUClassifier(
            model, pi=pi,
            lr=learning_rate_cls, weight_decay=weight_decay,
            pn_fraction=non_pu_fraction,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, n_set, u_set, test_set,
              p_batch_size, n_batch_size, u_batch_size,
              p_validation, n_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)