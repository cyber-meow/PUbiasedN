import argparse
import numpy as np
import pickle

import torch
import torch.utils.data
import torch.nn.functional as F

import training
import settings

# from cifar10.pu_biased_n import params, Net
# from cifar10.pu_biased_n import train_data, test_data
# from cifar10.pu_biased_n import train_labels, test_labels

# from mnist.pu_biased_n import params, Net
# from mnist.pu_biased_n import train_data, test_data
# from mnist.pu_biased_n import train_labels, test_labels

from uci.pu_biased_n import params, Net
from uci.pu_biased_n import train_data, test_data
from uci.pu_biased_n import train_labels, test_labels


num_classes = params['num_classes']

p_num = params['\np_num']
n_num = params.get('n_num', 0)
sn_num = params['sn_num']
u_num = params['u_num']

pv_num = params['\npv_num']
nv_num = params.get('nv_num', 0)
snv_num = params['snv_num']
uv_num = params['uv_num']

u_cut = params['\nu_cut']

pi = params['\npi']
rho = params['rho']
true_rho = params.get('true_rho', rho)

positive_classes = params['\npositive_classes']
neg_ps = params['neg_ps']

non_pu_fraction = params['\nnon_pu_fraction']
balanced = params['balanced']

u_per = params['\nu_per']
adjust_p = params['adjust_p']
adjust_sn = params['adjust_sn']

cls_training_epochs = params['\ncls_training_epochs']
convex_epochs = params['convex_epochs']

p_batch_size = params['\np_batch_size']
n_batch_size = params.get('n_batch_size', 0)
sn_batch_size = params['sn_batch_size']
u_batch_size = params['u_batch_size']

learning_rate_cls = params['\nlearning_rate_cls']
weight_decay = params['weight_decay']
validation_momentum = params['validation_momentum']

if 'learning_rate_ppe' in params:
    learning_rate_ppe = params['learning_rate_ppe']
else:
    learning_rate_ppe = learning_rate_cls

start_validation_epoch = params.get('\nstart_validation_epoch', 0)
milestones = params.get('milestones', [1000])
lr_d = params.get('lr_d', 1)

non_negative = params['\nnon_negative']
nn_threshold = params['nn_threshold']
nn_rate = params['nn_rate']

pu_prob_est = params['\npu_prob_est']
use_true_post = params['use_true_post']

partial_n = params['\npartial_n']
hard_label = params['hard_label']

pn_then_pu = params.get('pn_then_pu', False)
pu_then_pn = params.get('pu_then_pn', False)

iwpn = params['\niwpn']
pu = params['pu']
pnu = params['pnu']
unbiased_pn = params.get('unbiased_pn', False)

random_seed = params['\nrandom_seed']

ppe_save_name = params['ppe_save_name']
ppe_load_name = params['ppe_load_name']

priors = params.get('\npriors', None)
if priors is None:
    priors = [1/num_classes for _ in range(num_classes)]


parser = argparse.ArgumentParser(description='Main File')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--rho', type=float, default=0.2)
parser.add_argument('--u_per', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--adjust_p', default=True)
parser.add_argument('--algo', type=int, default=0)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.random_seed is not None:
    params['\nrandom_seed'] = args.random_seed
    random_seed = args.random_seed


settings.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor


for key, value in params.items():
    print('{}: {}'.format(key, value))
print('\nvalidation_interval', settings.validation_interval)
print('', flush=True)


def posteriors(labels):
    posteriors = torch.zeros(labels.size())
    for i in range(num_classes):
        if i in positive_classes:
            posteriors[labels == i] = 1
        else:
            posteriors[labels == i] = neg_ps[i] * rho * 1/priors[i]
    return posteriors.unsqueeze(1)


def pick_p_data(data, labels, n):
    p_idxs = np.zeros_like(labels)
    for i in range(num_classes):
        if i in positive_classes:
            p_idxs[(labels == i).numpy().astype(bool)] = 1
    p_idxs = np.argwhere(p_idxs == 1).reshape(-1)
    selected_p = np.random.choice(p_idxs, n, replace=False)
    return data[selected_p], posteriors(labels[selected_p])


def pick_n_data(data, labels, n):
    n_idxs = np.zeros_like(labels)
    for i in range(num_classes):
        if i not in positive_classes:
            n_idxs[(labels == i).numpy().astype(bool)] = 1
    n_idxs = np.argwhere(n_idxs == 1).reshape(-1)
    selected_n = np.random.choice(n_idxs, n, replace=False)
    return data[selected_n], labels[selected_n]


def pick_sn_data(data, labels, n):
    neg_nums = np.random.multinomial(n, neg_ps)
    print('numbers in each subclass', neg_nums)
    selected_sn = []
    for i in range(num_classes):
        if neg_nums[i] != 0:
            idxs = np.argwhere(labels == i).reshape(-1)
            selected = np.random.choice(idxs, neg_nums[i], replace=False)
            selected_sn.extend(selected)
    selected_sn = np.array(selected_sn)
    return data[selected_sn], posteriors(labels[selected_sn])


def pick_u_data(data, labels, n):
    selected_u = np.random.choice(len(data), n, replace=False)
    return data[selected_u], posteriors(labels[selected_u])


np.random.seed(random_seed)
idxs = np.random.permutation(len(train_data))

valid_data = train_data[idxs][u_cut:]
valid_labels = train_labels[idxs][u_cut:]
train_data = train_data[idxs][:u_cut]
train_labels = train_labels[idxs][:u_cut]


u_set = torch.utils.data.TensorDataset(
    *pick_u_data(train_data, train_labels, u_num))
u_validation = pick_u_data(valid_data, valid_labels, uv_num)

p_set = torch.utils.data.TensorDataset(
    *pick_p_data(train_data, train_labels, p_num))
p_validation = pick_p_data(valid_data, valid_labels, pv_num)

sn_set = torch.utils.data.TensorDataset(
    *pick_sn_data(train_data, train_labels, sn_num))
sn_validation = pick_sn_data(valid_data, valid_labels, snv_num)

n_set = torch.utils.data.TensorDataset(
    *pick_n_data(train_data, train_labels, n_num))
n_validation = pick_n_data(valid_data, valid_labels, nv_num)


t_labels = torch.zeros(test_labels.size())

for i in range(num_classes):
    if i in positive_classes:
        t_labels[test_labels == i] = 1
    else:
        t_labels[test_labels == i] = -1

test_posteriors = posteriors(test_labels)

test_set = torch.utils.data.TensorDataset(
    test_data, t_labels.unsqueeze(1).float(), test_posteriors)


torch.manual_seed(random_seed)


if pu_prob_est and ppe_load_name is None:
    print('')
    model = Net().cuda() if args.cuda else Net()
    ppe = training.PUClassifier3(
            model, pi=pi, rho=rho,
            lr=learning_rate_ppe, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate,
            milestones=milestones, lr_d=lr_d,
            prob_est=True, validation_momentum=validation_momentum,
            start_validation_epoch=start_validation_epoch)
    ppe.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)
    if ppe_save_name is not None:
        pickle.dump(ppe.model, open(ppe_save_name, 'wb'))
    ppe_model = ppe.model

if ppe_load_name is not None:
    ppe_model = pickle.load(open(ppe_load_name, 'rb'))


if (partial_n or (iwpn and (adjust_p or adjust_sn))) and not use_true_post:
    p_set = torch.utils.data.TensorDataset(
        p_set.tensors[0], F.sigmoid(
            training.Training()
            .feed_in_batches(ppe_model, p_set.tensors[0])).cpu())
    sn_set = torch.utils.data.TensorDataset(
        sn_set.tensors[0], F.sigmoid(
            training.Training()
            .feed_in_batches(ppe_model, sn_set.tensors[0])).cpu())
    u_set = torch.utils.data.TensorDataset(
        u_set.tensors[0], F.sigmoid(
            training.Training()
            .feed_in_batches(ppe_model, u_set.tensors[0])).cpu())
    p_validation = p_validation[0], F.sigmoid(
        training.Training()
        .feed_in_batches(ppe_model, p_validation[0])).cpu()
    sn_validation = sn_validation[0], F.sigmoid(
        training.Training()
        .feed_in_batches(ppe_model, sn_validation[0])).cpu()
    u_validation = u_validation[0], F.sigmoid(
        training.Training()
        .feed_in_batches(ppe_model, u_validation[0])).cpu()

sep_value = np.percentile(
    u_set.tensors[1].numpy().reshape(-1), int((1-pi-true_rho)*u_per*100))
print('\nsep_value =', sep_value)


if partial_n:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.WeightedClassifier(
            model, balanced=balanced, pi=pi, rho=rho,
            sep_value=sep_value,
            adjust_p=adjust_p, adjust_sn=adjust_sn, hard_label=hard_label,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d,
            validation_momentum=validation_momentum,
            start_validation_epoch=start_validation_epoch)
    cls.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

if iwpn:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNClassifier(
            model, pi=pi/(pi+rho),
            adjust_p=adjust_p, adjust_n=adjust_sn,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d,
            validation_momentum=validation_momentum,
            start_validation_epoch=start_validation_epoch)
    cls.train(p_set, sn_set, test_set, p_batch_size, sn_batch_size,
              p_validation, sn_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

if pn_then_pu:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNClassifier(
            model, pi=pi/(pi+rho),
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d,
            validation_momentum=validation_momentum,
            start_validation_epoch=start_validation_epoch)
    cls.train(p_set, sn_set, test_set, p_batch_size, sn_batch_size,
              p_validation, sn_validation,
              cls_training_epochs, convex_epochs=convex_epochs)
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls2 = training.PUClassifier(
            model, pn_model=cls.model, pi=pi, balanced=balanced,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate,
            validation_momentum=validation_momentum,
            start_validation_epoch=start_validation_epoch)
    cls2.train(p_set, u_set, test_set, p_batch_size, u_batch_size,
               p_validation, u_validation,
               cls_training_epochs, convex_epochs=convex_epochs)

if pu_then_pn:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PUClassifier3(
            model, pi=pi, rho=rho,
            lr=learning_rate_cls, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate,
            milestones=milestones, lr_d=lr_d,
            validation_momentum=validation_momentum,
            start_validation_epoch=start_validation_epoch)
    cls.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

    print('')
    model = Net().cuda() if args.cuda else Net()
    cls2 = training.PNClassifier(
            model, pi=pi/(pi+rho), pu_model=cls.model,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d,
            start_validation_epoch=start_validation_epoch)
    cls2.train(p_set, sn_set, test_set, p_batch_size, sn_batch_size,
               p_validation, sn_validation,
               cls_training_epochs, convex_epochs=convex_epochs)

if pu:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PUClassifier(
            model, pi=pi, balanced=balanced,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate,
            validation_momentum=validation_momentum,
            start_validation_epoch=start_validation_epoch)
    cls.train(p_set, u_set, test_set, p_batch_size, u_batch_size,
              p_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

if pnu:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNUClassifier(
            model, pi=pi,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d,
            pn_fraction=non_pu_fraction,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate,
            validation_momentum=validation_momentum,
            start_validation_epoch=start_validation_epoch)
    cls.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

if unbiased_pn:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNClassifier(
            model, pi=pi, lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d,
            validation_momentum=validation_momentum,
            start_validation_epoch=start_validation_epoch)
    cls.train(p_set, sn_set, test_set, p_batch_size, sn_batch_size,
              p_validation, sn_validation,
              cls_training_epochs, convex_epochs=convex_epochs)
