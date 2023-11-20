import argparse
import importlib
import random
import yaml

import numpy as np

import torch
import torch.utils.data
from tensorboardX import SummaryWriter

from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

import training
import settings
from utils import save_checkpoint, load_checkpoint


parser = argparse.ArgumentParser(description='Main File')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--dataset', type=str, default='mnist',
                    help='Name of dataset: mnist or fashion_mnist')

parser.add_argument('--random-seed', type=int, default=None)
parser.add_argument('--params-path', type=str, default=None)
parser.add_argument('--ppe-save-path', type=str, default=None)
parser.add_argument('--ppe-load-path', type=str, default=None)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


prepare_data = importlib.import_module(f'{args.dataset}.pu_biased_n')
params = prepare_data.params
Net = prepare_data.Net
train_data_orig = prepare_data.train_data
test_data_orig = prepare_data.test_data
train_labels_orig = prepare_data.train_labels
test_labels = prepare_data.test_labels


if args.params_path is not None:
    with open(args.params_path) as f:
        params_file = yaml.load(f, Loader=yaml.FullLoader)
    for key in params_file:
        params[key] = params_file[key]


if args.random_seed is not None:
    params['\nrandom_seed'] = args.random_seed

if args.ppe_save_path is not None:
    params['\nppe_save_name'] = args.ppe_save_path

if args.ppe_load_path is not None:
    params['ppe_load_name'] = args.ppe_load_path


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
negative_classes = params.get('negative_classes', None)
neg_ps = params['neg_ps']

non_pu_fraction = params['\nnon_pu_fraction']  # gamma
balanced = params['balanced']

u_per = params['\nu_per']  # tau
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

if 'learning_rate_ppe' in params:
    learning_rate_ppe = params['learning_rate_ppe']
else:
    learning_rate_ppe = learning_rate_cls

milestones = params.get('milestones', [1000])
milestones_ppe = params.get('milestones_ppe', milestones)
lr_d = params.get('lr_d', 1)

non_negative = params['\nnon_negative']
nn_threshold = params['nn_threshold']  # beta
nn_rate = params['nn_rate']

cbs_feature = params.get('\ncbs_feature', False)
cbs_feature_later = params.get('cbs_feature_later', False)
cbs_alpha = params.get('cbs_alpha', 10)
cbs_beta = params.get('cbs_beta', 4)
n_select_features = params.get('n_select_features', 0)
svm = params.get('svm', False)
svm_C = params.get('svm_C', 1)

pu_prob_est = params['\npu_prob_est']
use_true_post = params['use_true_post']

partial_n = params['\npartial_n']  # PUbN
hard_label = params['hard_label']

pn_then_pu = params.get('pn_then_pu', False)
pu_then_pn = params.get('pu_then_pn', False)  # PU -> PN

iwpn = params['\niwpn']
pu = params['pu']
pnu = params['pnu']
unbiased_pn = params.get('unbiased_pn', False)

random_seed = params['\nrandom_seed']

ppe_save_name = params.get('\nppe_save_name', None)
ppe_load_name = params.get('ppe_load_name', None)

log_dir = 'logs/MNIST'
visualize = False

priors = params.get('\npriors', None)
if priors is None:
    priors = [1/num_classes for _ in range(num_classes)]


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
        if negative_classes is None:
            if i not in positive_classes:
                n_idxs[(labels == i).numpy().astype(bool)] = 1
        else:
            if i in negative_classes:
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
    if negative_classes is None:
        selected_u = np.random.choice(len(data), n, replace=False)
    else:
        u_idxs = np.zeros_like(labels)
        for i in range(num_classes):
            if i in positive_classes or i in negative_classes:
                u_idxs[(labels == i).numpy().astype(bool)] = 1
        u_idxs = np.argwhere(u_idxs == 1).reshape(-1)
        selected_u = np.random.choice(u_idxs, n, replace=False)
    return data[selected_u], posteriors(labels[selected_u])


t_labels = torch.zeros(test_labels.size())

for i in range(num_classes):
    if i in positive_classes:
        t_labels[test_labels == i] = 1
    elif negative_classes is None or i in negative_classes:
        t_labels[test_labels == i] = -1
    else:
        t_labels[test_labels == i] = 0


# accs = []
# f1s = []
#
# for i in range(1, 11):
#
#     np.random.seed(i)
#     random.seed(i)
#     torch.manual_seed(i)
#     torch.cuda.manual_seed(i)
#     torch.backends.cudnn.deterministic = True
#
#     idxs = np.random.permutation(len(train_data_orig))
#
#     valid_data = train_data_orig[idxs][u_cut:]
#     valid_labels = train_labels_orig[idxs][u_cut:]
#     train_data = train_data_orig[idxs][:u_cut]
#     train_labels = train_labels_orig[idxs][:u_cut]
#
#     u_data, u_pos = pick_u_data(train_data, train_labels, u_num)
#     p_data, p_pos = pick_p_data(train_data, train_labels, p_num)
#     sn_data, sn_pos = pick_sn_data(train_data, train_labels, sn_num)
#     uv_data, uv_pos = pick_u_data(valid_data, valid_labels, uv_num)
#     pv_data, pv_pos = pick_p_data(valid_data, valid_labels, pv_num)
#     snv_data, snv_pos = pick_sn_data(valid_data, valid_labels, snv_num)
#
#     if cbs_feature:
#         cbs_features = generate_cbs_features(
#             p_data, sn_data, u_data,
#             pv_data, snv_data, uv_data,
#             test_data_orig, n_select_features=n_select_features,
#             alpha=cbs_alpha, beta=cbs_beta)
#         p_data, sn_data, u_data, pv_data, snv_data, uv_data, test_data = \
#             [torch.tensor(data) for data in cbs_features]
#         Net = NetCBS
#     else:
#         test_data = test_data_orig
#
#     clf = LinearSVC(max_iter=5000, C=svm_C)
#     labels = np.concatenate(
#         [np.ones(p_data.shape[0]), -np.ones(sn_data.shape[0])])
#     clf.fit(np.concatenate([p_data, sn_data]), labels)
#     acc = clf.score(test_data, t_labels.numpy())
#     f1 = f1_score(t_labels.numpy(), clf.predict(test_data))
#     accs.append(acc)
#     f1s.append(f1)
#     print('Accuracy: {:.2f}'.format(acc*100))
#     print('F1-score: {:.2f}'.format(f1*100))
#
# print('Avg Accuracy: {:.2f}'.format(np.mean(accs)*100))
# print('Std Accuracy: {:.2f}'.format(np.std(accs)*100))
# print('Avg F1: {:.2f}'.format(np.mean(f1s)*100))
# print('Std F1: {:.2f}'.format(np.std(f1s)*100))


np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True


idxs = np.random.permutation(len(train_data_orig))

valid_data = train_data_orig[idxs][u_cut:]
valid_labels = train_labels_orig[idxs][u_cut:]
train_data = train_data_orig[idxs][:u_cut]
train_labels = train_labels_orig[idxs][:u_cut]


u_data, u_pos = pick_u_data(train_data, train_labels, u_num)
p_data, p_pos = pick_p_data(train_data, train_labels, p_num)
sn_data, sn_pos = pick_sn_data(train_data, train_labels, sn_num)
uv_data, uv_pos = pick_u_data(valid_data, valid_labels, uv_num)
pv_data, pv_pos = pick_p_data(valid_data, valid_labels, pv_num)
snv_data, snv_pos = pick_sn_data(valid_data, valid_labels, snv_num)


# u_pos = torch.rand_like(u_pos)
u_set = torch.utils.data.TensorDataset(u_data, u_pos)
u_validation = uv_data, uv_pos

p_set = torch.utils.data.TensorDataset(p_data, p_pos)
p_validation = pv_data, pv_pos

sn_set = torch.utils.data.TensorDataset(sn_data, sn_pos)
sn_validation = snv_data, snv_pos


test_posteriors = posteriors(test_labels)
test_idxs = np.argwhere(t_labels != 0).reshape(-1)

test_set = torch.utils.data.TensorDataset(
    test_data[test_idxs],
    t_labels.unsqueeze(1).float()[test_idxs],
    test_posteriors[test_idxs])

# print(len(test_set.tensors[0]))


if svm:
    clf = LinearSVC(max_iter=5000)
    labels = np.concatenate(
        [np.ones(p_data.shape[0]), -np.ones(sn_data.shape[0])])
    clf.fit(np.concatenate([p_data, sn_data]), labels)
    print('Accuracy: {:.2f}'.format(
          clf.score(test_data, t_labels.numpy())*100))
    print('F1-score: {:.2f}'.format(
          f1_score(t_labels.numpy(), clf.predict(test_data))*100))


if pu_prob_est and ppe_load_name is None:
    print('')
    model = Net().cuda() if args.cuda else Net()
    ppe = training.PUClassifier3(
            model, pi=pi, rho=rho,
            lr=learning_rate_ppe, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate,
            milestones=milestones_ppe, lr_d=lr_d, prob_est=True)
    ppe.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)
    if ppe_save_name is not None:
        save_checkpoint(ppe.model, cls_training_epochs, ppe_save_name)
    ppe_model = ppe.model

if ppe_load_name is not None:
    ppe_model = Net().cuda() if args.cuda else Net()
    ppe_model = load_checkpoint(ppe_model, ppe_load_name)


if (partial_n or (iwpn and (adjust_p or adjust_sn))) and not use_true_post:
    p_set = torch.utils.data.TensorDataset(
        p_set.tensors[0], torch.sigmoid(
            training.Training()
            .feed_in_batches(ppe_model, p_set.tensors[0])).cpu())
    sn_set = torch.utils.data.TensorDataset(
        sn_set.tensors[0], torch.sigmoid(
            training.Training()
            .feed_in_batches(ppe_model, sn_set.tensors[0])).cpu())
    u_set = torch.utils.data.TensorDataset(
        u_set.tensors[0], torch.sigmoid(
            training.Training()
            .feed_in_batches(ppe_model, u_set.tensors[0])).cpu())
    p_validation = p_validation[0], torch.sigmoid(
        training.Training()
        .feed_in_batches(ppe_model, p_validation[0])).cpu()
    sn_validation = sn_validation[0], torch.sigmoid(
        training.Training()
        .feed_in_batches(ppe_model, sn_validation[0])).cpu()
    u_validation = u_validation[0], torch.sigmoid(
        training.Training()
        .feed_in_batches(ppe_model, u_validation[0])).cpu()

# eta
sep_value = np.percentile(
    u_set.tensors[1].numpy().reshape(-1), int((1-pi-true_rho)*u_per*100))
print('\nsep_value =', sep_value)


u_set = torch.utils.data.TensorDataset(u_data, u_set.tensors[1])
u_validation = uv_data, u_validation[1]

p_set = torch.utils.data.TensorDataset(p_data, p_set.tensors[1])
p_validation = pv_data, p_validation[1]

sn_set = torch.utils.data.TensorDataset(sn_data, sn_set.tensors[1])
sn_validation = snv_data, sn_validation[1]

test_set = torch.utils.data.TensorDataset(
    test_data[test_idxs], test_set.tensors[1], test_set.tensors[2])


if partial_n:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PUbNClassifier(
            model, balanced=balanced, pi=pi, rho=rho,
            sep_value=sep_value,
            adjust_p=adjust_p, adjust_sn=adjust_sn, hard_label=hard_label,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d)
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
            milestones=milestones, lr_d=lr_d)
    cls.train(p_set, sn_set, test_set, p_batch_size, sn_batch_size,
              p_validation, sn_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

if pn_then_pu:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNClassifier(
            model, pi=pi/(pi+rho),
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d)
    cls.train(p_set, sn_set, test_set, p_batch_size, sn_batch_size,
              p_validation, sn_validation,
              cls_training_epochs, convex_epochs=convex_epochs)
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls2 = training.PUClassifier(
            model, pn_model=cls.model, pi=pi, balanced=balanced,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls2.train(p_set, u_set, test_set, p_batch_size, u_batch_size,
               p_validation, u_validation,
               cls_training_epochs, convex_epochs=convex_epochs)

if pu_then_pn:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PUClassifier3(
            model, pi=pi, rho=rho,
            lr=learning_rate_ppe, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate,
            milestones=milestones, lr_d=lr_d)
    cls.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

    print('')
    model = Net().cuda() if args.cuda else Net()
    cls2 = training.PNClassifier(
            model, pi=pi/(pi+rho), pu_model=cls.model,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d)
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
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
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
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

if unbiased_pn:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNClassifier(
            model, pi=pi,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d)
    cls.train(p_set, sn_set, test_set, p_batch_size, sn_batch_size,
              p_validation, sn_validation,
              cls_training_epochs, convex_epochs=convex_epochs)


n_embedding_points = 500

if visualize:

    indx = np.random.choice(test_data.size(0), size=n_embedding_points)

    embedding_data = test_data[indx]
    embedding_labels = t_labels.numpy().copy()
    # Negative data that are not sampled
    embedding_labels[test_posteriors.numpy().flatten() < 1/2] = 0
    embedding_labels = embedding_labels[indx]
    features = cls.last_layer_activation(embedding_data)
    writer = SummaryWriter(log_dir=log_dir)
    # writer.add_embedding(embedding_data.view(n_embedding_points, -1),
    #                      metadata=embedding_labels,
    #                      tag='Input', global_step=0)
    writer.add_embedding(features, metadata=embedding_labels,
                         tag='PUbN Features',
                         global_step=cls_training_epochs)
