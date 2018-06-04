import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.utils.data

import training
import settings
import gaussian.nets as nets
from gaussian.generate_data import ThreeGaussian


pi = 0.5

p_num = 60
sn_num = 300
u_num = 100

pv_num = 15
snv_num = 75
uv_num = 25

t_num = 60000
es_num = 50000

sep_value = 0.5
adjust_p = False
adjust_sn = True

dre_training_epochs = 50
cls_training_epochs = 100

p_batch_size = 30
sn_batch_size = 150
u_batch_size = 50

learning_rate_dre = 5e-3
learning_rate_cls = 5e-2
weight_decay = 1e-4
validation_momentum = 0.5

non_negative = True
nn_threshold = 0
nn_rate = 1/10000
sigmoid_output = True

partial_n = True
pu = False
pu_then_pn = False


params = OrderedDict([
    ('pi', pi),
    ('\np_num', p_num),
    ('sn_num', sn_num),
    ('u_num', u_num),
    ('\npv_num', pv_num),
    ('snv_num', snv_num),
    ('uv_num', uv_num),
    ('\nt_num', t_num),
    ('es_num', es_num),
    ('\nsep_value', sep_value),
    ('adjust_p', adjust_p),
    ('adjust_sn', adjust_sn),
    ('\ndre_training_epochs', dre_training_epochs),
    ('cls_training_epochs', cls_training_epochs),
    ('\np_batch_size', p_batch_size),
    ('sn_batch_size', sn_batch_size),
    ('u_batch_size', u_batch_size),
    ('\nlearning_rate_dre', learning_rate_dre),
    ('learning_rate_cls', learning_rate_cls),
    ('weight_decay', weight_decay),
    ('validation_momentum', validation_momentum),
    ('\nnon_negative', non_negative),
    ('nn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
    ('sigmoid_output', sigmoid_output),
    ('\npartial_n', partial_n),
    ('pu', pu),
    ('pu_then_pn', pu_then_pn),
])

for key, value in params.items():
    print('{}: {}'.format(key, value))
print('')


parser = argparse.ArgumentParser(description='denstiy ratio gaussian')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

settings.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor


tg = ThreeGaussian()

plt.ion()
plt.axis('off')

rho = tg.estimate_neg_observed_prob(es_num)
print('rho', rho)
print('')

p_samples = tg.draw_positive(p_num)
p_set = torch.utils.data.TensorDataset(
    torch.from_numpy(p_samples))

u_samples = tg.draw_unlabeled(u_num)
u_set = torch.utils.data.TensorDataset(
    torch.from_numpy(u_samples))

sn_samples = tg.draw_observed_negative(sn_num)
sn_set = torch.utils.data.TensorDataset(
    torch.from_numpy(sn_samples))

fig, ax = plt.subplots()
tg.plot_samples()
tg.clear_samples()
plt.title('training')
plt.legend()
plt.show()
plt.pause(0.05)

p_validation = torch.from_numpy(tg.draw_positive(pv_num))
u_validation = torch.from_numpy(tg.draw_unlabeled(uv_num))
sn_validation = torch.from_numpy(tg.draw_observed_negative(snv_num))

# plt.figure()
# tg.plot_samples()
# tg.clear_samples()
# plt.title('validation')
# plt.legend()
# plt.show()

t_samples = tg.draw_unlabeled(t_num)
t_observe_probs = []
for i in range(t_samples.shape[0]):
    x = t_samples[i]
    t_observe_probs.append(tg.observed_prob(x))
t_observe_probs = torch.tensor(t_observe_probs).unsqueeze(1)

test_set_dre = torch.utils.data.TensorDataset(
    torch.from_numpy(t_samples), t_observe_probs)

t_labels = torch.zeros_like(t_observe_probs)
t_labels[t_observe_probs < 1/2] = -1
t_labels[t_observe_probs >= 1/2] = 1

test_set_pre_cls = torch.utils.data.TensorDataset(
    torch.from_numpy(t_samples), t_labels)


tg.clear_samples()
tp_samples = tg.draw_positive(int(t_num/2))
tn_samples = tg.draw_negative(int(t_num/2))
t_samples = np.r_[tp_samples, tn_samples]
t_labels = np.r_[np.ones(int(t_num/2)), -np.ones(int(t_num/2))]

test_set_cls = torch.utils.data.TensorDataset(
    torch.from_numpy(t_samples),
    torch.from_numpy(t_labels).unsqueeze(1))

fig, ax2 = plt.subplots()
tg.positive_samples = tg.positive_samples[:2000]
n_plot_idxs = np.random.choice(int(t_num/2), 2000, replace=False)
tg.negative_samples = np.array(tg.negative_samples)[n_plot_idxs]
tg.plot_samples()
plt.legend()
plt.pause(0.05)


if partial_n:
    print('')
    model = nets.Net(sigmoid_output=sigmoid_output)
    if args.cuda:
        model = model.cuda()
    dre = training.PosteriorProbability(
            model, pi=pi, rho=rho,
            lr=learning_rate_dre, weight_decay=weight_decay)
    dre.train(p_set, sn_set, u_set, test_set_dre,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation, dre_training_epochs)
    CS = dre.model.plot_boundary(ax, cmap='jet', linestyles='dashed')
    CS2 = dre.model.plot_boundary(ax2, cmap='jet', linestyles='dashed')
    fmt = '%.1f'
    plt.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
    plt.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
    plt.pause(0.05)

    print('')
    model = nets.Net().cuda() if args.cuda else nets.Net()
    # model = nets.LinearModel().cuda() if args.cuda else nets.LinearModel()
    cls = training.WeightedClassifier(
            model, dre.model, sep_value=sep_value,
            adjust_p=adjust_p, adjust_sn=adjust_sn,
            pi=pi, rho=rho, lr=learning_rate_cls, weight_decay=weight_decay)
    cls.train(p_set, sn_set, u_set, test_set_cls,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation, cls_training_epochs)
    cls.model.plot_boundary(ax, levels=[0], colors='black')
    cls.model.plot_boundary(ax2, levels=[0], colors='black')
    plt.pause(0.05)


if pu:
    print('')
    # model = nets.LinearModel().cuda() if args.cuda else nets.LinearModel()
    model = nets.Net().cuda() if args.cuda else nets.Net()
    cls = training.PUClassifier(
            model, pi=pi, lr=learning_rate_cls, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, u_set, test_set_cls, p_batch_size, u_batch_size,
              p_validation, u_validation, cls_training_epochs)
    cls.model.plot_boundary(ax, levels=[0], colors='brown')
    cls.model.plot_boundary(ax2, levels=[0], colors='brown')
    plt.pause(0.05)


if pu_then_pn:
    print('')
    model = nets.Net().cuda() if args.cuda else nets.Net()
    # model = nets.LinearModel().cuda() if args.cuda else nets.LinearModel()
    cls = training.PUClassifier3(
            model, pi=pi, rho=rho,
            lr=learning_rate_cls, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, sn_set, u_set, test_set_pre_cls,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation, cls_training_epochs)
    cls.model.plot_boundary(ax, levels=[0], colors='red')
    cls.model.plot_boundary(ax2, levels=[0], colors='red')
    plt.pause(0.05)

    print('')
    model = nets.Net().cuda() if args.cuda else nets.Net()
    # model = nets.LinearModel().cuda() if args.cuda else nets.LinearModel()
    cls2 = training.PNClassifier(
            model, pu_model=cls.model,
            pi=pi/(pi+rho), lr=learning_rate_cls, weight_decay=weight_decay)
    cls2.train(p_set, sn_set, test_set_cls, p_batch_size, sn_batch_size,
               p_validation, sn_validation, cls_training_epochs)
    cls2.model.plot_boundary(ax, levels=[0], colors='blue')
    cls2.model.plot_boundary(ax2, levels=[0], colors='blue')
    plt.pause(0.05)


while not plt.waitforbuttonpress(1):
    pass
