import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.utils.data
import torch.nn.functional as F

import training
import settings
import gaussian.nets as nets
from gaussian.generate_data import ThreeGaussian


pi = 0.5

p_num = 50
sn_num = 50
u_num = 100

pv_num = 50
snv_num = 50
uv_num = 100

t_num = 60000
es_num = 50000

sep_value = 0.5
adjust_p = True
adjust_sn = True

training_epochs = 60
convex_epochs = 60

p_batch_size = 50
sn_batch_size = 50
u_batch_size = 100

learning_rate = 1e-2
weight_decay = 1e-4
validation_momentum = 0
start_validation_epoch = 45

non_negative = True
nn_threshold = 0
nn_rate = 1

partial_n = True
pu = True
unbiased_pn = True


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
    ('\ntraining_epochs', training_epochs),
    ('convex_epochs', convex_epochs),
    ('\np_batch_size', p_batch_size),
    ('sn_batch_size', sn_batch_size),
    ('u_batch_size', u_batch_size),
    ('\nlearning_rate', learning_rate),
    ('weight_decay', weight_decay),
    ('validation_momentum', validation_momentum),
    ('\nnon_negative', non_negative),
    ('nn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
    ('\npartial_n', partial_n),
    ('pu', pu),
    ('unbiased_pn', unbiased_pn),
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
tg.plot_samples(s=23)
tg.clear_samples()
# plt.title('training')
plt.legend()
plt.tight_layout()
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

t_labels = torch.zeros_like(t_observe_probs)
t_labels[t_observe_probs < 1/2] = -1
t_labels[t_observe_probs >= 1/2] = 1

test_set_dre = torch.utils.data.TensorDataset(
    torch.from_numpy(t_samples), t_labels, t_observe_probs)


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
# plt.title('test')
plt.legend()
plt.tight_layout()
plt.pause(0.05)

tp_set = torch.utils.data.TensorDataset(
    torch.from_numpy(tg.draw_positive(30000, store=False)))
tn_set = torch.utils.data.TensorDataset(
    torch.from_numpy(tg.draw_negative(30000, store=False)))
tp_validation = torch.from_numpy(tg.draw_positive(10000, store=False))
tn_validation = torch.from_numpy(tg.draw_negative(10000, store=False))

hdls, lbs = [], []


if partial_n:
    print('')
    model = nets.Net()
    if args.cuda:
        model = model.cuda()
    ppe = training.PUClassifier3(
            model, pi=pi, rho=rho,
            lr=learning_rate, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate,
            start_validation_epoch=start_validation_epoch,
            prob_est=True, validation_momentum=validation_momentum)
    ppe.train(p_set, sn_set, u_set, test_set_dre,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              training_epochs, convex_epochs=convex_epochs)
    # CS = ppe.model.plot_boundary(ax, cmap='jet', linestyles='dashed')
    # CS2 = ppe.model.plot_boundary(ax2, cmap='jet', linestyles='dashed')
    # fmt = '%.1f'
    # plt.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
    # plt.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
    # plt.pause(0.05)

    ppe_model = ppe.model

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
        u_set.tensors[1].numpy().reshape(-1), int((1-pi-rho)*100))
    print('\nsep_value =', sep_value)

    print('')
    model = nets.Net().cuda() if args.cuda else nets.Net()
    cls = training.WeightedClassifier(
            model, sep_value=sep_value,
            pi=pi, rho=rho, lr=learning_rate,
            adjust_p=adjust_p, adjust_sn=adjust_sn,
            weight_decay=weight_decay,
            start_validation_epoch=start_validation_epoch,
            validation_momentum=validation_momentum)
    cls.train(p_set, sn_set, u_set, test_set_cls,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              training_epochs, convex_epochs=convex_epochs)
    PUBN = plt.Line2D((0, 1), (0, 0), color='brown',
                      linestyle='-', linewidth=1.5)
    hdls.append(PUBN)
    lbs.append('PUbN (proposed)')
    for ax_c in [ax, ax2]:
        cls.model.plot_boundary(
            ax_c, levels=[0], colors='brown',
            linestyles='-', linewidths=1.5)
    plt.pause(0.05)


if pu:
    print('')
    model = nets.Net().cuda() if args.cuda else nets.Net()
    cls = training.PUClassifier(
            model, pi=pi,
            lr=learning_rate, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate,
            start_validation_epoch=start_validation_epoch,
            validation_momentum=validation_momentum)
    cls.train(p_set, u_set, test_set_cls, p_batch_size, u_batch_size,
              p_validation, u_validation,
              training_epochs, convex_epochs=convex_epochs)
    nnPU = plt.Line2D((0, 1), (0, 0), color='darkslategrey',
                      linestyle='--', linewidth=1.5)
    hdls.append(nnPU)
    lbs.append('nnPU')
    for ax_c in [ax, ax2]:
        cls.model.plot_boundary(
            ax_c, levels=[0], colors='darkslategrey',
            linestyles='--', linewidths=1.5)
    plt.pause(0.05)


if unbiased_pn:
    print('')
    model = nets.Net().cuda() if args.cuda else nets.Net()
    cls = training.PNClassifier(
            model, pi=pi,
            lr=learning_rate, weight_decay=weight_decay,
            start_validation_epoch=start_validation_epoch,
            validation_momentum=validation_momentum)
    cls.train(tp_set, tn_set, test_set_cls, p_batch_size, sn_batch_size,
              tp_validation, tn_validation,
              2, convex_epochs=convex_epochs)
    PN = plt.Line2D((0, 1), (0, 0), color='black',
                    linestyle=':', linewidth=1.5)
    hdls.append(PN)
    lbs.append('supervised (oracle)')
    for ax_c in [ax, ax2]:
        cls.model.plot_boundary(
            ax_c, levels=[0], colors='black',
            linestyles=':', linewidths=1.5)
    plt.pause(0.05)


for ax_c in [ax, ax2]:
    handles, labels = ax_c.get_legend_handles_labels()
    handles = hdls + handles
    labels = lbs + labels
    ax_c.legend(handles, labels, ncol=2, loc=4)

while not plt.waitforbuttonpress(1):
    pass
