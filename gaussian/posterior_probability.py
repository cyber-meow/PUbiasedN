import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import training
import settings
from gaussian.generate_data import TwoGaussian


pi = 0.5

p_num = 500
on_num = 1000
u_num = 6000

pv_num = 100
onv_num = 200
uv_num = 1200

t_num = 60000
es_num = 50000

training_epochs = 100
p_batch_size = 250
sn_batch_size = 500
u_batch_size = 3000
n_batches = 2

learning_rate = 5e-3
weight_decay = 1e-4
validation_momentum = 0.5

non_negative = False
nn_threshold = -0.02
nn_rate = 1/50000
adjust_ux = False
adjust_rate = 1/10
train_valid = False
sigmoid_output = True


params = OrderedDict([
    ('pi', pi),
    ('\np_num', p_num),
    ('on_num', on_num),
    ('u_num', u_num),
    ('\npv_num', pv_num),
    ('onv_num', onv_num),
    ('uv_num', uv_num),
    ('\nt_num', t_num),
    ('es_num', es_num),
    ('\ntraining_epochs', training_epochs),
    ('p_batch_size', p_batch_size),
    ('sn_batch_size', sn_batch_size),
    ('u_batch_size', u_batch_size),
    ('\nlearning_rate', learning_rate),
    ('weight_decay', weight_decay),
    ('validation_momentum', validation_momentum),
    ('\nnon_negative', non_negative),
    ('nn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
    ('\nadjust_ux', adjust_ux),
    ('adjust_rate', adjust_rate),
    ('\ntrain_valid', train_valid),
    ('sigmoid_output', sigmoid_output),
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


tg = TwoGaussian()

plt.ion()

rho = tg.estimate_neg_observed_prob(es_num)
print('rho', rho)

p_es = tg.draw_positive(es_num, store=False)
on_es = tg.draw_observed_negative(5000, store=False)

print('drawn')

posteriors = []
for i in range(p_es.shape[0]):
    x = p_es[i]
    posteriors.append(tg.observed_prob(x))
M = np.mean(np.array(posteriors)) * pi

print(M)

posteriors = []
for i in range(on_es.shape[0]):
    x = on_es[i]
    posteriors.append(tg.observed_prob(x))
M += np.mean(np.array(posteriors)) * rho

print('M', M)
print('')

p_samples = tg.draw_positive(p_num)
p_set = torch.utils.data.TensorDataset(
    torch.from_numpy(p_samples))

u_samples = tg.draw_unlabeled(u_num)
u_set = torch.utils.data.TensorDataset(
    torch.from_numpy(u_samples))

on_samples = tg.draw_observed_negative(on_num)
on_set = torch.utils.data.TensorDataset(
    torch.from_numpy(on_samples))

fig, ax = plt.subplots()
tg.plot_samples()
tg.clear_samples()
plt.title('training')
plt.legend()
plt.show()
plt.pause(0.05)

p_validation = torch.from_numpy(tg.draw_positive(pv_num))
u_validation = torch.from_numpy(tg.draw_unlabeled(uv_num))
sn_validation = torch.from_numpy(tg.draw_observed_negative(onv_num))

# plt.figure()
# tg.plot_samples()
# tg.clear_samples()
# plt.title('validation')
# plt.legend()
# plt.show()

t_samples = tg.draw_unlabeled(t_num)
t_posteriors = []
for i in range(t_samples.shape[0]):
    x = t_samples[i]
    t_posteriors.append(tg.positive_posterior(x))

test_set = torch.utils.data.TensorDataset(
    torch.from_numpy(t_samples).float(),
    torch.tensor(t_posteriors).unsqueeze(1).float())


class PosteriorProbability(training.PosteriorProbability):

    def __init__(self, model, nn=False, nn_threshold=-0.02, *args, **kwargs):
        self.times = 0
        self.nn = nn
        self.nn_threshold = nn_threshold
        super().__init__(model, *args, **kwargs)

    def compute_loss(self, px, snx, ux):
        fpx = self.model(px.type(settings.dtype))
        fsnx = self.model(snx.type(settings.dtype))
        fux = self.model(ux.type(settings.dtype))
        fpx_mean = torch.mean(fpx)
        fsnx_mean = torch.mean(fsnx)
        fux_mean = torch.mean(fux)
        fux2_mean = torch.mean(fux**2)
        true_loss = fux2_mean - 2*fpx_mean*self.pi - 2*fsnx_mean*self.rho
        loss = true_loss
        print(fux2_mean.item(),
              fpx_mean.item()*self.pi + fsnx_mean.item()*self.rho,
              fux_mean.item(), true_loss.item())
        if self.nn and loss + M < self.nn_threshold:
            loss = -loss * nn_rate
        if adjust_ux and torch.mean(fux) > self.pi+self.rho and self.times > 3:
            loss = fux_mean * adjust_rate
        self.times += 1
        return loss.cpu(), true_loss.cpu()


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(2, 10)
        self.linear2 = nn.Linear(10, 5)
        self.linear3 = nn.Linear(5, 1)
        # self.linear4 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        # x = F.dropout(x, training=self.training)
        if sigmoid_output:
            x = F.sigmoid(self.linear3(x))
        else:
            x = F.relu(self.linear3(x))
        return x

    def border_func(self, x, y):
        inp = torch.tensor([x, y]).type(settings.dtype)
        return self.forward(inp).detach().cpu().numpy()

    def plot_boundary(self, ax, **kwargs):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xs = np.linspace(xmin, xmax, 100)
        ys = np.linspace(ymin, ymax, 100)
        xv, yv = np.meshgrid(xs, ys)
        border_func = np.vectorize(self.border_func)
        cont = plt.contour(xv, yv, border_func(xv, yv), **kwargs)
        return cont


model = Net().cuda() if args.cuda else Net()
cls = PosteriorProbability(
        model, pi=pi, rho=rho, lr=learning_rate, weight_decay=weight_decay,
        nn=non_negative, nn_threshold=nn_threshold)
cls.train(p_set, on_set, u_set, test_set,
          p_batch_size, sn_batch_size, u_batch_size,
          p_validation, sn_validation, u_validation, training_epochs)
CS = cls.model.plot_boundary(ax, cmap='jet')
fmt = '%.1f'
plt.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)


while not plt.waitforbuttonpress(1):
    pass
