import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


s_num = 500
u_num = 6000
sv_num = 100
uv_num = 1200
t_num = 60000

training_epochs = 400
s_batch_size = 50
u_batch_size = 600
learning_rate = 5e-3
weight_decay = 1e-4

non_negative = False
and_nn = False
nn_threshold = 0
nn_rate = 1/1000
adjust_ux = False
adjust_rate = 1/10


params = OrderedDict([
    ('s_num', s_num),
    ('u_num', u_num),
    ('sv_num', sv_num),
    ('uv_num', uv_num),
    ('t_num', t_num),
    ('\ntraining_epochs', training_epochs),
    ('s_batch_size', s_batch_size),
    ('u_batch_size', u_batch_size),
    ('learning_rate', learning_rate),
    ('weight_decay', weight_decay),
    ('\nnon_negative', non_negative),
    ('nn_threshold', nn_threshold),
    ('and_nn', and_nn),
    ('nn_rate', nn_rate),
    ('\nadjust_ux', adjust_ux),
    ('adjust_rate', adjust_rate),
])

for key, value in params.items():
    print('{}: {}'.format(key, value))
print('')


parser = argparse.ArgumentParser(description='denstiy ratio gaussian')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor


class TwoGaussian(object):

    def __init__(self, **kwargs):
        self.mu1 = np.array([0, 0])
        self.mu2 = np.array([2, 2])
        self.cov = np.array([[1, 0], [0, 1]])
        self.positive_prior = 0.5
        self.positive_samples = []
        self.negative_samples = []
        self.__dict__.update(kwargs)

    def positive_importance(self, x):
        conditional_positive = (
            np.exp(-0.5*(x-self.mu1).T.dot(x-self.mu1)) / (2*np.pi))
        conditional_negative = (
            np.exp(-0.5*(x-self.mu2).T.dot(x-self.mu2)) / (2*np.pi))
        marginal_dist = (
            self.positive_prior * conditional_positive
            + (1-self.positive_prior) * conditional_negative)
        positive_importance = conditional_positive / marginal_dist
        return positive_importance

    def draw_positive(self, n, store=True):
        drawn = np.random.multivariate_normal(self.mu1, self.cov, n)
        if store:
            self.positive_samples.extend(drawn)
        return drawn

    def draw_negative(self, n, store=True):
        drawn = np.random.multivariate_normal(self.mu2, self.cov, n)
        if store:
            self.negative_samples.extend(drawn)
        return drawn

    def plot_samples(self):
        px, py = np.array(self.positive_samples).T
        nx, ny = np.array(self.negative_samples).T
        plt.scatter(px, py, color='salmon', s=3)
        plt.scatter(nx, ny, color='turquoise', s=3)

    def clear_samples(self):
        self.positive_samples = []
        self.negative_samples = []


tg = TwoGaussian()

s_samples = tg.draw_positive(s_num)
s_set = torch.utils.data.TensorDataset(
    torch.from_numpy(s_samples).float())

p_samples = tg.draw_positive(int(u_num/2))
n_samples = tg.draw_negative(int(u_num/2))
u_samples = np.r_[p_samples, n_samples]
u_samples = np.random.permutation(u_samples)
u_set = torch.utils.data.TensorDataset(
    torch.from_numpy(u_samples).float())


s_validation = torch.from_numpy(tg.draw_positive(sv_num))

pv_samples = tg.draw_positive(int(uv_num/2))
nv_samples = tg.draw_negative(int(uv_num/2))
uv_samples = np.r_[pv_samples, nv_samples]
u_validation = torch.from_numpy(np.random.permutation(uv_samples))

tg.clear_samples()

tp_samples = tg.draw_positive(int(t_num/2))
tn_samples = tg.draw_negative(int(t_num/2))
t_samples = np.r_[tp_samples, tn_samples]

t_importances = []
for i in range(t_samples.shape[0]):
    x = t_samples[i]
    t_importances.append(tg.positive_importance(x))
t_importances = np.array(t_importances)
M = np.mean(t_importances[:int(t_num/2)])
# M = np.mean(np.log(t_importances)[:int(t_num/2)])-1
print('expected M', M)

test_set = torch.utils.data.TensorDataset(
    torch.from_numpy(t_samples).float(),
    torch.from_numpy(t_importances).unsqueeze(1).float())

tg.plot_samples()
# plt.show()


class Classifier(object):

    def __init__(self, model, lr=5e-3,
                 weight_decay=1e-2, nn=True, nn_threshold=0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.nn = nn
        self.nn_threshold = nn_threshold
        self.test_accuracies = []
        self.init_optimizer()
        self.times = 0

    def init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay)

    def train(self, s_set, u_set, test_set,
              s_batch_size, u_batch_size, num_epochs,
              test_interval=1, print_interval=1):

        self.init_optimizer()
        self.test(test_set, True)

        s_loader = torch.utils.data.DataLoader(
            s_set, batch_size=s_batch_size,
            shuffle=True, num_workers=1)

        u_loader = torch.utils.data.DataLoader(
            u_set, batch_size=u_batch_size,
            shuffle=True, num_workers=1)

        for epoch in range(num_epochs):

            total_loss = self.train_step(s_loader, u_loader)

            if (epoch+1) % test_interval == 0 or epoch+1 == num_epochs:

                to_print = (epoch+1) % print_interval == 0
                if to_print:
                    sys.stdout.write('Epoch: {}  '.format(epoch))
                    print('Train Loss: {:.6f}'.format(total_loss))
                self.test(test_set, to_print)

    def train_step(self, s_loader, u_loader):
        self.model.train()
        total_loss = 0
        compute_loss = self.compute_loss_density
        for x in s_loader:
            self.optimizer.zero_grad()
            ux = next(iter(u_loader))[0]
            loss, true_loss = compute_loss(x[0], ux)
            total_loss += true_loss.item()
            loss = loss
            loss.backward()
            self.optimizer.step()
        return total_loss

    def compute_loss_density(self, sx, ux):
        fsx = self.model(sx.type(dtype))
        fux = self.model(ux.type(dtype))
        fux2_mean = torch.mean(fux**2)
        fux_mean = torch.mean(fux)
        fsx_mean = torch.mean(fsx)
        true_loss = fux2_mean/2 - fsx_mean
        loss = true_loss
        M_estimated = torch.mean((5-fux)**2-15).item()
        M2_estimated = torch.mean((3+fux)**2-15).item()
        print(fux2_mean.item(), fsx_mean.item(),
              fux_mean.item(), true_loss.item()*2)
        print(M_estimated, M2_estimated)
        fsvx = self.model(s_validation.type(dtype))
        fuvx = self.model(u_validation.type(dtype))
        # print('valid', (torch.mean(fuvx**2)/2 - torch.mean(fsvx)).item())
        print('valid', (torch.mean(fuvx**2)/2-torch.mean(fsvx)).item())
        # if torch.mean(fsx) == 0:
        #     loss = -torch.mean(fsx)
        # if self.times > 10:
        #     loss += (torch.mean(fux)-1)**2/10
        if self.nn and loss + M/2 < self.nn_threshold:
            loss = -loss * nn_rate
        if and_nn and loss + M_estimated/2 < self.nn_threshold:
            loss = -loss * nn_rate
        if adjust_ux and torch.mean(fux) > 1 and self.times > 3:
            loss = fsx_mean * adjust_rate
        self.times += 1
        return loss.cpu(), true_loss.cpu()

    def test(self, test_set, to_print=True):
        self.model.eval()
        x = test_set.tensors[0].type(dtype)
        target = test_set.tensors[1]
        output = self.model(x).cpu()
        error = torch.mean((target-output)**2).item()
        if to_print:
            print('Test set: Error: {}'.format(error))


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(2, 10)
        self.linear2 = nn.Linear(10, 20)
        self.linear3 = nn.Linear(20, 5)
        self.linear4 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.linear4(x))
        return x


model = Net().cuda() if args.cuda else Net()
cls = Classifier(model, lr=learning_rate, weight_decay=weight_decay,
                 nn=non_negative, nn_threshold=nn_threshold)
cls.train(s_set, u_set, test_set, s_batch_size, u_batch_size, training_epochs)
