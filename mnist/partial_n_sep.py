import sys
import argparse
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


p_num = 3000
sn_num = 10000
u_num = 60000
pi = 0.49
pho = 0.1

training_epochs = 200
p_batch_size = 300
sn_batch_size = 1000
u_batch_size = 6000
learning_rate = 1e-3
weight_decay = 1e-2

loss_type = 'density'


params = OrderedDict([
    ('p_num', p_num),
    ('sn_num', sn_num),
    ('u_num', u_num),
    ('pi', pi),
    ('pho', pho),
    ('\ntraining_epochs', training_epochs),
    ('p_batch_size', p_batch_size),
    ('sn_batch_size', p_batch_size),
    ('u_batch_size', u_batch_size),
    ('learning_rate', learning_rate),
    ('weight_decay', weight_decay),
    ('\nloss_type', loss_type),
])

for key, value in params.items():
    print('{}: {}'.format(key, value))
print('')


parser = argparse.ArgumentParser(description='MMD matching PU')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor


# torchvision.datasets.MNIST outputs a set of PIL images
# We transform them to tensors
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

# Load and transform data
mnist = torchvision.datasets.MNIST(
    './data/MNIST', train=True, download=True, transform=transform)

mnist_test = torchvision.datasets.MNIST(
    './data/MNIST', train=False, download=True, transform=transform)


train_labels = mnist.train_labels.numpy()

p_idxs = np.argwhere(train_labels % 2 == 0).reshape(-1)
selected_p = np.random.choice(p_idxs, p_num, replace=False)

p_set = torch.utils.data.TensorDataset(
    mnist.train_data[selected_p].unsqueeze(1).float())

sn_idxs = np.argwhere(train_labels % 2 == 1).reshape(-1)
# sn_idxs = np.argwhere(
#     np.logical_and(train_labels % 2 == 1, train_labels < 6)).reshape(-1)
selected_sn = np.random.choice(sn_idxs, sn_num, replace=False)

sn_set = torch.utils.data.TensorDataset(
    mnist.train_data[selected_sn].unsqueeze(1).float())

u_idxs = np.random.choice(60000, u_num, replace=False)

u_set = torch.utils.data.TensorDataset(
    mnist.train_data[u_idxs].unsqueeze(1).float())

test_labels = deepcopy(mnist_test.test_labels)
test_labels[test_labels % 2 == 1] = -1
test_labels[test_labels % 2 == 0] = 1

test_set = torch.utils.data.TensorDataset(
    mnist_test.test_data.unsqueeze(1).float(),
    test_labels.unsqueeze(1).float())

test_labels = deepcopy(mnist_test.test_labels)
test_labels[test_labels == 7] = -1
test_labels[test_labels == 9] = -1
test_labels[test_labels != -1] = 1

test_set2 = torch.utils.data.TensorDataset(
    mnist_test.test_data.unsqueeze(1).float(),
    test_labels.unsqueeze(1).float())


class Classifier(object):

    def __init__(self, model, pi=0.49, pho=0.3, lr=5e-3,
                 weight_decay=1e-2, nn=True,
                 loss_type='density', dr_cls=None, pu_model=None):
        self.model = model
        self.pi = pi
        self.pho = pho
        self.lr = lr
        self.weight_decay = weight_decay
        self.nn = nn
        self.loss_type = loss_type
        if loss_type == 'density':
            self.compute_loss = self.compute_loss_density
        elif loss_type == 'pnu':
            self.compute_loss = self.compute_loss_pnu
        else:
            self.compute_loss = self.compute_loss_pu
        if dr_cls is not None:
            self.dr_model = dr_cls.model
            self.dr_cls = dr_cls
            self.loss_type = 'weighted'
            self.compute_loss = self.compute_loss_weighted
        if pu_model is not None:
            self.pu_model = pu_model
            self.loss_type = 'pn'
            self.compute_loss = self.compute_loss_pn
        self.test_accuracies = []
        self.init_optimizer()

    def init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay)

    def train(self, p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              num_epochs, test_interval=1, print_interval=1, dr_epochs=None):

        self.init_optimizer()
        self.test(test_set, True)

        p_loader = torch.utils.data.DataLoader(
            p_set, batch_size=p_batch_size,
            shuffle=True, num_workers=1)

        sn_loader = torch.utils.data.DataLoader(
            sn_set, batch_size=sn_batch_size,
            shuffle=True, num_workers=1)

        u_loader = torch.utils.data.DataLoader(
            u_set, batch_size=u_batch_size,
            shuffle=True, num_workers=1)

        for epoch in range(num_epochs):

            convex = True if epoch < 5 else False
            total_loss = self.train_step(
                p_loader, sn_loader, u_loader, convex)

            if (epoch+1) % test_interval == 0 or epoch+1 == num_epochs:

                to_print = (epoch+1) % print_interval == 0
                if to_print:
                    sys.stdout.write('Epoch: {}  '.format(epoch))
                    print('Train Loss: {:.6f}'.format(total_loss))
                self.test(test_set, to_print)

            if dr_epochs is not None and (epoch+1) % dr_epochs == 0:
                sys.stdout.write('Train dr model ')
                self.dr_cls.train_step(p_loader, sn_loader, u_loader, True)
                self.dr_cls.test(test_set, True)

    def train_step(self, p_loader, sn_loader, u_loader, convex):
        self.model.train()
        total_loss = 0
        for x in p_loader:
            self.optimizer.zero_grad()
            snx = next(iter(sn_loader))[0]
            if self.loss_type == 'pn':
                loss = self.compute_loss(x[0], snx, convex)
            elif self.loss_type == 'pnu':
                ux = next(iter(u_loader))[0]
                snx2 = next(iter(sn_loader))[0]
                loss = self.compute_loss(x[0], snx, snx2, ux, convex)
            else:
                ux = next(iter(u_loader))[0]
                loss = self.compute_loss(x[0], snx, ux, convex)
            total_loss += loss.item()
            loss = loss
            loss.backward()
            self.optimizer.step()
        return total_loss

    def basic_loss(self, fx, convex):
        if convex:
            negative_logistic = nn.LogSigmoid()
            return -negative_logistic(fx)
        else:
            sigmoid = nn.Sigmoid()
            return sigmoid(-fx)

    def compute_loss_pu(self, px, snx, ux, convex):
        fpx = self.model(px.type(dtype))
        fsnx = self.model(snx.type(dtype))
        print(torch.mean(F.sigmoid(fsnx)).item())
        fux = self.model(ux.type(dtype))
        p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
        sn_loss = self.pho * torch.mean(self.basic_loss(fsnx, convex))
        n_loss = (torch.mean(self.basic_loss(-fux, convex))
                  - self.pho * torch.mean(self.basic_loss(-fsnx, convex))
                  - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
        loss = p_loss + sn_loss + n_loss if n_loss > 0 else -n_loss/10
        return loss.cpu()

    def compute_loss_pn(self, px, nx, convex):
        fpx = self.model(px.type(dtype))
        fnx = self.model(nx.type(dtype))
        p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
        n_loss = self.pho * torch.mean(self.basic_loss(-fnx, convex))
        loss = p_loss + n_loss
        return loss.cpu()

    def compute_loss_pnu(self, px, snx, snx2, ux, convex):
        fpx = self.model(px.type(dtype))
        fsnx = self.model(snx.type(dtype))
        fsnx2 = self.model(snx2.type(dtype))
        fux = self.model(ux.type(dtype))
        p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
        sn_loss = self.pho * torch.mean(self.basic_loss(-fsnx, convex))
        n_loss = (torch.mean(self.basic_loss(-fux, convex))
                  - self.pho * torch.mean(self.basic_loss(-fsnx2, convex))
                  - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
        loss = p_loss + sn_loss + n_loss if n_loss > 0 else -n_loss/10
        return loss.cpu()

    def compute_loss_weighted(self, px, snx, ux, convex):
        fpx = self.model(px.type(dtype))
        fsnx = self.model(snx.type(dtype))
        fux = self.model(ux.type(dtype))
        fux_prob = F.sigmoid(self.dr_model(ux.type(dtype)))
        pred_nprob = 1-torch.mean(fux_prob).item()
        loss = (
            self.pi * torch.mean(self.basic_loss(fpx, convex))
            + self.pho * torch.mean(self.basic_loss(-fsnx, convex))
            + torch.mean(self.basic_loss(-fux, convex) * (1-fux_prob))
            * (1-self.pi-self.pho) / pred_nprob)
        return loss.cpu()

    def compute_loss_density(self, px, snx, ux, convex):
        fpx = F.sigmoid(self.model(px.type(dtype)))
        fsnx = F.sigmoid(self.model(snx.type(dtype)))
        print(torch.mean(fsnx).item())
        fux = F.sigmoid(self.model(ux.type(dtype)))
        loss = (torch.mean(fux**2)/2
                - torch.mean(fpx) * self.pi
                - torch.mean(fsnx) * self.pho)
        loss = loss if loss > -(self.pi+self.pho)/2 else -loss/10
        return loss.cpu()

    def test(self, test_set, to_print=True):
        if hasattr(self, 'pu_model'):
            self.test_two_stage(test_set, to_print)
            return
        self.model.eval()
        x = test_set.tensors[0].type(dtype)
        target = test_set.tensors[1].type(dtype)
        output = self.model(x)
        pred = torch.sign(output)
        correct = torch.sum(pred.eq(target).float()).item()
        accuracy = 100 * correct/len(test_set)
        self.test_accuracies.append(accuracy)
        if to_print:
            print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                    correct, len(test_set), accuracy))

    def test_two_stage(self, test_set, to_print=True):
        self.pu_model.eval()
        self.model.eval()
        x = test_set.tensors[0].type(dtype)
        target = test_set.tensors[1].type(dtype)
        output1 = self.pu_model(x)
        pred1 = torch.sign(output1)
        output2 = self.model(x)
        pred = torch.sign(output2)
        pred[pred1 == -1] = -1
        correct = torch.sum(pred.eq(target).float()).item()
        accuracy = 100 * correct/len(test_set)
        self.test_accuracies.append(accuracy)
        if to_print:
            print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                    correct, len(test_set), accuracy))


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
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
        return x


model = Net().cuda() if args.cuda else Net()
cls = Classifier(model, pi=pi, pho=pho, lr=learning_rate,
                 weight_decay=weight_decay, loss_type=loss_type)
cls.train(p_set, sn_set, u_set, test_set,
          p_batch_size, sn_batch_size, u_batch_size, 30)

print('')
model = Net().cuda() if args.cuda else Net()
cls2 = Classifier(model, pi=pi, pho=pho, lr=learning_rate,
                  weight_decay=weight_decay, dr_cls=cls)
cls2.train(p_set, sn_set, u_set, test_set,
           p_batch_size, sn_batch_size, u_batch_size, 100, dr_epochs=20)

# print('')
# model = Net().cuda() if args.cuda else Net()
# cls2 = Classifier(model, pi=pi, pho=pho, lr=learning_rate,
#                   weight_decay=weight_decay, pu_model=cls.model)
# cls2.train(p_set, sn_set, u_set, test_set,
#            p_batch_size, sn_batch_size, u_batch_size, 100)
