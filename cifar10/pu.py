import sys
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


p_num = 1000
u_num = 50000
pi = 0.4

training_epochs = 50
p_batch_size = 100
u_batch_size = 5000
learning_rate = 1e-3
weight_decay = 1e-2

non_negative = True
loss_type = 'pu'


params = OrderedDict([
    ('p_num', p_num),
    ('u_num', u_num),
    ('pi', pi),
    ('\ntraining_epochs', training_epochs),
    ('p_batch_size', p_batch_size),
    ('u_batch_size', u_batch_size),
    ('learning_rate', learning_rate),
    ('weight_decay', weight_decay),
    ('\nnon_negative', non_negative),
    ('loss_type', loss_type),
])

for key, value in params.items():
    print('{}: {}'.format(key, value))
print('')


parser = argparse.ArgumentParser(description='PU CIFAR10')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load and transform data
cifar10 = torchvision.datasets.CIFAR10(
    './data', train=True, download=True, transform=transform)

cifar10_test = torchvision.datasets.CIFAR10(
    './data', train=False, download=True, transform=transform)


train_labels = torch.tensor(cifar10.train_labels)
p_idxs = np.argwhere(
    np.logical_or(train_labels < 2, train_labels > 7)).reshape(-1)
selected_p = np.random.choice(p_idxs, p_num, replace=False)

u_idxs = np.random.choice(50000, u_num, replace=False)

p_set = torch.utils.data.TensorDataset(
    torch.from_numpy(cifar10.train_data[selected_p]).permute(0, 3, 1, 2))

u_set = torch.utils.data.TensorDataset(
    torch.from_numpy(cifar10.train_data[u_idxs]).permute(0, 3, 1, 2))

test_labels = torch.tensor(cifar10_test.test_labels)
test_labels[test_labels < 2] = 1
test_labels[test_labels > 7] = 1
test_labels[test_labels != 1] = -1

test_set = torch.utils.data.TensorDataset(
    torch.from_numpy(cifar10_test.test_data).permute(0, 3, 1, 2),
    test_labels.unsqueeze(1).float())


class Classifier(object):

    def __init__(self, model, pi=0.49, lr=5e-3, weight_decay=1e-2,
                 nn=True, loss_type='density', dr_cls=None):
        self.model = model
        self.pi = pi
        self.lr = lr
        self.weight_decay = weight_decay
        self.nn = nn
        self.loss_type = loss_type
        if loss_type == 'density':
            self.compute_loss = self.compute_loss_density
        else:
            self.compute_loss = self.compute_loss_pu
        if dr_cls is not None:
            self.dr_model = dr_cls.model
            self.dr_cls = dr_cls
            self.compute_loss = self.compute_loss_weighted
        self.test_accuracies = []
        self.init_optimizer()

    def init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay)

    def train(self, p_set, u_set, test_set,
              p_batch_size, u_batch_size, num_epochs,
              test_interval=1, print_interval=1, dr_epochs=None):

        self.init_optimizer()
        self.test(test_set, True)

        p_loader = torch.utils.data.DataLoader(
            p_set, batch_size=p_batch_size,
            shuffle=True, num_workers=2)

        u_loader = torch.utils.data.DataLoader(
            u_set, batch_size=u_batch_size,
            shuffle=True, num_workers=2)

        for epoch in range(num_epochs):

            convex = True if epoch < 3 else False
            total_loss = self.train_step(p_loader, u_loader, convex)

            if (epoch+1) % test_interval == 0 or epoch+1 == num_epochs:

                to_print = (epoch+1) % print_interval == 0
                if to_print:
                    sys.stdout.write('Epoch: {}  '.format(epoch))
                    print('Train Loss: {:.6f}'.format(total_loss))
                self.test(test_set, to_print)

            if dr_epochs is not None and (epoch+1) % dr_epochs == 0:
                sys.stdout.write('Train dr model ')
                self.dr_cls.train_step(p_loader, u_loader)
                self.dr_cls.test(test_set, True)

    def train_step(self, p_loader, u_loader, convex=False):
        self.model.train()
        total_loss = 0
        for x in p_loader:
            self.optimizer.zero_grad()
            px = x[0]
            ux = next(iter(u_loader))[0]
            loss = self.compute_loss(px, ux, convex)
            total_loss += loss.item()
            loss = loss
            loss.backward()
            self.optimizer.step()
        return total_loss

    def basic_loss(self, fx, convex=False):
        if convex:
            negative_logistic = nn.LogSigmoid()
            return -negative_logistic(fx)
        else:
            sigmoid = nn.Sigmoid()
            return sigmoid(-fx)

    def compute_loss_pu(self, px, ux, convex):
        fpx = self.model(px.type(dtype))
        fux = self.model(ux.type(dtype))
        p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
        n_loss = (torch.mean(self.basic_loss(-fux, convex))
                  - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
        if self.nn:
            loss = p_loss + n_loss if n_loss >= 0 else -n_loss/10
        else:
            loss = p_loss + n_loss
        return loss.cpu()

    def compute_loss_weighted(self, px, ux, convex):
        fpx = self.model(px.type(dtype))
        fux = self.model(ux.type(dtype))
        fux_prob = F.sigmoid(self.dr_model(ux.type(dtype)))
        pred_nprob = 1-torch.mean(fux_prob).item()
        print(pred_nprob)
        loss = (
            self.pi * torch.mean(self.basic_loss(fpx, True))
            + torch.mean(self.basic_loss(-fux, True) * (1-fux_prob))
            * (1-self.pi) / pred_nprob)
        return loss.cpu()

    def compute_loss_density(self, px, ux, convex):
        fpx = F.sigmoid(self.model(px.type(dtype)))
        fux = F.sigmoid(self.model(ux.type(dtype)))
        loss = torch.mean(fux**2)/2 - torch.mean(fpx)*self.pi
        if self.nn:
            loss = loss if loss > -self.pi/2 else -loss/10
        return loss.cpu()

    def test(self, test_set, to_print=True):
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x


model = Net().cuda() if args.cuda else Net()
cls = Classifier(model, pi=pi, lr=learning_rate, weight_decay=weight_decay,
                 loss_type=loss_type, nn=non_negative)
cls.train(p_set, u_set, test_set, p_batch_size, u_batch_size, 10)

print('')
model = Net().cuda() if args.cuda else Net()
cls2 = Classifier(model, pi=pi, dr_cls=cls)
cls2.train(p_set, u_set, test_set, p_batch_size, u_batch_size,
           training_epochs, dr_epochs=10)
