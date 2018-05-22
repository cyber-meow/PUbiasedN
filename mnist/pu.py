import sys
import argparse
import numpy as np
from scipy.spatial import distance
from collections import OrderedDict

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


p_num = 2000
n_num = 1000
u_num = 60000
pi = 0.49

training_epochs = 50
p_batch_size = 300
u_batch_size = 6000
learning_rate = 5e-3
weight_decay = 1e-2

non_negative = False
nn_threshold = 0
loss_type = 'density'


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
    ('nn_threshold', nn_threshold),
    ('loss_type', loss_type),
])

for key, value in params.items():
    print('{}: {}'.format(key, value))
print('')


parser = argparse.ArgumentParser(description='PU MNIST')

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


train_data = mnist.train_data.numpy()
train_labels = mnist.train_labels.numpy()
p_idxs = np.argwhere(train_labels % 2 == 0).reshape(-1)
n_idxs = np.argwhere(train_labels % 2 == 1).reshape(-1)
# p_idxs = np.argwhere(train_labels < 8).reshape(-1)
# p_idxs = np.argwhere(
#     np.logical_or(train_labels % 2 == 0, train_labels < 6)).reshape(-1)
selected_p = np.random.choice(p_idxs, p_num, replace=False)
selected_n = np.random.choice(n_idxs, n_num, replace=False)
selected = np.concatenate([selected_p, selected_n])

p_set = torch.utils.data.TensorDataset(
    torch.from_numpy(train_data[selected]).unsqueeze(1).float())

u_idxs = np.random.choice(60000, u_num, replace=False)

u_set = torch.utils.data.TensorDataset(
    mnist.train_data[u_idxs].unsqueeze(1).float())

test_labels = mnist_test.test_labels
test_labels[test_labels % 2 == 1] = -1
test_labels[test_labels != -1] = 1
# test_labels[test_labels != 1] = -1
# test_labels[test_labels > 7] = -1
# test_labels[test_labels < 6] = 1

test_set = torch.utils.data.TensorDataset(
    mnist_test.test_data.unsqueeze(1).float(),
    test_labels.unsqueeze(1).float())


class Classifier(object):

    def __init__(self, model, pi=0.49, lr=5e-3, weight_decay=1e-2,
                 nn=True, nn_threshold=-0.01,
                 loss_type='density', dr_cls=None):
        self.model = model
        self.pi = pi
        self.lr = lr
        self.weight_decay = weight_decay
        self.nn = nn
        self.nn_threshold = nn_threshold
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
        self.M = self.pi/2

    def init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay)

    def train(self, p_set, u_set, test_set,
              p_batch_size, u_batch_size, num_epochs,
              test_interval=1, print_interval=1):

        self.init_optimizer()
        self.test(test_set, True)

        p_loader = torch.utils.data.DataLoader(
            p_set, batch_size=p_batch_size,
            shuffle=True, num_workers=1)

        u_loader = torch.utils.data.DataLoader(
            u_set, batch_size=u_batch_size,
            shuffle=True, num_workers=1)

        for epoch in range(num_epochs):

            convex = True if epoch < 5 else False
            total_loss = self.train_step(p_loader, u_loader, convex)

            if (epoch+1) % test_interval == 0 or epoch+1 == num_epochs:

                to_print = (epoch+1) % print_interval == 0
                if to_print:
                    sys.stdout.write('Epoch: {}  '.format(epoch))
                    print('Train Loss: {:.6f}'.format(total_loss))
                self.test(test_set, to_print)

    def train_step(self, p_loader, u_loader, convex):
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

    def basic_loss(self, fx, convex):
        if convex:
            negative_logistic = nn.LogSigmoid()
            return -negative_logistic(fx)
        else:
            sigmoid = nn.Sigmoid()
            return sigmoid(-fx)

    def compute_loss_pu(self, px, ux, convex):
        fpx = self.model(px.type(dtype))
        fux = self.model(ux.type(dtype))
        # print(torch.mean(F.sigmoid(fpx)).item(),
        #       torch.mean(F.sigmoid(fux)).item())
        p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
        n_loss = (torch.mean(self.basic_loss(-fux, convex))
                  - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
        if self.nn:
            loss = p_loss + n_loss if n_loss > 0 else -n_loss/10
        else:
            loss = p_loss + n_loss
        return loss.cpu()

    def compute_loss_weighted(self, px, ux, convex):
        fpx = self.model(px.type(dtype))
        fux = self.model(ux.type(dtype))
        fux_prob = F.sigmoid(self.dr_model(ux.type(dtype)))
        loss = (
            self.pi * torch.mean(self.basic_loss(fpx, convex))
            + torch.mean(self.basic_loss(-fux, convex) * (1-fux_prob)))
        return loss.cpu()

    def compute_loss_MMD(self, px, ux):
        fux = F.sigmoid(self.model(ux.type(dtype))).cpu()
        fux_flat = fux.view(-1).cpu()
        px = px.numpy().reshape(len(px), -1)
        ux = ux.numpy().reshape(len(ux), -1)
        pp = self.pi**2 * torch.mean(self.kernel(px, px))
        pu = self.pi * torch.mean(self.kernel(px, ux) * fux_flat)
        uu = torch.mean(self.kernel(ux, ux) * fux * fux_flat)
        return torch.sqrt(pp - 2*pu + uu)

    def kernel(self, A, B, sigma=5e8):
        diss = distance.cdist(A, B, 'euclidean')
        return torch.from_numpy(np.exp(-diss**2/(2*sigma))).float()

    def compute_loss_density(self, px, ux, convex):
        fpx = F.sigmoid(self.model(px.type(dtype)))
        fux = F.sigmoid(self.model(ux.type(dtype)))
        fpx_mean = torch.mean(fpx)*self.pi
        loss = torch.mean(fux**2)/2 - fpx_mean
        self.M = self.M * 0.9 + fpx_mean.item()/2*0.1
        # self.M = 5/36
        print(loss.item(), self.M, loss.item()+self.M)
        # loss += (torch.mean(fux) - self.pi)**2
        if self.nn and loss + self.M < self.nn_threshold:
            loss = -loss/10000
        return loss.cpu()

    def test(self, test_set, to_print=True):
        self.model.eval()
        x = test_set.tensors[0].type(dtype)
        target = test_set.tensors[1].type(dtype)
        output = self.model(x)
        fpx = F.sigmoid(output[target == 1])
        fnx = F.sigmoid(output[target == -1])
        print('pp', torch.mean(fpx).item(), torch.mean((fpx-2/3)**2).item())
        print('np', torch.mean(fnx).item(), torch.mean((fnx-1/3)**2).item())
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
cls = Classifier(model, pi=pi, lr=learning_rate,
                 weight_decay=weight_decay, loss_type=loss_type,
                 nn=non_negative, nn_threshold=nn_threshold)
cls.train(p_set, u_set, test_set, p_batch_size, u_batch_size, training_epochs)

# print('')
# model = Net().cuda() if args.cuda else Net()
# cls2 = Classifier(model, pi=pi, dr_model=cls.model)
# cls2.train(p_set, u_set, test_set, p_batch_size, u_batch_size, 100)
