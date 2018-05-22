import sys
import argparse
import numpy as np
from copy import deepcopy

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


pn_num = 1600
u_num = 60000
pi = 0.49
pho = 0.8

training_epochs = 200
pn_batch_size = 160
u_batch_size = 6000


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
    './data', train=True, download=True, transform=transform)

mnist_test = torchvision.datasets.MNIST(
    './data', train=False, download=True, transform=transform)


train_data = mnist.train_data.numpy()
train_labels = mnist.train_labels.numpy()
# pn_idxs = np.argwhere(train_labels % 2 == 0).reshape(-1)
pn_idxs = np.argwhere(
    np.logical_or(train_labels % 2 == 0, train_labels < 6)).reshape(-1)
selected_pn = np.random.choice(pn_idxs, pn_num, replace=False)

u_idxs = np.random.choice(60000, u_num, replace=False)

pn_data = train_data[selected_pn]
pn_labels = train_labels[selected_pn]
pn_labels[pn_labels % 2 == 1] = -1
pn_labels[pn_labels % 2 == 0] = 1

pn_set = torch.utils.data.TensorDataset(
    torch.from_numpy(pn_data).unsqueeze(1).float(),
    torch.from_numpy(pn_labels).unsqueeze(1).float())

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
print(torch.sum(test_labels))

test_set2 = torch.utils.data.TensorDataset(
    mnist_test.test_data.unsqueeze(1).float(),
    test_labels.unsqueeze(1).float())


class Classifier(object):

    def __init__(self, model, pi=0.49, pho=0.8,
                 lr=5e-3, weight_decay=1e-2, dr_model=None):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.dr_model = dr_model

        self.pi = pi
        self.pho = pho
        self.test_accuracies = []
        self.init_optimizer()

        self.times = 0

    def init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay)

    def train(self, pn_set, u_set, test_set,
              pn_batch_size, u_batch_size, num_epochs,
              test_interval=1, print_interval=1):

        self.init_optimizer()
        self.test(test_set, True)

        pn_loader = torch.utils.data.DataLoader(
            pn_set, batch_size=pn_batch_size,
            shuffle=True, num_workers=2)

        u_loader = torch.utils.data.DataLoader(
            u_set, batch_size=u_batch_size,
            shuffle=True, num_workers=2)

        for epoch in range(num_epochs):

            total_loss = self.train_step(pn_loader, u_loader)

            if (epoch+1) % test_interval == 0 or epoch+1 == num_epochs:

                to_print = (epoch+1) % print_interval == 0
                if to_print:
                    sys.stdout.write('Epoch: {}  '.format(epoch))
                    print('Train Loss: {:.6f}'.format(total_loss))
                self.test(test_set, to_print)

    def train_step(self, pn_loader, u_loader):
        self.model.train()
        total_loss = 0
        compute_loss = (self.compute_loss_density
                        if self.dr_model is None
                        else self.compute_loss_weighted)
        for (x, target) in pn_loader:
            self.optimizer.zero_grad()
            ux = next(iter(u_loader))[0]
            loss = compute_loss(x, target, ux)
            total_loss += loss.item()
            loss = loss
            loss.backward()
            self.optimizer.step()
        return total_loss

    def basic_loss(self, fx):
        if self.times < 2:
            negative_logistic = nn.LogSigmoid()
            return -negative_logistic(fx)
        else:
            sigmoid = nn.Sigmoid()
            return sigmoid(-fx)

    def compute_loss(self, pnx, target, ux):
        fpnx = self.model(pnx.type(dtype))
        fux = self.model(ux.type(dtype))
        # fpx = fpnx[target == 1]
        # print(torch.mean(F.sigmoid(fpnx)).item(),
        #       torch.mean(F.sigmoid(fux)).item())
        pn_loss = self.pho * torch.mean(self.basic_loss(fpnx))
        n_loss = (torch.mean(self.basic_loss(-fux))
                  - self.pho * torch.mean(self.basic_loss(-fpnx)))
        loss = pn_loss + n_loss if n_loss > 0 else -n_loss
        return loss.cpu()

    def compute_loss_weighted(self, pnx, target, ux):
        fpnx = self.model(pnx.type(dtype))
        fux = self.model(ux.type(dtype))
        fux_prob = F.sigmoid(self.dr_model(ux.type(dtype)))
        loss = (
            self.pho * torch.mean(self.basic_loss(fpnx * target.type(dtype)))
            + torch.mean(self.basic_loss(-fux) * (1-fux_prob)))
        return loss.cpu()

    def compute_loss_density(self, pnx, target, ux):
        fpnx = F.sigmoid(self.model(pnx.type(dtype)))
        fux = F.sigmoid(self.model(ux.type(dtype)))
        # print(torch.mean(F.sigmoid(fpnx)).item(),
        #       torch.mean(F.sigmoid(fux)).item())
        loss = torch.mean(fux**2)/2 - torch.mean(fpnx)*self.pho
        loss = loss if loss > -self.pho/2 else -loss/10
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
cls = Classifier(model, pi=pi, pho=pho)
cls.train(pn_set, u_set, test_set2, pn_batch_size, u_batch_size, 50)

print('')
model = Net().cuda() if args.cuda else Net()
cls2 = Classifier(model, pi=pi, dr_model=cls.model)
cls2.train(pn_set, u_set, test_set, pn_batch_size, u_batch_size, 100)
