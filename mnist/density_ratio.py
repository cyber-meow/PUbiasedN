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


s_num = 1000
n_num = 500
u_num = 60000
sv_num = 100
nv_num = 50
uv_num = 3000

training_epochs = 100
s_batch_size = 150
u_batch_size = 6000
learning_rate = 5e-3
weight_decay = 1e-2

non_negative = False
M = 20/36
nn_threshold = -0.02
nn_rate = 1/10000
adjust_ux = False
adjust_rate = 1/2


params = OrderedDict([
    ('s_num', s_num),
    ('n_num', n_num),
    ('u_num', u_num),
    ('\nsv_num', sv_num),
    ('nv_num', nv_num),
    ('uv_num', uv_num),
    ('\ntraining_epochs', training_epochs),
    ('s_batch_size', s_batch_size),
    ('u_batch_size', u_batch_size),
    ('learning_rate', learning_rate),
    ('weight_decay', weight_decay),
    ('\nnon_negative', non_negative),
    ('nn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
    ('M', M),
    ('\nadjust_ux', adjust_ux),
    ('adjust_rate', adjust_rate),
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


train_labels = mnist.train_labels
s_idxs = np.argwhere(train_labels % 2 == 0).reshape(-1)
n_idxs = np.argwhere(train_labels % 2 == 1).reshape(-1)

selected_s = np.random.choice(s_idxs, s_num, replace=False)
selected_n = np.random.choice(n_idxs, n_num, replace=False)
selected = np.concatenate([selected_s, selected_n])

s_set = torch.utils.data.TensorDataset(
    mnist.train_data[selected].unsqueeze(1).float())

u_idxs = np.random.choice(60000, u_num, replace=False)

u_set = torch.utils.data.TensorDataset(
    mnist.train_data[u_idxs].unsqueeze(1).float())

# idxs = np.random.permutation(60000)
#
# s_set = torch.utils.data.TensorDataset(
#     mnist.train_data[idxs[:30000]].unsqueeze(1).float())
#
# u_set = torch.utils.data.TensorDataset(
#     mnist.train_data[idxs[30000:]].unsqueeze(1).float())

test_labels = mnist_test.test_labels
test_labels[test_labels % 2 == 1] = -1
test_labels[test_labels != -1] = 1

test_set = torch.utils.data.TensorDataset(
    mnist_test.test_data.unsqueeze(1).float(),
    test_labels.unsqueeze(1).float())

s_idxs = np.argwhere(test_labels == 1).reshape(-1)
n_idxs = np.argwhere(test_labels == -1).reshape(-1)
selected_s = np.random.choice(s_idxs, sv_num, replace=False)
selected_n = np.random.choice(n_idxs, nv_num, replace=False)
selected = np.concatenate([selected_s, selected_n])
s_validation = mnist_test.test_data[selected].unsqueeze(1).float()

u_idxs = np.random.choice(10000, uv_num, replace=False)
u_validation = mnist_test.test_data[u_idxs].unsqueeze(1).float()


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
        self.M = 1

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
            loss = compute_loss(x[0], ux)
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

    def compute_loss(self, sx, ux):
        fsx = self.model(sx.type(dtype))
        fux = self.model(ux.type(dtype))
        s_loss = self.pho * torch.mean(self.basic_loss(fsx))
        n_loss = (torch.mean(self.basic_loss(-fux))
                  - self.pho * torch.mean(self.basic_loss(-fsx)))
        if self.nn:
            if n_loss > self.nn_threshold:
                loss = s_loss + n_loss
            else:
                loss = -n_loss/10
        return loss.cpu()

    def compute_loss_density(self, sx, ux):
        fsx = self.model(sx.type(dtype))
        fux = self.model(ux.type(dtype))
        fux_mean = torch.mean(fux**2)
        fsx_mean = torch.mean(fsx)
        loss = fux_mean/2 - fsx_mean
        self.M = self.M * 0.9 + (fsx_mean+fux_mean).item()/4*0.1
        print(fux_mean.item()/2, fsx_mean.item()/2,
              torch.mean(fux).item(), loss.item())
        fsvx = self.model(s_validation.type(dtype))
        fuvx = self.model(u_validation.type(dtype))
        print('valid', (torch.mean(fuvx**2)/2 - torch.mean(fsvx)).item())
        # if self.times > 3:
        #     loss += (fux_mean - fsx_mean)**2/100
        if adjust_ux and torch.mean(fux) > 1 and self.times > 3:
            loss = torch.mean(fux) * adjust_rate
        if self.nn and loss + M < self.nn_threshold:
            loss = -loss * nn_rate
        self.times += 1
        # print(loss.item(), (loss+fux_mean/2).item(),
        #       (loss+fsx_mean/2).item(), loss.item()+self.M)
        # if self.nn and loss + self.M < self.nn_threshold:
        #     loss = -loss/10000
        return loss.cpu()

    def test(self, test_set, to_print=True):
        self.model.eval()
        x = test_set.tensors[0].type(dtype)
        target = test_set.tensors[1].type(dtype)
        output = self.model(x)
        fpx = output[target == 1]
        fnx = output[target == -1]
        print('pp', torch.mean(fpx).item(), torch.mean((fpx-4/3)**2).item())
        print('np', torch.mean(fnx).item(), torch.mean((fnx-2/3)**2).item())
        pred = torch.sign(output-1)
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
cls = Classifier(model, lr=learning_rate, weight_decay=weight_decay,
                 nn=non_negative, nn_threshold=nn_threshold)
cls.train(s_set, u_set, test_set, s_batch_size, u_batch_size, training_epochs)
