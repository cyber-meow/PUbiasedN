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


p_num = 2500
sn_num = 5000
u_num = 50000

pv_num = 500
snv_num = 1000
uv_num = 10000

pi = 0.49
pho = 0.1
sn_prob = 1/3

training_epochs = 100
p_batch_size = 250
sn_batch_size = 500
u_batch_size = 5000
n_batches = 10

learning_rate = 1e-3
weight_decay = 1e-2
validation_momentum = 0.5

non_negative = True
nn_threshold = -0.02
nn_rate = 1/10000

params = OrderedDict([
    ('p_num', p_num),
    ('sn_num', sn_num),
    ('u_num', u_num),
    ('\npv_num', pv_num),
    ('snv_num', snv_num),
    ('uv_num', uv_num),
    ('\npi', pi),
    ('pho', pho),
    ('sn_prob', sn_prob),
    ('\ntraining_epochs', training_epochs),
    ('p_batch_size', p_batch_size),
    ('sn_batch_size', p_batch_size),
    ('u_batch_size', u_batch_size),
    ('\nlearning_rate', learning_rate),
    ('weight_decay', weight_decay),
    ('validation_momentum', validation_momentum),
    ('\nnon_negative', non_negative),
    ('nn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
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
    '../git/datasets/MNIST', train=True, download=True, transform=transform)

mnist_test = torchvision.datasets.MNIST(
    '../git/datasets/MNIST', train=False, download=True, transform=transform)


def pick_p_data(data, labels, n):
    p_idxs = np.argwhere(labels % 2 == 0).reshape(-1)
    selected_p = np.random.choice(p_idxs, n, replace=False)
    return data[selected_p]


def pick_sn_data(data, labels, n):
    sn_idxs = np.argwhere(
        np.logical_and(labels % 2 == 1, labels < 6)).reshape(-1)
    selected_sn = np.random.choice(sn_idxs, n, replace=False)
    return data[selected_sn]


def pick_u_data(data, n):
    selected_u = np.random.choice(len(data), n, replace=False)
    return data[selected_u]


train_data = mnist.train_data
train_labels = mnist.train_labels
p_idxs = np.argwhere(train_labels % 2 == 0).reshape(-1)
selected_p = np.random.choice(p_idxs, p_num, replace=False)

p_set = torch.utils.data.TensorDataset(
    pick_p_data(train_data, train_labels, p_num).unsqueeze(1))

sn_set = torch.utils.data.TensorDataset(
    pick_sn_data(train_data, train_labels, sn_num).unsqueeze(1))

u_set = torch.utils.data.TensorDataset(
    pick_u_data(train_data, u_num).unsqueeze(1).float())


test_data = mnist_test.test_data
test_labels = mnist_test.test_labels

p_validation = pick_p_data(test_data, test_labels, pv_num).unsqueeze(1)
sn_validation = pick_sn_data(test_data, test_labels, snv_num).unsqueeze(1)
u_validation = pick_u_data(test_data, uv_num).unsqueeze(1)

test_posteriors = torch.zeros(test_labels.size())
test_posteriors[test_labels % 2 == 0] = 1
# test_posteriors[test_labels % 2 == 1] = -1
test_posteriors[test_labels == 1] = sn_prob
test_posteriors[test_labels == 3] = sn_prob
test_posteriors[test_labels == 5] = sn_prob
test_posteriors[test_labels == 7] = 0
test_posteriors[test_labels == 9] = 0

test_set = torch.utils.data.TensorDataset(
    mnist_test.test_data.unsqueeze(1).float(),
    test_posteriors.unsqueeze(1).float())


class PosterirorProbability(object):

    def __init__(self, model, pi, pho, lr=5e-3,
                 weight_decay=1e-2, nn=True, nn_threshold=0):
        self.model = model
        self.pi = pi
        self.pho = pho
        self.lr = lr
        self.weight_decay = weight_decay
        self.nn = nn
        self.nn_threshold = nn_threshold
        self.test_accuracies = []
        self.init_optimizer()
        self.times = 0
        self.min_vloss = float('inf')
        self.curr_accu_vloss = None
        self.final_model = None

    def init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay)

    def train(self, p_set, sn_set, u_set, test_set,
              p_batch_size, sn_bath_size, u_batch_size,
              num_epochs, test_interval=1, print_interval=1):

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

            # convex = True if epoch < 4 else False
            total_loss = self.train_step(p_loader, sn_loader, u_loader)

            if (epoch+1) % test_interval == 0 or epoch+1 == num_epochs:

                to_print = (epoch+1) % print_interval == 0
                if to_print:
                    sys.stdout.write('Epoch: {}  '.format(epoch))
                    print('Train Loss: {:.6f}'.format(total_loss/n_batches))
                self.test(test_set, to_print)

        self.model = self.final_model
        print('Final error:')
        self.test(test_set, True)

    def train_step(self, p_loader, sn_loader, u_loader):
        self.model.train()
        total_loss = 0
        compute_loss = self.compute_loss_density
        for x in p_loader:
            self.optimizer.zero_grad()
            snx = next(iter(sn_loader))[0]
            ux = next(iter(u_loader))[0]
            loss, true_loss = compute_loss(x[0], snx, ux)
            total_loss += true_loss.item()
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

    def compute_loss_pu(self, px, snx, ux, convex=False):
        fpx = self.model(px.type(dtype))
        fsnx = self.model(snx.type(dtype))
        fux = self.model(ux.type(dtype))
        p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
        sn_loss = self.pho * torch.mean(self.basic_loss(fsnx, convex))
        n_loss = (torch.mean(self.basic_loss(-fux, convex))
                  - self.pho * torch.mean(self.basic_loss(-fsnx, convex))
                  - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
        print('n_loss: ', n_loss.item())
        loss = p_loss + sn_loss + n_loss if n_loss > 0 else -n_loss/100
        return loss.cpu(), loss.cpu()

    def compute_loss_density(self, px, snx, ux):
        fpx = self.model(px.type(dtype))
        fsnx = self.model(snx.type(dtype))
        fux = self.model(ux.type(dtype))
        fpx_mean = torch.mean(fpx)
        fsnx_mean = torch.mean(fsnx)
        fux_mean = torch.mean(fux)
        fux2_mean = torch.mean(fux**2)
        true_loss = fux2_mean - 2*fpx_mean*self.pi - 2*fsnx_mean*self.pho
        loss = true_loss
        print(fux2_mean.item(),
              fpx_mean.item()*self.pi + fsnx_mean.item()*self.pho,
              fux_mean.item(), true_loss.item())
        self.validation()
        if self.nn and loss + self.pi + self.pho < self.nn_threshold:
            loss = -loss * nn_rate
        return loss.cpu(), true_loss.cpu()

    def validation(self):
        fpvx = self.model(p_validation.type(dtype))
        fsnvx = self.model(sn_validation.type(dtype))
        fuvx = self.model(u_validation.type(dtype))
        validation_loss = (torch.mean(fuvx**2)
                           - 2*torch.mean(fpvx) * self.pi
                           - 2*torch.mean(fsnvx) * self.pho).item()
        print('valid', validation_loss)
        if self.curr_accu_vloss is None:
            self.curr_accu_vloss = validation_loss
        else:
            self.curr_accu_vloss = (
                self.curr_accu_vloss * validation_momentum
                + validation_loss * (1-validation_momentum))
        if self.curr_accu_vloss < self.min_vloss:
            self.min_vloss = self.curr_accu_vloss
            self.final_model = deepcopy(self.model)
        return validation_loss

    def test(self, test_set, to_print=True):
        self.model.eval()
        x = test_set.tensors[0].type(dtype)
        target = test_set.tensors[1].type(dtype)
        target_n = target/torch.mean(target)*(self.pi+self.pho)
        output = self.model(x)
        output = output/torch.mean(output)*(self.pi+self.pho)
        print('average',
              torch.mean(output[target == 1]).item(),
              torch.mean(output[target == sn_prob]).item(),
              torch.mean(output[target == 0]).item())
        error = torch.mean((target_n-output)**2).item()
        if to_print:
            print('Test set: Error: {}'.format(error))

    def test_pu(self, test_set, to_print=True):
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
        x = F.sigmoid(self.fc2(x))
        return x


model = Net().cuda() if args.cuda else Net()
cls = PosterirorProbability(
        model, pi, pho, lr=learning_rate, weight_decay=weight_decay,
        nn=non_negative, nn_threshold=nn_threshold)
cls.train(p_set, sn_set, u_set, test_set,
          p_batch_size, sn_batch_size, u_batch_size, training_epochs)
