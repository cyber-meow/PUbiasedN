import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import settings
import training


p_num = 1000
sn_num = 10000
u_num = 10000

pv_num = 100
snv_num = 1000
uv_num = 1000

u_cut = 50000

pi = 0.49
rho = 0.3
sn_prob = 1

training_epochs = 50

p_batch_size = 100
sn_batch_size = 500
u_batch_size = 3000
n_batches = 10

learning_rate = 1e-4
weight_decay = 1e-4
validation_momentum = 0.5

non_negative = False
nn_threshold = 0
nn_rate = 1/3

params = OrderedDict([
    ('p_num', p_num),
    ('sn_num', sn_num),
    ('u_num', u_num),
    ('\npv_num', pv_num),
    ('snv_num', snv_num),
    ('uv_num', uv_num),
    ('\npi', pi),
    ('rho', rho),
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


parser = argparse.ArgumentParser(description='MNIST postitive posteiror')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

settings.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor


# torchvision.datasets.MNIST outputs a set of PIL images
# We transform them to tensors
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

# Load and transform data
mnist = torchvision.datasets.MNIST(
    './data/MNIST', train=True, download=True, transform=transform)

mnist_test = torchvision.datasets.MNIST(
    './data/MNIST', train=False, download=True, transform=transform)


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


train_data = mnist.train_data.float()
train_labels = mnist.train_labels.float()
idxs = np.random.permutation(len(train_data))

valid_data = train_data[idxs][u_cut:]
valid_labels = train_labels[idxs][u_cut:]
train_data = train_data[idxs][:u_cut]
train_labels = train_labels[idxs][:u_cut]


p_set = torch.utils.data.TensorDataset(
    pick_p_data(train_data, train_labels, p_num).unsqueeze(1))

sn_set = torch.utils.data.TensorDataset(
    pick_sn_data(train_data, train_labels, sn_num).unsqueeze(1))

u_set = torch.utils.data.TensorDataset(
    pick_u_data(train_data, u_num).unsqueeze(1))

p_validation = pick_p_data(valid_data, valid_labels, pv_num).unsqueeze(1)
sn_validation = pick_sn_data(valid_data, valid_labels, snv_num).unsqueeze(1)
u_validation = pick_u_data(valid_data, uv_num).unsqueeze(1)


test_data = mnist_test.test_data.float()
test_labels = mnist_test.test_labels.float()

test_posteriors = torch.zeros(test_labels.size())
test_posteriors[test_labels % 2 == 0] = 1
test_posteriors[test_labels == 1] = sn_prob
test_posteriors[test_labels == 3] = sn_prob
test_posteriors[test_labels == 5] = sn_prob
test_posteriors[test_labels == 7] = 0
test_posteriors[test_labels == 9] = 0

test_set = torch.utils.data.TensorDataset(
    test_data.unsqueeze(1), test_posteriors.unsqueeze(1))


class Net(nn.Module):

    def __init__(self, sigmoid_output=False):
        super(Net, self).__init__()
        self.sigmoid_output = sigmoid_output
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
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


model = Net(True).cuda() if args.cuda else Net(True)
cls = training.PosteriorProbability(
        model, pi=pi, rho=rho,
        lr=learning_rate, weight_decay=weight_decay)
cls.train(p_set, sn_set, u_set, test_set,
          p_batch_size, sn_batch_size, u_batch_size,
          p_validation, sn_validation, u_validation, training_epochs)
# cls = training.PosteriorProbability2(
#         model, pi=rho,
#         lr=learning_rate, weight_decay=weight_decay)
# cls.train(sn_set, u_set, test_set, sn_batch_size, u_batch_size,
#           sn_validation, u_validation, training_epochs)

# model = Net().cuda() if args.cuda else Net()
# cls = training.PUClassifier(
#         model, pi=rho, prob_est=True,
#         lr=learning_rate, weight_decay=weight_decay)
# cls.train(sn_set, u_set, test_set, sn_batch_size, u_batch_size,
#           sn_validation, u_validation, training_epochs)

cls.model.eval()
predict_probs = cls.model(test_data.unsqueeze(1).type(settings.dtype)).cpu()
predict_probs_norm = (1-predict_probs)/(1-torch.mean(predict_probs))*(1-rho-pi)
predict_probs_norm2 = 1-predict_probs/torch.mean(predict_probs)*(rho+pi)

for k in range(10):
    k_predict_probs = predict_probs[test_labels == k]
    k_predict_probs_norm = predict_probs_norm[test_labels == k]
    k_predict_probs_norm2 = predict_probs_norm2[test_labels == k]
    print(k, torch.mean(k_predict_probs).item(),
          torch.mean(k_predict_probs_norm).item(),
          torch.mean(k_predict_probs_norm2).item())
    plt.figure()
    plt.title(k)
    plt.hist(k_predict_probs.detach().numpy().reshape(-1))
    plt.savefig('{}'.format(k))
plt.show()
