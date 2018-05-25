import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


training_epochs = 5
batch_size = 150
learning_rate = 5e-3
weight_decay = 0


params = OrderedDict([
    ('training_epochs', training_epochs),
    ('batch_size', batch_size),
    ('learning_rate', learning_rate),
    ('weight_decay', weight_decay),
])

for key, value in params.items():
    print('{}: {}'.format(key, value))
print('')


parser = argparse.ArgumentParser(description='MNIST perfect classifier')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    dtype = torch.cuda.FloatTensor


# torchvision.datasets.MNIST outputs a set of PIL images
# We transform them to tensors
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

# Load and transform data
mnist = torchvision.datasets.MNIST(
    './data/MNIST', train=True, download=True, transform=transform)

mnist_test = torchvision.datasets.MNIST(
    './data/MNIST', train=False, download=True, transform=transform)


train_data = mnist.train_data
train_labels = mnist.train_labels
t_labels = torch.zeros_like(train_labels)

t_labels[train_labels % 2 == 1] = -1
t_labels[train_labels % 2 == 0] = 1

training_set = torch.utils.data.TensorDataset(
    train_data.unsqueeze(1).float(), t_labels.unsqueeze(1).float())


test_data = mnist_test.test_data
test_labels = mnist_test.test_labels

test_labels[test_labels % 2 == 1] = -1
test_labels[test_labels != -1] = 1

test_set = torch.utils.data.TensorDataset(
    test_data.unsqueeze(1), test_labels.unsqueeze(1))


class Classifier(object):

    def __init__(self, model, lr=5e-3, weight_decay=1e-2):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.test_accuracies = []
        self.init_optimizer()

    def init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay)

    def train(self, training_set, test_set,
              batch_size, num_epochs,
              test_interval=1, print_interval=1):

        self.init_optimizer()
        self.test(test_set, True)

        train_loader = torch.utils.data.DataLoader(
            training_set, batch_size=batch_size,
            shuffle=True, num_workers=2)

        for epoch in range(num_epochs):

            total_loss = self.train_step(train_loader)

            if (epoch+1) % test_interval == 0 or epoch+1 == num_epochs:

                to_print = (epoch+1) % print_interval == 0
                if to_print:
                    sys.stdout.write('Epoch: {}  '.format(epoch))
                    print('Train Loss: {:.6f}'.format(total_loss))
                self.test(test_set, to_print)

    def train_step(self, train_loader, convex=True):
        self.model.train()
        total_loss = 0
        for x, target in train_loader:
            self.optimizer.zero_grad()
            loss = self.compute_loss(x, target, convex)
            total_loss += loss.item()
            loss = loss
            loss.backward()
            self.optimizer.step()
        return total_loss

    def basic_loss(self, fx, convex=True):
        if convex:
            negative_logistic = nn.LogSigmoid()
            return -negative_logistic(fx)
        else:
            sigmoid = nn.Sigmoid()
            return sigmoid(-fx)

    def compute_loss(self, x, target, convex=True):
        fx = self.model(x.type(dtype))
        target = target.type(dtype)
        loss = torch.sum(self.basic_loss(fx * target, convex))
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


# model = Net().cuda() if args.cuda else Net()
# cls = Classifier(model, lr=learning_rate, weight_decay=weight_decay)
# cls.train(training_set, test_set, batch_size, training_epochs)
# pickle.dump(cls, open('perfect_cls.p', 'wb'))

cls = pickle.load(open('perfect_cls.p', 'rb'))


n_samples = train_data[t_labels == -1].unsqueeze(1).type(dtype)
n_labels = train_labels[t_labels == -1].numpy()
num_n = len(n_samples)
print(num_n)
positive_scores = cls.model(n_samples).detach().cpu().numpy().reshape(-1)
# positive_scores = 1/(1+np.exp(-positive_scores))

print(len(positive_scores))
positive_orders = np.argsort(positive_scores)
print(positive_scores[positive_orders][-20:])
print(n_labels[positive_orders][-20:])

if False:
    one_scores = positive_scores[n_labels == 1]
    three_scores = positive_scores[n_labels == 3]
    five_scores = positive_scores[n_labels == 5]
    seven_scores = positive_scores[n_labels == 7]
    nine_scores = positive_scores[n_labels == 9]

    plt.figure()
    plt.title('positive_scores')
    plt.hist(
        [one_scores, three_scores, five_scores, seven_scores, nine_scores],
        label=['1', '3', '5', '7', '9'], log=True)
    plt.legend()


positions = np.ones(num_n)
for pos, k in enumerate(positive_orders):
    positions[k] = pos

if False:
    one_positions = positions[n_labels == 1]
    three_positions = positions[n_labels == 3]
    five_positions = positions[n_labels == 5]
    seven_positions = positions[n_labels == 7]
    nine_positions = positions[n_labels == 9]

    plt.figure()
    plt.title('positive_positions')
    plt.hist(
        [one_positions, three_positions, five_positions,
            seven_positions, nine_positions],
        label=['1', '3', '5', '7', '9'], log=True)
    plt.legend()
    plt.show()


probs = np.exp(-positions/1e3)/np.sum(np.exp(-positions/1e3))

pickle.dump(positions, open('prob_ac_pos.p', 'wb'))

selected_ind = np.random.choice(positions, 1000, replace=False, p=probs)
# print(selected_ind)
plt.hist(selected_ind)
plt.show()
# print(np.sum(selected_ind == 1))
# print(np.sum(selected_ind == 3))
# print(np.sum(selected_ind == 5))
# print(np.sum(selected_ind == 7))
# print(np.sum(selected_ind == 9))
