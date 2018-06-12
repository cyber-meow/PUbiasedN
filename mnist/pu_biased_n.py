from collections import OrderedDict

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


num_classes = 10

p_num = 500
sn_num = 500
u_num = 6000

pv_num = 100
snv_num = 100
uv_num = 1200

u_cut = 40000

pi = 0.49
# pi = 0.097
rho = 0.2

positive_classes = [0, 2, 4, 6, 8]

neg_ps = [0, 0.03, 0, 0.15, 0, 0.3, 0, 0.02, 0, 0.5]
# neg_ps = [0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2]
# neg_ps = [0, 1/3, 0, 1/3, 0, 1/3, 0, 0, 0, 0]
# neg_ps = [1/6, 1/6, 1/6, 1/6, 0, 1/6, 1/6, 0, 0, 0]

non_pu_fraction = 0.7
balanced = False

sep_value = 0.3
adjust_p = False
adjust_sn = True

cls_training_epochs = 100
convex_epochs = 100

p_batch_size = 100
sn_batch_size = 100
u_batch_size = 1200

learning_rate_cls = 1e-3
weight_decay = 1e-4
validation_momentum = 0.5

non_negative = True
nn_threshold = 0
nn_rate = 1/3

pu_prob_est = True
use_true_post = False

partial_n = True
hard_label = False

iwpn = False
pu = False
pnu = False

sets_save_name = None
# sets_load_name = 'pickle/mnist/1000_1000_10000/imbN/sets_imbN_a.p'
sets_load_name = None

ppe_save_name = None
# dre_load_name = ('pickle/mnist/1000_1000_10000/imbN/'
#                  + 'ls_prob_est_rho015_imbN_a.p')
ppe_load_name = None


params = OrderedDict([
    ('num_classes', num_classes),
    ('\np_num', p_num),
    ('sn_num', sn_num),
    ('u_num', u_num),
    ('\npv_num', pv_num),
    ('snv_num', snv_num),
    ('uv_num', uv_num),
    ('\nu_cut', u_cut),
    ('\npi', pi),
    ('rho', rho),
    ('\npositive_classes', positive_classes),
    ('neg_ps', neg_ps),
    ('\nnon_pu_fraction', non_pu_fraction),
    ('balanced', balanced),
    ('\nsep_value', sep_value),
    ('adjust_p', adjust_p),
    ('adjust_sn', adjust_sn),
    ('\ncls_training_epochs', cls_training_epochs),
    ('convex_epochs', convex_epochs),
    ('\np_batch_size', p_batch_size),
    ('sn_batch_size', sn_batch_size),
    ('u_batch_size', u_batch_size),
    ('\nlearning_rate_cls', learning_rate_cls),
    ('weight_decay', weight_decay),
    ('validation_momentum', validation_momentum),
    ('\nnon_negative', non_negative),
    ('nn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
    ('\npu_prob_est', pu_prob_est),
    ('use_true_post', use_true_post),
    ('\npartial_n', partial_n),
    ('hard_label', hard_label),
    ('\niwpn', iwpn),
    ('pu', pu),
    ('pnu', pnu),
    ('\nsets_save_name', sets_save_name),
    ('sets_load_name', sets_load_name),
    ('ppe_save_name', ppe_save_name),
    ('ppe_load_name', ppe_load_name),
])


# torchvision.datasets.MNIST outputs a set of PIL images
# We transform them to tensors
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load and transform data
mnist = torchvision.datasets.MNIST(
    './data/MNIST', train=True, download=True, transform=transform)

mnist_test = torchvision.datasets.MNIST(
    './data/MNIST', train=False, download=True, transform=transform)


train_data = torch.zeros(mnist.train_data.size())

for i, (image, _) in enumerate(mnist):
    train_data[i] = image

train_data = train_data.unsqueeze(1)
train_labels = mnist.train_labels

test_data = torch.zeros(mnist_test.test_data.size())

for i, (image, _) in enumerate(mnist_test):
    test_data[i] = image

test_data = test_data.unsqueeze(1)
test_labels = mnist_test.test_labels

# for i in range(10):
#     print(torch.sum(train_labels == i))


class Net(nn.Module):

    def __init__(self, num_classes=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, 1)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 10, 5, 1)
        self.bn2 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(4*4*10, 40)
        self.fc2 = nn.Linear(40, num_classes)

    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*10)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
