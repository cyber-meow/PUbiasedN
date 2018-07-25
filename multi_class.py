import argparse

import torch
import torch.utils.data

import training
import settings

# from cifar10.pu_biased_n import params, Net
# from cifar10.pu_biased_n import train_data, test_data
# from cifar10.pu_biased_n import train_labels, test_labels

# from mnist.pu_biased_n import params, Net
# from mnist.pu_biased_n import train_data, test_data
# from mnist.pu_biased_n import train_labels, test_labels

# from uci.pu_biased_n import params, Net
# from uci.pu_biased_n import train_data, test_data
# from uci.pu_biased_n import train_labels, test_labels

from newsgroups.pu_biased_n import params, Net
from newsgroups.pu_biased_n import train_data, test_data
from newsgroups.pu_biased_n import train_labels, test_labels


training_epochs = 100
batch_size = 240
learning_rate = 1e-3
weight_decay = 1e-4

num_classes = params['num_classes']

validation_momentum = params['validation_momentum']
start_validation_epoch = params.get('\nstart_validation_epoch', 0)

milestones = params.get('milestones', [1000])
lr_d = params.get('lr_d', 1)


parser = argparse.ArgumentParser(description='Main File')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.random_seed is not None:
    params['\nrandom_seed'] = args.random_seed
    random_seed = args.random_seed


settings.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

training_set = torch.utils.data.TensorDataset(train_data, train_labels)
test_set = torch.utils.data.TensorDataset(test_data, test_labels)

torch.manual_seed(random_seed)

print('')

model = Net(num_classes=num_classes)
if args.cuda:
    model = model.cuda()

cls = training.MultiClassClassifier(
        model, lr=learning_rate, weight_decay=weight_decay,
        milestones=milestones, lr_d=lr_d,
        validation_momentum=validation_momentum,
        start_validation_epoch=start_validation_epoch)

cls.train(training_set, test_set, batch_size, training_epochs)
