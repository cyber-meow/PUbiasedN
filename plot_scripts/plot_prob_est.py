import pickle
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from training import Training
import settings

sets_load_name = 'pickle/mnist/1000_1000_10000/imbN/sets_imbN_a.p'
dre_load_name = ('pickle/mnist/1000_1000_10000/imbN/'
                 + 'ls_prob_est_rho015_imbN_a.p')


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


settings.dtype = torch.cuda.FloatTensor

p_set, sn_set, n_set, u_set,\
    p_validation, sn_validation, n_validation, u_validation =\
    pickle.load(open(sets_load_name, 'rb'))

dre_model = pickle.load(open(dre_load_name, 'rb'))

fux = Training(None).feed_in_batches(dre_model, p_set.tensors[0])
fux_prob = F.sigmoid(fux)
pickle.dump(fux_prob, open('prob', 'wb'))
# plt.hist(fux_prob.cpu().numpy())
# plt.show()
