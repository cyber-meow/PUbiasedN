import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import settings


class LinearModel(nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

    def plot_boundary(self, ax, **kwargs):
        params = list(self.parameters())
        w1 = params[0][0][0].cpu().detach().numpy()
        w2 = params[0][0][1].cpu().detach().numpy()
        b = params[1][0].cpu().detach().numpy()
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        x = np.linspace(xmin, xmax, 1000)
        y = -b/w2 - w1/w2*x
        x = x[np.logical_and(y > ymin, y < ymax)]
        y = y[np.logical_and(y > ymin, y < ymax)]
        ax.plot(x, y, **kwargs)


class Net(nn.Module):

    def __init__(self, sigmoid_output=False):
        super(Net, self).__init__()
        self.sigmoid_output = sigmoid_output
        self.linear1 = nn.Linear(2, 10)
        self.linear2 = nn.Linear(10, 5)
        self.linear3 = nn.Linear(5, 1)
        # self.linear4 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        if self.sigmoid_output:
            x = F.sigmoid(self.linear3(x))
        else:
            x = self.linear3(x)
        return x

    def border_func(self, x, y):
        inp = torch.tensor([x, y]).type(settings.dtype)
        return self.forward(inp).detach().cpu().numpy()

    def plot_boundary(self, ax, **kwargs):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xs = np.linspace(xmin, xmax, 100)
        ys = np.linspace(ymin, ymax, 100)
        xv, yv = np.meshgrid(xs, ys)
        border_func = np.vectorize(self.border_func)
        cont = ax.contour(xv, yv, border_func(xv, yv), **kwargs)
        return cont
