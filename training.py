import sys
import numpy as np
from copy import deepcopy

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

import settings


class Training(object):

    def __init__(self, model,
                 lr=5e-3, weight_decay=1e-2,
                 validation_momentum=0.5):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation_momentum = validation_momentum
        self.min_vloss = float('inf')
        self.curr_accu_vloss = None
        self.final_model = None
        self.test_accuracies = []

    def init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay)

    def validation(self, *args):
        self.model.eval()
        _, validation_loss = self.compute_loss(*args)
        print('Validation Loss:', validation_loss.item())
        if self.curr_accu_vloss is None:
            self.curr_accu_vloss = validation_loss
        else:
            self.curr_accu_vloss = (
                self.curr_accu_vloss * self.validation_momentum
                + validation_loss * (1-self.validation_momentum))
        if self.curr_accu_vloss < self.min_vloss:
            self.min_vloss = self.curr_accu_vloss
            self.final_model = deepcopy(self.model)
        return validation_loss

    def train(self, *args):
        raise NotImplementedError

    def test(self, *args):
        raise NotImplementedError

    def compute_loss(self, *args):
        raise NotImplementedError


class Classifier(Training):

    def test(self, test_set, to_print=True):
        self.model.eval()
        x = test_set.tensors[0].type(settings.dtype)
        target = test_set.tensors[1].type(settings.dtype)
        output = self.model(x)
        pred = torch.sign(output)
        correct = torch.sum(pred.eq(target).float()).item()
        accuracy = 100 * correct/len(test_set)
        self.test_accuracies.append(accuracy)
        if to_print:
            print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                    correct, len(test_set), accuracy))

    def basic_loss(self, fx, convex):
        if convex:
            negative_logistic = nn.LogSigmoid()
            return -negative_logistic(fx)
        else:
            sigmoid = nn.Sigmoid()
            return sigmoid(-fx)


class Classifier_from2(Classifier):

    def __init__(self, model, pi=0.5, *args, **kwargs):
        self.pi = pi
        super().__init__(model, *args, **kwargs)

    def train(self, p_set, n_set, test_set,
              p_batch_size, n_batch_size,
              p_validation, n_validation,
              num_epochs, convex_epochs=5,
              test_interval=1, print_interval=1):

        self.init_optimizer()
        self.test(test_set, True)

        p_loader = torch.utils.data.DataLoader(
            p_set, batch_size=p_batch_size,
            shuffle=True, num_workers=1)

        n_loader = torch.utils.data.DataLoader(
            n_set, batch_size=n_batch_size,
            shuffle=True, num_workers=1)

        for epoch in range(num_epochs):

            convex = True if epoch < convex_epochs else False
            average_loss = self.train_step(
                p_loader, n_loader,
                p_validation, n_validation, convex)

            if (epoch+1) % test_interval == 0 or epoch+1 == num_epochs:

                to_print = (epoch+1) % print_interval == 0
                if to_print:
                    sys.stdout.write('Epoch: {}  '.format(epoch))
                    print('Train Loss: {:.6f}'.format(average_loss))
                self.test(test_set, to_print)

        self.model = self.final_model
        print('Final error:')
        self.test(test_set, True)

    def train_step(self, p_loader, n_loader,
                   p_validation, n_validation, convex):
        self.model.train()
        losses = []
        for x in p_loader:
            self.optimizer.zero_grad()
            nx = next(iter(n_loader))[0]
            loss, true_loss = self.compute_loss(x[0], nx, convex)
            losses.append(true_loss.item())
            loss.backward()
            self.optimizer.step()
        self.validation(p_validation, n_validation, convex)
        return np.mean(np.array(losses))


class PNClassifier(Classifier_from2):

    def __init__(self, model, pu_model=None, *args, **kwargs):
        if pu_model is not None:
            self.pu_model = pu_model
            self.test = self.test_two_stage
        super().__init__(model, *args, **kwargs)

    def compute_loss(self, px, nx, convex):
        fpx = self.model(px.type(settings.dtype))
        fnx = self.model(nx.type(settings.dtype))
        p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
        n_loss = (1-self.pi) * torch.mean(self.basic_loss(-fnx, convex))
        loss = p_loss + n_loss
        return loss.cpu(), loss.cpu()

    def test_two_stage(self, test_set, to_print=True):
        self.pu_model.eval()
        self.model.eval()
        x = test_set.tensors[0].type(settings.dtype)
        target = test_set.tensors[1].type(settings.dtype)
        output1 = self.pu_model(x)
        pred1 = torch.sign(output1)
        output2 = self.model(x)
        pred = torch.sign(output2)
        pred[pred1 == -1] = -1
        correct = torch.sum(pred.eq(target).float()).item()
        accuracy = 100 * correct/len(test_set)
        self.test_accuracies.append(accuracy)
        if to_print:
            print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                    correct, len(test_set), accuracy))


class PUClassifier(Classifier_from2):

    def __init__(self, model, nn=True, nn_threshold=0, nn_rate=1/100,
                 *args, **kwargs):
        self.nn_rate = nn_rate
        self.nn_threshold = nn_threshold
        self.nn = nn
        super().__init__(model, *args, **kwargs)

    def compute_loss(self, px, ux, convex):
        fpx = self.model(px.type(settings.dtype))
        fux = self.model(ux.type(settings.dtype))
        p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
        n_loss = (torch.mean(self.basic_loss(-fux, convex))
                  - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
        true_loss = p_loss + n_loss
        loss = true_loss
        print(n_loss.item())
        if self.nn and n_loss < self.nn_threshold:
            loss = -n_loss * self.nn_rate
        return loss.cpu(), true_loss.cpu()


class Classifier_from3(Classifier):

    def __init__(self, model, pi=0.5, rho=0.1, *args, **kwargs):
        self.pi = pi
        self.rho = rho
        super().__init__(model, *args, **kwargs)

    def train(self, p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              num_epochs, convex_epochs=5,
              test_interval=1, print_interval=1):

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

            convex = True if epoch < convex_epochs else False
            total_loss = self.train_step(
                p_loader, sn_loader, u_loader,
                p_validation, sn_validation, u_validation, convex)

            if (epoch+1) % test_interval == 0 or epoch+1 == num_epochs:

                to_print = (epoch+1) % print_interval == 0
                if to_print:
                    sys.stdout.write('Epoch: {}  '.format(epoch))
                    print('Train Loss: {:.6f}'.format(total_loss))
                self.test(test_set, to_print)

        self.model = self.final_model
        print('Final error:')
        self.test(test_set, True)

    def train_step(self, p_loader, sn_loader, u_loader,
                   p_validation, sn_validation, u_validation, convex):
        self.model.train()
        losses = []
        for x in p_loader:
            self.optimizer.zero_grad()
            snx = next(iter(sn_loader))[0]
            ux = next(iter(u_loader))[0]
            loss, true_loss = self.compute_loss(x[0], snx, ux, convex)
            losses.append(true_loss.item())
            loss.backward()
            self.optimizer.step()
        self.validation(p_validation, sn_validation, u_validation, convex)
        return np.mean(np.array(losses))


class PUClassifier3(Classifier_from3):

    def __init__(self, model, nn=True, nn_threshold=0, nn_rate=1/100,
                 *args, **kwargs):
        self.nn_rate = nn_rate
        self.nn_threshold = nn_threshold
        self.nn = nn
        super().__init__(model, *args, **kwargs)

    def compute_loss(self, px, snx, ux, convex):
        fpx = self.model(px.type(settings.dtype))
        fsnx = self.model(snx.type(settings.dtype))
        fux = self.model(ux.type(settings.dtype))
        p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
        sn_loss = self.rho * torch.mean(self.basic_loss(fsnx, convex))
        n_loss = (torch.mean(self.basic_loss(-fux, convex))
                  - self.rho * torch.mean(self.basic_loss(-fsnx, convex))
                  - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
        true_loss = p_loss + sn_loss + n_loss
        loss = true_loss
        if self.nn and n_loss < self.nn_threshold:
            loss = -n_loss * self.nn_rate
        return loss.cpu(), true_loss.cpu()


class WeightedClassifier(Classifier_from3):

    def __init__(self, model, pp_model, *args, **kwargs):
        self.pp_model = pp_model
        super().__init__(model, *args, **kwargs)

    def compute_loss(self, px, snx, ux, convex):
        fpx = self.model(px.type(settings.dtype))
        fsnx = self.model(snx.type(settings.dtype))
        fux = self.model(ux.type(settings.dtype))
        fux_prob = self.pp_model(ux.type(settings.dtype))
        fux_prob = fux_prob/torch.mean(fux_prob)*(self.pi+self.rho)
        loss = (
            self.pi * torch.mean(self.basic_loss(fpx, convex))
            + self.rho * torch.mean(self.basic_loss(-fsnx, convex))
            + torch.mean(self.basic_loss(-fux, convex) * (1-fux_prob)))
        return loss.cpu(), loss.cpu()


class PosteriorProbability(Training):

    def __init__(self, model, pi=0.5, rho=0.1, *args, **kwargs):
        self.pi = pi
        self.rho = rho
        super().__init__(model, *args, **kwargs)

    def train(self, p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
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

            total_loss = self.train_step(
                p_loader, sn_loader, u_loader,
                p_validation, sn_validation, u_validation)

            if (epoch+1) % test_interval == 0 or epoch+1 == num_epochs:

                to_print = (epoch+1) % print_interval == 0
                if to_print:
                    sys.stdout.write('Epoch: {}  '.format(epoch))
                    print('Train Loss: {:.6f}'.format(total_loss))
                self.test(test_set, to_print)

        self.model = self.final_model
        print('Final error:')
        self.test(test_set, True)

    def train_step(self, p_loader, sn_loader, u_loader,
                   p_validation, sn_validation, u_validation):
        self.model.train()
        losses = []
        for x in p_loader:
            self.optimizer.zero_grad()
            snx = next(iter(sn_loader))[0]
            ux = next(iter(u_loader))[0]
            loss, true_loss = self.compute_loss(x[0], snx, ux)
            losses.append(true_loss.item())
            loss.backward()
            self.optimizer.step()
            self.validation(p_validation, sn_validation, u_validation)
        return np.mean(np.array(losses))

    def compute_loss(self, px, snx, ux):
        fpx = self.model(px.type(settings.dtype))
        fsnx = self.model(snx.type(settings.dtype))
        fux = self.model(ux.type(settings.dtype))
        fpx_mean = torch.mean(fpx)
        fsnx_mean = torch.mean(fsnx)
        fux_mean = torch.mean(fux)
        fux2_mean = torch.mean(fux**2)
        loss = fux2_mean - 2*fpx_mean*self.pi - 2*fsnx_mean*self.rho
        print(fux2_mean.item(),
              fpx_mean.item()*self.pi + fsnx_mean.item()*self.rho,
              fux_mean.item(), loss.item())
        return loss.cpu(), loss.cpu()

    def test(self, test_set, to_print=True):
        self.model.eval()
        x = test_set.tensors[0].type(settings.dtype)
        target = test_set.tensors[1]
        target = target/torch.mean(target)*(self.pi+self.rho)
        output = self.model(x).cpu()
        output = output/torch.mean(output)*(self.pi+self.rho)
        error = torch.mean((target-output)**2).item()
        if to_print:
            print('Test set: Error: {}'.format(error))
