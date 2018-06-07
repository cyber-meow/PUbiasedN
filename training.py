import sys
import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_auc_score

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
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
        self.auc_scores = []

    def init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

    def validation(self, *args):
        _, validation_loss = self.compute_loss(*args, validation=True)
        print('Validation Loss:', validation_loss.item(), flush=True)
        if self.curr_accu_vloss is None:
            self.curr_accu_vloss = validation_loss.item()
        else:
            self.curr_accu_vloss = (
                self.curr_accu_vloss * self.validation_momentum
                + validation_loss.item() * (1-self.validation_momentum))
        if self.curr_accu_vloss < self.min_vloss:
            self.min_vloss = self.curr_accu_vloss
            self.final_model = deepcopy(self.model)
        return validation_loss

    @staticmethod
    def feed_together(model, *args):
        split_sizes = []
        for x in args:
            split_sizes.append(x.size(0))
        x_cat = torch.cat(args).type(settings.dtype)
        fx = model(x_cat)
        return torch.split(fx, split_sizes)

    @staticmethod
    def feed_in_batches(model, x, batch_size=None):
        if batch_size is None:
            batch_size = settings.test_batch_size
        model.eval()
        if len(x) <= batch_size:
            with torch.no_grad():
                fx = model(x.type(settings.dtype))
            return fx
        fxs = []
        for i in range(0, x.size(0), batch_size):
            x_batch = x[i: min(i+batch_size, x.size(0))]
            with torch.no_grad():
                fxs.append(model(x_batch.type(settings.dtype)))
        return torch.cat(fxs)[:x.size(0)]

    def train(self, *args):
        raise NotImplementedError

    def test(self, *args):
        raise NotImplementedError

    def compute_loss(self, *args):
        raise NotImplementedError


class Classifier(Training):

    def test(self, test_set, to_print=True):
        x = test_set.tensors[0]
        target = test_set.tensors[1].type(settings.dtype)
        output = self.feed_in_batches(self.model, x, settings.test_batch_size)
        pred = torch.sign(output)
        correct = torch.sum(pred.eq(target).float()).item()
        accuracy = 100 * correct/len(test_set)
        self.test_accuracies.append(accuracy)
        if to_print:
            print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                    correct, len(test_set), accuracy), flush=True)
        target = test_set.tensors[1].numpy().reshape(-1)
        output = output.cpu().numpy().reshape(-1)
        auc_score = roc_auc_score(target, output) * 100
        self.auc_scores.append(auc_score)
        if to_print:
            print('Test set: Auc Score: {:.2f}%'.format(auc_score),
                  flush=True)

    def basic_loss(self, fx, convex):
        if convex:
            negative_logistic = nn.LogSigmoid()
            return -negative_logistic(fx)
        else:
            sigmoid = nn.Sigmoid()
            return sigmoid(-fx)


class ClassifierFrom2(Classifier):

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
            shuffle=True, num_workers=0)

        n_loader = torch.utils.data.DataLoader(
            n_set, batch_size=n_batch_size,
            shuffle=True, num_workers=0)

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
        losses = []
        # for i in range(int(len(p_set)/p_batch_size)):
        for i, x in enumerate(p_loader):
            self.model.train()
            self.optimizer.zero_grad()
            # x = p_set.tensors[0][np.random.choice(len(p_set), p_batch_size)]
            # nx = n_set.tensors[0][np.random.choice(len(n_set), n_batch_size)]
            nx = next(iter(n_loader))
            loss, true_loss = self.compute_loss(x, nx, convex)
            losses.append(true_loss.item())
            loss.backward()
            self.optimizer.step()
            if (i+1) % settings.validation_interval == 0:
                self.validation(p_validation, n_validation, convex)
        self.optimizer.zero_grad()
        return np.mean(np.array(losses))


class PNClassifier(ClassifierFrom2):

    def __init__(self, model, pu_model=None, pp_model=None,
                 adjust_p=False, adjust_n=False, *args, **kwargs):
        if pu_model is not None:
            self.pu_model = pu_model
            self.test = self.test_two_stage
        self.pp_model = pp_model
        self.adjust_p = adjust_p
        self.adjust_n = adjust_n
        super().__init__(model, *args, **kwargs)

    def compute_loss(self, px, nx, convex, validation=False):
        px, nx = px[0], nx[0]
        if validation:
            fpx = self.feed_in_batches(self.model, px)
            fnx = self.feed_in_batches(self.model, nx)
            convex = False
        else:
            fpx, fnx = self.feed_together(self.model, px, nx)
        if self.adjust_p:
            fpx_prob = self.feed_in_batches(self.pp_model, px)
            p_loss = self.pi * torch.mean(
                self.basic_loss(fpx, convex) / fpx_prob)
        else:
            p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
        if self.adjust_n:
            fnx_prob = self.feed_in_batches(self.pp_model, nx)
            n_loss = (1-self.pi) * torch.mean(
                self.basic_loss(-fnx, convex) / fnx_prob)
        else:
            n_loss = (1-self.pi) * torch.mean(self.basic_loss(-fnx, convex))
        loss = p_loss + n_loss
        return loss.cpu(), loss.cpu()

    def test_two_stage(self, test_set, to_print=True):
        x = test_set.tensors[0]
        target = test_set.tensors[1].type(settings.dtype)
        output1 = self.feed_in_batches(
            self.pu_model, x, settings.test_batch_size)
        pred1 = torch.sign(output1)
        output2 = self.feed_in_batches(
            self.model, x, settings.test_batch_size)
        pred = torch.sign(output2)
        pred[pred1 == -1] = -1
        correct = torch.sum(pred.eq(target).float()).item()
        accuracy = 100 * correct/len(test_set)
        self.test_accuracies.append(accuracy)
        if to_print:
            print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                    correct, len(test_set), accuracy), flush=True)


class PUClassifier(ClassifierFrom2):

    def __init__(self, model,
                 nn=True, nn_threshold=0, nn_rate=1/100, prob_est=False,
                 *args, **kwargs):
        self.nn_rate = nn_rate
        self.nn_threshold = nn_threshold
        self.nn = nn
        if prob_est:
            self.test = self.test_prob_est
        super().__init__(model, *args, **kwargs)

    def compute_loss(self, px, ux, convex, validation=False):
        px, ux = px[0], ux[0]
        if validation:
            fpx = self.feed_in_batches(self.model, px)
            fux = self.feed_in_batches(self.model, ux)
            convex = False
        else:
            fpx, fux = self.feed_together(self.model, px, ux)
        p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
        n_loss = (torch.mean(self.basic_loss(-fux, convex))
                  - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
        true_loss = p_loss + n_loss
        loss = true_loss
        if not validation:
            print(n_loss.item())
        if self.nn and n_loss < self.nn_threshold:
            loss = -n_loss * self.nn_rate
        return loss.cpu(), true_loss.cpu()

    def test_prob_est(self, test_set, to_print=True):
        x = test_set.tensors[0]
        target = test_set.tensors[1]
        target = target/torch.mean(target)*self.pi
        output = F.sigmoid(self.feed_in_batches(
            self.model, x, settings.test_batch_size)).cpu()
        output = output/torch.mean(output)*self.pi
        error = torch.mean((target-output)**2).item()
        if to_print:
            print('Test set: Error: {}'.format(error), flush=True)


class ClassifierFrom3(Classifier):

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
            shuffle=True, num_workers=0)

        sn_loader = torch.utils.data.DataLoader(
            sn_set, batch_size=sn_batch_size,
            shuffle=True, num_workers=0)

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
        losses = []
        for i, x in enumerate(p_loader):
            self.model.train()
            self.optimizer.zero_grad()
            snx = next(iter(sn_loader))
            ux = next(iter(u_loader))
            loss, true_loss = self.compute_loss(x, snx, ux, convex)
            losses.append(true_loss.item())
            loss.backward()
            self.optimizer.step()
            if (i+1) % settings.validation_interval == 0:
                self.validation(
                    p_validation, sn_validation, u_validation, convex)
        return np.mean(np.array(losses))


class PUClassifier3(ClassifierFrom3):

    def __init__(self, model,
                 nn=True, nn_threshold=0, nn_rate=1/2,
                 pre_model=None, prob_est=False, *args, **kwargs):
        self.nn_rate = nn_rate
        self.nn_threshold = nn_threshold
        self.nn = nn
        self.prob_est = prob_est
        self.pre_model = pre_model
        if prob_est:
            self.test = self.test_prob_est
            self.validation = self.validation_prob_est
        super().__init__(model, *args, **kwargs)

    def compute_loss(self, px, snx, ux, convex):
        px, snx, ux = px[0], snx[0], ux[0]
        if self.pre_model is not None:
            px, snx, ux = self.feed_together(self.pre_model, px, snx, ux)
        fpx, fsnx, fux = self.feed_together(self.model, px, snx, ux)
        p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
        sn_loss = self.rho * torch.mean(self.basic_loss(fsnx, convex))
        n_loss = (torch.mean(self.basic_loss(-fux, convex))
                  - self.rho * torch.mean(self.basic_loss(-fsnx, convex))
                  - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
        true_loss = p_loss + sn_loss + n_loss
        loss = true_loss
        print(n_loss.item())
        if self.nn and n_loss < self.nn_threshold:
            loss = -n_loss * self.nn_rate
        return loss.cpu(), true_loss.cpu()

    def average_loss(self, fx):
        negative_logistic = nn.LogSigmoid()
        logistic_loss = torch.mean(-negative_logistic(fx))
        sigmoid = nn.Sigmoid()
        sigmoid_loss = torch.mean(sigmoid(-fx))
        return torch.tensor([logistic_loss, sigmoid_loss])

    def validation_prob_est(self, p_val, sn_val, u_val, convex):
        p_val, sn_val, u_val = p_val[0], sn_val[0], u_val[0]
        if self.pre_model is not None:
            p_val = self.feed_in_batches(self.pre_model, p_val)
            sn_val = self.feed_in_batches(self.pre_model, sn_val)
            u_val = self.feed_in_batches(self.pre_model, u_val)
        fpx = self.feed_in_batches(self.model, p_val)
        fsnx = self.feed_in_batches(self.model, sn_val)
        fux = self.feed_in_batches(self.model, u_val)
        ls_loss = (torch.mean(F.sigmoid(fux)**2)
                   - 2 * torch.mean(F.sigmoid(fpx)) * self.pi
                   - 2 * torch.mean(F.sigmoid(fsnx)) * self.rho)
        print('Validation Ls Loss:', ls_loss.cpu().item(), flush=True)
        p_loss = self.pi * self.average_loss(fpx)
        sn_loss = self.rho * self.average_loss(fsnx)
        n_loss = (self.average_loss(-fux)
                  - self.rho * self.average_loss(-fsnx)
                  - self.pi * self.average_loss(-fpx))
        logistic_loss = p_loss[0] + sn_loss[0] + n_loss[0]
        print('Validation Log Loss:', logistic_loss.cpu().item(), flush=True)
        sigmoid_loss = p_loss[1] + sn_loss[1] + n_loss[1]
        print('Validation Sig Loss:', sigmoid_loss.cpu().item(), flush=True)
        if self.curr_accu_vloss is None:
            self.curr_accu_vloss = ls_loss.cpu().item()
        else:
            self.curr_accu_vloss = (
                self.curr_accu_vloss * self.validation_momentum
                + ls_loss.cpu().item() * (1-self.validation_momentum))
        if self.curr_accu_vloss < self.min_vloss:
            self.min_vloss = self.curr_accu_vloss
            self.final_model = deepcopy(self.model)
        return ls_loss

    def test_prob_est(self, test_set, to_print=True):
        x = test_set.tensors[0]
        if self.pre_model is not None:
            x = self.feed_in_batches(self.pre_model, x)
        target = test_set.tensors[1]
        output = self.feed_in_batches(
                self.model, x, settings.test_batch_size).cpu()
        output_prob = F.sigmoid(output)
        error = torch.mean((target-output_prob)**2).item()
        error_std = torch.std((target-output_prob)**2).item()
        if to_print:
            print('Test set: Error: {}'.format(error), flush=True)
            print('Test set: Error Std: {}'.format(error_std), flush=True)
        target_normalized = target/torch.mean(target)
        output_prob = output_prob/torch.mean(output_prob)
        error = torch.mean((target_normalized-output_prob)**2).item()
        error_std = torch.std((target_normalized-output_prob)**2).item()
        if to_print:
            print('Test set: Normalized Error: {}'.format(error))
            print('Test set: Normalized Error Std: {}'.format(error_std))
        pred = torch.sign(output)
        target_pred = torch.ones_like(target)
        target_pred[target < 1/2] = -1
        correct = torch.sum(pred.eq(target_pred).float()).item()
        accuracy = 100 * correct/len(test_set)
        self.test_accuracies.append(accuracy)
        if to_print:
            print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                    correct, len(test_set), accuracy), flush=True)
        target_pred = target_pred.numpy().reshape(-1)
        output = output.cpu().numpy().reshape(-1)
        auc_score = roc_auc_score(target_pred, output) * 100
        self.auc_scores.append(auc_score)
        if to_print:
            print('Test set: Auc Score: {:.2f}%'.format(auc_score))
        x = test_set.tensors[0]


class PUClassifierPlusN(ClassifierFrom3):

    def __init__(self, model,
                 nn=True, nn_threshold=0, nn_rate=1/2, minus_n=True,
                 *args, **kwargs):
        self.nn_rate = nn_rate
        self.nn_threshold = nn_threshold
        self.nn = nn
        self.minus_n = minus_n
        super().__init__(model, *args, **kwargs)

    def train_step(self, p_loader, sn_loader, u_loader,
                   p_validation, sn_validation, u_validation, convex):
        losses = []
        for i, x in enumerate(p_loader):
            self.model.train()
            self.optimizer.zero_grad()
            snx = next(iter(sn_loader))[0]
            snx2 = next(iter(sn_loader))[0]
            ux = next(iter(u_loader))[0]
            loss, true_loss = self.compute_loss(x[0], snx, snx2, ux, convex)
            losses.append(true_loss.item())
            loss.backward()
            self.optimizer.step()
            if (i+1) % settings.validation_interval == 0:
                self.validation(
                    p_validation, sn_validation, sn_validation,
                    u_validation, convex)
        return np.mean(np.array(losses))

    def compute_loss(self, px, snx, snx2, ux, convex, validation=False):
        if validation:
            fpx = self.feed_in_batches(self.model, px)
            fsnx = self.feed_in_batches(self.model, snx)
            fsnx2 = self.feed_in_batches(self.model, snx2)
            fux = self.feed_in_batches(self.model, ux)
            convex = False
        else:
            fpx, fsnx, fsnx2, fux = self.feed_together(
                self.model, px, snx, snx2, ux)
        p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
        sn_loss = self.rho * torch.mean(self.basic_loss(-fsnx, convex))
        n_loss = (torch.mean(self.basic_loss(-fux, convex))
                  - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
        if self.minus_n:
            n_loss -= self.rho * torch.mean(self.basic_loss(-fsnx2, convex))
        if not validation:
            print(n_loss.item())
        true_loss = p_loss + sn_loss + n_loss
        loss = true_loss
        if self.nn and n_loss < self.nn_threshold:
            loss = -n_loss * self.nn_rate
        return loss.cpu(), true_loss.cpu()


class PNUClassifier(ClassifierFrom3):

    def __init__(self, model, pn_fraction=0.5,
                 nn=True, nn_threshold=0, nn_rate=1/2,
                 *args, **kwargs):
        self.nn_rate = nn_rate
        self.nn_threshold = nn_threshold
        self.nn = nn
        self.pn_fraction = pn_fraction
        super().__init__(model, *args, **kwargs)

    def compute_loss(self, px, nx, ux, convex, validation=False):
        px, nx, ux = px[0], nx[0], ux[0]
        if validation:
            fpx = self.feed_in_batches(self.model, px)
            fnx = self.feed_in_batches(self.model, nx)
            fux = self.feed_in_batches(self.model, ux)
            convex = False
        else:
            fpx, fnx, fux = self.feed_together(self.model, px, nx, ux)
        p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
        n_loss = (1-self.pi) * torch.mean(self.basic_loss(-fnx, convex))
        n_loss2 = (torch.mean(self.basic_loss(-fux, convex))
                   - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
        true_loss = (
            p_loss + n_loss * self.pn_fraction
            + n_loss * (1-self.pn_fraction))
        loss = true_loss
        if self.nn and n_loss2 < self.nn_threshold:
            loss = -n_loss2 * self.nn_rate
        return loss.cpu(), true_loss.cpu()


class WeightedClassifier(ClassifierFrom3):

    def __init__(self, model, pp_model, sep_value=0.3,
                 adjust_p=True, adjust_sn=True, *args, **kwargs):
        self.pp_model = pp_model
        if pp_model is not None:
            for param in self.pp_model.parameters():
                param.requires_grad = False
        self.times = 0
        self.sep_value = sep_value
        self.adjust_p = adjust_p
        self.adjust_sn = adjust_sn
        super().__init__(model, *args, **kwargs)

    def compute_loss(self, px, snx, ux, convex, validation=False):

        if validation:
            fpx = self.feed_in_batches(self.model, px[0])
            fux = self.feed_in_batches(self.model, ux[0])
            fsnx = self.feed_in_batches(self.model, snx[0])
            convex = False
        else:
            fpx, fsnx, fux = self.feed_together(
                self.model, px[0], snx[0], ux[0])

        # Divide into two parts according to the value of p(s=1|x)
        if self.pp_model is None:
            fux_prob = ux[1].type(settings.dtype)
        else:
            fux_prob = F.sigmoid(self.feed_in_batches(self.pp_model, ux[0]))
        fux_prob[fux_prob > self.sep_value] = 1

        if self.adjust_p:
            if self.pp_model is None:
                fpx_prob = px[1].type(settings.dtype)
            else:
                fpx_prob = F.sigmoid(
                    self.feed_in_batches(self.pp_model, px[0]))
            fpx_prob[fpx_prob <= self.sep_value] = 1
            p_loss = self.pi * torch.mean(
                self.basic_loss(fpx, convex)
                + self.basic_loss(-fpx, convex) * (1-fpx_prob)/fpx_prob)
        else:
            p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))

        if self.adjust_sn:
            if self.pp_model is None:
                fsnx_prob = snx[1].type(settings.dtype)
            else:
                fsnx_prob = F.sigmoid(
                    self.feed_in_batches(self.pp_model, snx[0]))
            fsnx_prob[fsnx_prob <= self.sep_value] = 1
            sn_loss = self.rho * torch.mean(
                self.basic_loss(-fsnx, convex) / fsnx_prob)
        else:
            sn_loss = self.rho * torch.mean(self.basic_loss(-fsnx, convex))

        loss = (
            p_loss + sn_loss
            + torch.mean(self.basic_loss(-fux, convex) * (1-fux_prob)))
        return loss.cpu(), loss.cpu()

    def test(self, test_set, to_print=True):
        x = test_set.tensors[0]
        target = test_set.tensors[1].type(settings.dtype)
        output = self.feed_in_batches(self.model, x, settings.test_batch_size)
        pred = torch.sign(output)
        correct = torch.sum(pred.eq(target).float()).item()
        accuracy = 100 * correct/len(test_set)
        self.test_accuracies.append(accuracy)
        if to_print:
            print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                    correct, len(test_set), accuracy), flush=True)
        target = test_set.tensors[1].numpy().reshape(-1)
        output = output.cpu().numpy().reshape(-1)
        auc_score = roc_auc_score(target, output) * 100
        self.auc_scores.append(auc_score)
        if to_print:
            print('Test set: Auc Score: {:.2f}%'.format(auc_score),
                  flush=True)


class PosteriorProbability(Training):

    def __init__(self, model, pi=0.5, rho=0.1, beta=1, *args, **kwargs):
        self.pi = pi
        self.rho = rho
        self.beta = beta
        super().__init__(model, *args, **kwargs)

    def train(self, p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              num_epochs, test_interval=1, print_interval=1):

        self.init_optimizer()
        self.test(test_set, True)

        p_loader = torch.utils.data.DataLoader(
            p_set, batch_size=p_batch_size,
            shuffle=True, num_workers=0)

        sn_loader = torch.utils.data.DataLoader(
            sn_set, batch_size=sn_batch_size,
            shuffle=True, num_workers=0)

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
        losses = []
        for i, x in enumerate(p_loader):
            self.model.train()
            self.optimizer.zero_grad()
            snx = next(iter(sn_loader))[0]
            ux = next(iter(u_loader))[0]
            loss, true_loss = self.compute_loss(x[0], snx, ux)
            losses.append(true_loss.item())
            loss.backward()
            self.optimizer.step()
            if (i+1) % settings.validation_interval == 0:
                self.validation(p_validation, sn_validation, u_validation)
        return np.mean(np.array(losses))

    def compute_loss(self, px, snx, ux):
        px, snx, ux = px[0], snx[0], ux[0]
        fpx, fsnx, fux = self.feed_together(self.model, px, snx, ux)
        fpx_mean = torch.mean(F.sigmoid(fpx))
        fsnx_mean = torch.mean(F.sigmoid(fsnx))
        fux_mean = torch.mean(F.sigmoid(fux))
        fux2_mean = torch.mean(F.sigmoid(fux)**2)
        loss = (fux2_mean - 2*fpx_mean*self.pi - 2*fsnx_mean*self.rho
                + self.beta * (fux_mean - self.pi - self.rho)**2)
        print(fux2_mean.item(),
              fpx_mean.item()*self.pi + fsnx_mean.item()*self.rho,
              fux_mean.item(), loss.item())
        return loss.cpu(), loss.cpu()

    def average_loss(self, fx):
        negative_logistic = nn.LogSigmoid()
        logistic_loss = torch.mean(-negative_logistic(fx))
        sigmoid = nn.Sigmoid()
        sigmoid_loss = torch.mean(sigmoid(-fx))
        return torch.tensor([logistic_loss, sigmoid_loss])

    def validation(self, p_val, sn_val, u_val):
        fpx = self.feed_in_batches(self.model, p_val[0])
        fsnx = self.feed_in_batches(self.model, sn_val[0])
        fux = self.feed_in_batches(self.model, u_val[0])
        ls_loss = (torch.mean(F.sigmoid(fux)**2)
                   - 2 * torch.mean(F.sigmoid(fpx)) * self.pi
                   - 2 * torch.mean(F.sigmoid(fsnx)) * self.rho)
        print('Validation Ls Loss:', ls_loss.cpu().item(), flush=True)
        p_loss = self.pi * self.average_loss(fpx)
        sn_loss = self.rho * self.average_loss(fsnx)
        n_loss = (self.average_loss(-fux)
                  - self.rho * self.average_loss(-fsnx)
                  - self.pi * self.average_loss(-fpx))
        logistic_loss = p_loss[0] + sn_loss[0] + n_loss[0]
        print('Validation Log Loss:', logistic_loss.cpu().item(), flush=True)
        sigmoid_loss = p_loss[1] + sn_loss[1] + n_loss[1]
        print('Validation Sig Loss:', sigmoid_loss.cpu().item(), flush=True)
        if self.curr_accu_vloss is None:
            self.curr_accu_vloss = ls_loss.cpu().item()
        else:
            self.curr_accu_vloss = (
                self.curr_accu_vloss * self.validation_momentum
                + ls_loss.cpu().item() * (1-self.validation_momentum))
        if self.curr_accu_vloss < self.min_vloss:
            self.min_vloss = self.curr_accu_vloss
            self.final_model = deepcopy(self.model)
        return ls_loss

    def test(self, test_set, to_print=True):
        x = test_set.tensors[0]
        target = test_set.tensors[1]
        output = F.sigmoid(self.feed_in_batches(
                self.model, x, settings.test_batch_size).cpu())
        error = torch.mean((target-output)**2).item()
        error_std = torch.std((target-output)**2).item()
        if to_print:
            print('Test set: Error: {}'.format(error), flush=True)
            print('Test set: Error Std: {}'.format(error_std), flush=True)
        target_norm = target/torch.mean(target)
        output_norm = output/torch.mean(output)
        error = torch.mean((target_norm-output_norm)**2).item()
        error_std = torch.std((target_norm-output_norm)**2).item()
        if to_print:
            print('Test set: Normalized Error: {}'.format(error))
            print('Test set: Normalized Error Std: {}'.format(error_std))
        pred = torch.sign(output-1/2)
        target_pred = torch.ones_like(target)
        target_pred[target < 1/2] = -1
        correct = torch.sum(pred.eq(target_pred).float()).item()
        accuracy = 100 * correct/len(test_set)
        self.test_accuracies.append(accuracy)
        if to_print:
            print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                    correct, len(test_set), accuracy), flush=True)
        target_pred = target_pred.numpy().reshape(-1)
        output = output.cpu().numpy().reshape(-1)
        auc_score = roc_auc_score(target_pred, output) * 100
        self.auc_scores.append(auc_score)
        if to_print:
            print('Test set: Auc Score: {:.2f}%'.format(auc_score))


class PosteriorProbability2(Training):

    def __init__(self, model, pi=0.5, *args, **kwargs):
        self.pi = pi
        super().__init__(model, *args, **kwargs)

    def train(self, p_set, u_set, test_set,
              p_batch_size, u_batch_size, p_validation, u_validation,
              num_epochs, test_interval=1, print_interval=1):

        self.init_optimizer()
        self.test(test_set, True)

        p_loader = torch.utils.data.DataLoader(
            p_set, batch_size=p_batch_size,
            shuffle=True, num_workers=0)

        u_loader = torch.utils.data.DataLoader(
            u_set, batch_size=u_batch_size,
            shuffle=True, num_workers=1)

        for epoch in range(num_epochs):

            total_loss = self.train_step(
                p_loader, u_loader, p_validation, u_validation)

            if (epoch+1) % test_interval == 0 or epoch+1 == num_epochs:

                to_print = (epoch+1) % print_interval == 0
                if to_print:
                    sys.stdout.write('Epoch: {}  '.format(epoch))
                    print('Train Loss: {:.6f}'.format(total_loss))
                self.test(test_set, to_print)

        self.model = self.final_model
        print('Final error:')
        self.test(test_set, True)

    def train_step(self, p_loader, u_loader, p_validation, u_validation):
        losses = []
        for i, x in enumerate(p_loader):
            self.model.train()
            self.optimizer.zero_grad()
            ux = next(iter(u_loader))[0]
            loss, true_loss = self.compute_loss(x[0], ux)
            losses.append(true_loss.item())
            loss.backward()
            self.optimizer.step()
            if (i+1) % settings.validation_interval == 0:
                self.validation(p_validation, u_validation)
        return np.mean(np.array(losses))

    def compute_loss(self, px, ux, validation=False):
        if validation:
            fpx = self.feed_in_batches(self.model, px)
            fux = self.feed_in_batches(self.model, ux)
        else:
            fpx, fux = self.feed_together(self.model, px, ux)
        fpx_mean = torch.mean(fpx)
        fux_mean = torch.mean(fux)
        fux2_mean = torch.mean(fux**2)
        loss = fux2_mean - 2*fpx_mean*self.pi + (fux_mean-self.pi)**2
        print(fux2_mean.item(), fpx_mean.item() * self.pi,
              fux_mean.item(), loss.item())
        return loss.cpu(), loss.cpu()

    def test(self, test_set, to_print=True):
        x = test_set.tensors[0]
        target = test_set.tensors[1]
        target = target/torch.mean(target)*self.pi
        output = self.feed_in_batches(
            self.model, x, settings.test_batch_size).cpu()
        output = output/torch.mean(output)*self.pi
        error = torch.mean((target-output)**2).item()
        if to_print:
            print('Test set: Error: {}'.format(error), flush=True)
