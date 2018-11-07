"""Adapt VAT loss to binary output"""

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def _kl_div_with_logit(logits_p, logits_m):
    kl_p = torch.sigmoid(logits_p)*(
            F.logsigmoid(logits_p)-F.logsigmoid(logits_m))
    kl_q = torch.sigmoid(-logits_p)*(
            F.logsigmoid(-logits_p)-F.logsigmoid(-logits_m))
    return torch.mean(kl_p)+torch.mean(kl_q)


def entropy_with_logit(logits):
    ent_p = F.logsigmoid(logits)*torch.sigmoid(logits)
    ent_q = F.logsigmoid(-logits)*torch.sigmoid(-logits)
    return -torch.mean(ent_p)-torch.mean(ent_q)


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        # why??
        with torch.no_grad():
            # shape [batch_size, 1]
            logits_p = model(x)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                logits_m = model(x + self.xi * d)
                adv_distance = _kl_div_with_logit(logits_p, logits_m)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            logits_m = model(x + r_adv)
            lds = _kl_div_with_logit(logits_p, logits_m)

        return lds
