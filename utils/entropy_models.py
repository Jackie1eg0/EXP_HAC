"""
Entropy models for rate estimation during training.
Ported from HAC-plus official implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from utils.encodings import use_clamp


class Entropy_gaussian(nn.Module):
    """Single Gaussian entropy model."""
    def __init__(self, Q=1):
        super(Entropy_gaussian, self).__init__()
        self.Q = Q

    def forward(self, x, mean, scale, Q=None, x_mean=None):
        if Q is None:
            Q = self.Q
        if use_clamp:
            if x_mean is None:
                x_mean = x.mean()
            x_min = x_mean - 15_000 * Q
            x_max = x_mean + 15_000 * Q
            x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())
        scale = torch.clamp(scale, min=1e-9)
        m1 = torch.distributions.normal.Normal(mean, scale)
        lower = m1.cdf(x - 0.5 * Q)
        upper = m1.cdf(x + 0.5 * Q)
        likelihood = torch.abs(upper - lower)
        likelihood = Low_bound.apply(likelihood)
        bits = -torch.log2(likelihood)
        return bits


class Entropy_gaussian_mix_prob_2(nn.Module):
    """Two-component Gaussian mixture entropy model."""
    def __init__(self, Q=1):
        super(Entropy_gaussian_mix_prob_2, self).__init__()
        self.Q = Q

    def forward(self, x,
                mean1, mean2,
                scale1, scale2,
                probs1, probs2,
                Q=None, x_mean=None, return_lkl=False):
        if Q is None:
            Q = self.Q
        if use_clamp:
            if x_mean is None:
                x_mean = x.mean().detach()
            x_min = x_mean - 15_000 * Q
            x_max = x_mean + 15_000 * Q
            x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())

        scale1 = torch.clamp(scale1, min=1e-9)
        scale2 = torch.clamp(scale2, min=1e-9)

        m1 = torch.distributions.normal.Normal(mean1, scale1)
        m2 = torch.distributions.normal.Normal(mean2, scale2)

        likelihood1 = torch.abs(m1.cdf(x + 0.5 * Q) - m1.cdf(x - 0.5 * Q))
        likelihood2 = torch.abs(m2.cdf(x + 0.5 * Q) - m2.cdf(x - 0.5 * Q))

        likelihood = Low_bound.apply(probs1 * likelihood1 + probs2 * likelihood2)

        if return_lkl:
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)
            bits = -torch.log2(likelihood)
            return bits


class Low_bound(torch.autograd.Function):
    """Lower-bound the likelihood to avoid log(0)."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        pass_through_if = np.logical_or(
            x.cpu().numpy() >= 1e-6, g.cpu().numpy() < 0.0)
        t = torch.Tensor(pass_through_if + 0.0).cuda()
        return grad1 * t
