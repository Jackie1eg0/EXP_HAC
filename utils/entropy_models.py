"""
Entropy models for rate estimation during training.
Ported from HAC-plus official implementation.

Performance optimizations:
  - Low_bound: pure-CUDA backward (no CPU-GPU sync)
  - Gaussian CDF: fused erfc computation (avoid torch.distributions overhead)
"""
import torch
import torch.nn as nn
import math
from utils.encodings import use_clamp

# Pre-computed constant: 1 / sqrt(2)
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _normal_cdf(x, mean, scale):
    """Vectorized Gaussian CDF using erfc — ~2x faster than torch.distributions."""
    return 0.5 * torch.erfc(-(x - mean) / (scale * math.sqrt(2.0) + 1e-10))


class Entropy_gaussian(nn.Module):
    """Single Gaussian entropy model (optimized)."""
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
        half_Q = 0.5 * Q
        upper = _normal_cdf(x + half_Q, mean, scale)
        lower = _normal_cdf(x - half_Q, mean, scale)
        likelihood = torch.abs(upper - lower)
        likelihood = Low_bound.apply(likelihood)
        bits = -torch.log2(likelihood)
        return bits


class Entropy_gaussian_mix_prob_2(nn.Module):
    """Two-component Gaussian mixture entropy model (optimized)."""
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

        half_Q = 0.5 * Q
        x_upper = x + half_Q
        x_lower = x - half_Q

        likelihood1 = torch.abs(_normal_cdf(x_upper, mean1, scale1) - _normal_cdf(x_lower, mean1, scale1))
        likelihood2 = torch.abs(_normal_cdf(x_upper, mean2, scale2) - _normal_cdf(x_lower, mean2, scale2))

        likelihood = Low_bound.apply(probs1 * likelihood1 + probs2 * likelihood2)

        if return_lkl:
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)
            bits = -torch.log2(likelihood)
            return bits


class Low_bound(torch.autograd.Function):
    """
    Lower-bound the likelihood to avoid log(0).

    PERF FIX: Pure CUDA backward — no CPU-GPU synchronization!
    Original code used x.cpu().numpy() which forces a full CUDA pipeline stall.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(x, min=1e-6)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        # Pure GPU: pass gradient where x >= threshold OR g < 0 (allow negative gradients to push values up)
        pass_through = (x >= 1e-6) | (g < 0.0)
        return g * pass_through.float()
