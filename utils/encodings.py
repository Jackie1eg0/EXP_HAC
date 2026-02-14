"""
Pure-Python STE quantization functions and utility functions.
Ported from HAC-plus official implementation.
"""
import torch
import torch.nn as nn
import numpy as np


use_clamp = True


def get_binary_vxl_size(binary_vxl):
    """Estimate the bit cost of a binary tensor using entropy coding."""
    ttl_num = binary_vxl.numel()
    pos_num = torch.sum(binary_vxl)
    neg_num = ttl_num - pos_num
    Pg = pos_num / ttl_num
    Pg = torch.clamp(Pg, min=1e-6, max=1 - 1e-6)
    pos_prob = Pg
    neg_prob = (1 - Pg)
    pos_bit = pos_num * (-torch.log2(pos_prob))
    neg_bit = neg_num * (-torch.log2(neg_prob))
    ttl_bit = pos_bit + neg_bit
    ttl_bit += 32  # overhead for Pg
    return Pg, ttl_bit, ttl_bit.item() / 8.0 / 1024 / 1024, ttl_num


class STE_binary(torch.autograd.Function):
    """Straight-Through Estimator for binary quantization to {-1, +1}."""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.clamp(input, min=-1, max=1)
        p = (input >= 0) * (+1.0)
        n = (input < 0) * (-1.0)
        out = p + n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


class STE_multistep(torch.autograd.Function):
    """Straight-Through Estimator for multi-step uniform quantization."""
    @staticmethod
    def forward(ctx, input, Q, input_mean=None):
        if use_clamp:
            if input_mean is None:
                input_mean = input.mean()
            input_min = input_mean - 15_000 * Q
            input_max = input_mean + 15_000 * Q
            input = torch.clamp(input, min=input_min.detach(), max=input_max.detach())
        Q_round = torch.round(input / Q)
        Q_q = Q_round * Q
        return Q_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
