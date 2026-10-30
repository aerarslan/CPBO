import torch
import random

def jnd_oracle(u_a, u_b, idx_a, idx_b, sigma, gamma):
    """
    Simulate a 3-outcome preference with JND and Gaussian perceptual noise.
    """
    eps_a = torch.randn(1) * sigma
    eps_b = torch.randn(1) * sigma
    y_a = u_a + eps_a
    y_b = u_b + eps_b
    diff = (y_b - y_a).abs()
    # tie
    if diff < gamma:
        if random.random() < 0.5:
            return torch.tensor([[idx_a, idx_b]])
        else:
            return torch.tensor([[idx_b, idx_a]])
    # winner by noisy utility
    if y_b > y_a:
        return torch.tensor([[idx_b, idx_a]])
    else:
        return torch.tensor([[idx_a, idx_b]])