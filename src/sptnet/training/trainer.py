"""Small training utilities shared by scripts and notebooks."""

import torch


def normalize_training_inputs(inputs):
    """Per-sample min-max normalization for `[B, 1, T, H, W]` input tensors."""
    img = inputs[:, 0]
    mins = img.reshape(img.shape[0], -1).min(dim=1).values[:, None, None, None]
    maxs = img.reshape(img.shape[0], -1).max(dim=1).values[:, None, None, None]
    inputs = inputs.clone()
    inputs[:, 0] = (img - mins) / (maxs - mins + 1e-8)
    return inputs
