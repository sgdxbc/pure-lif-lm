"""Training-only cosine-similarity distillation at the GLA and MLP projections."""
from contextlib import contextmanager
import math

import numpy as np
import torch
import torch.nn.functional as F


def sample_positions(batch, seq, count, seed, step, device):
    rng = np.random.default_rng([seed, step])
    rows = np.stack([rng.choice(seq, min(seq, count), replace=False) for _ in range(batch)])
    return torch.from_numpy(rows).to(device)


def cosine_gram(features):
    with torch.autocast(features.device.type, enabled=False):
        features = features.float()
        norms = features.norm(dim=-1, keepdim=True)
        # A silent projection has no direction. Give it zero similarity and zero gradient.
        unit = features / norms.clamp_min(1e-6).masked_fill(norms == 0, float("inf"))
        return unit @ unit.transpose(-2, -1)


@contextmanager
def capture_grams(model, positions):
    if positions is None:
        yield None
        return
    grams = []
    rows = torch.arange(positions.size(0), device=positions.device)[:, None]

    def capture(_module, _inputs, output):
        grams.append(cosine_gram(output[rows, positions]))

    hooks = []
    try:
        for block in model.blocks:
            hooks.append(block.la.proj.register_forward_hook(capture))
            hooks.append(block.fc.register_forward_hook(capture))
        yield grams
    finally:
        for hook in hooks:
            hook.remove()


def feature_mse(student, teacher):
    return torch.stack([F.mse_loss(s, t) for s, t in zip(student, teacher, strict=True)]).mean()


def calibrated_weight(batches, params, accum, fraction):
    """Match RMS full-batch gradient norms at the starting weights, before clipping.

    Each batch yields a task/feature loss pair per micro-batch. Sum gradients within
    the batch first, as in a training update. autograd.grad leaves .grad untouched.
    """
    squared_norms = [0., 0.]
    for batch in batches:
        sums = [[None] * len(params) for _ in squared_norms]
        for task, feature in batch:
            for k, loss in enumerate((task, feature)):
                grads = torch.autograd.grad(loss / accum, params, retain_graph=k == 0, allow_unused=True)
                for i, grad in enumerate(grads):
                    if grad is not None:
                        if sums[k][i] is None:
                            sums[k][i] = grad.clone()
                        else:
                            sums[k][i].add_(grad)
                del grads
            del task, feature, loss
        for k, group in enumerate(sums):
            squared_norms[k] += torch.stack([g.norm().square() for g in group if g is not None]).sum().item()
        del sums, group
    task_sq, feature_sq = squared_norms
    if not math.isfinite(task_sq) or not math.isfinite(feature_sq) or feature_sq <= 0:
        raise ValueError("feature weight calibration requires finite gradients and a nonzero feature gradient")
    return fraction * math.sqrt(task_sq / feature_sq)
