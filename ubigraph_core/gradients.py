"""Gradient direction and magnitude matching."""
import torch
from torch.nn import functional as F


def gradient_distance(synthetic, real, magnitude_weight: float = 0.05):
    """Column cosine of mathematical W[in,out] == rows of torch W[out,in].

    One-dimensional bias arrays are one vector; scalar biases use only magnitude.
    Relative squared distance controls magnitudes and scalar regression gradients.
    """
    angle_terms, magnitude_terms = [], []
    for gs, gr in zip(synthetic, real, strict=True):
        if gs.numel() > 1:
            s = gs.reshape(gs.shape[0], -1) if gs.ndim > 1 else gs.reshape(1, -1)
            r = gr.reshape(gr.shape[0], -1) if gr.ndim > 1 else gr.reshape(1, -1)
            cosine = F.cosine_similarity(s, r, dim=1, eps=1e-8)
            both_zero = (s.norm(dim=1) < 1e-8) & (r.norm(dim=1) < 1e-8)
            angle_terms.append(torch.where(both_zero, torch.zeros_like(cosine), 1 - cosine).mean())
        magnitude_terms.append((gs - gr).square().sum() / (gs.square().sum() + gr.square().sum() + 1e-8))
    angular = torch.stack(angle_terms).mean()
    magnitude = torch.stack(magnitude_terms).mean()
    return angular + magnitude_weight * magnitude

