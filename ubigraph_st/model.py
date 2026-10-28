"""Shared temporal message passing -> time-conditioned spatial attention -> regression."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ubigraph_core.temporal import tensor_key
from .memory import gather_memory


MODES = ("attention", "local", "gate", "uniform", "no_prior", "no_residual")


class JointGraphPredictor(nn.Module):
    def __init__(self, cfg, features=None):
        super().__init__()
        self.cfg = cfg
        d, f = cfg.hidden, features or (3 * cfg.window + 8)
        self.encoder = nn.Sequential(nn.Linear(f + 2, d), nn.SiLU())
        self.temporal_self = nn.Linear(d, d)
        self.temporal_message = nn.Linear(d, d, bias=False)
        self.q = nn.Linear(d + 8, d)
        self.k, self.v = nn.Linear(d, d), nn.Linear(d, d)
        self.spatial_out = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.spatial_head = nn.Linear(d, 1)
        self.output_gate = nn.Linear(2 * d, 1)
        # Small nonzero head permits attention gradients in the first matching step.
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, bank, query, index, prior, edge=None, mode="attention", return_info=False):
        if mode not in MODES:
            raise ValueError(f"Unknown fusion mode: {mode}")
        cfg = self.cfg
        x = query.x
        b, n, f = x.shape
        rx, delta, valid = gather_memory(bank, query, index, cfg)
        qkey = tensor_key(x.reshape(-1, f), cfg.window).reshape(b, n, -1)
        rkey = tensor_key(rx.reshape(-1, f), cfg.window).reshape(b, n, cfg.neighbors, -1)
        distance = (qkey[:, :, None] - rkey).square().sum(-1)
        bandwidth = distance.masked_fill(~valid, 0).amax(-1, keepdim=True).clamp_min(.02**2)
        scores = -distance / bandwidth
        if edge is not None:
            scores = scores + edge(qkey.reshape(b*n, -1), rkey.reshape(b*n, cfg.neighbors, -1)).reshape_as(scores)
        weights = torch.softmax(scores.masked_fill(~valid, -1e9), -1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        reference = self.encoder(torch.cat([rx, delta[..., None], torch.ones_like(delta[..., None])], -1))
        own = self.encoder(torch.cat([x, torch.zeros(b, n, 2)], -1))
        context = (weights[..., None] * reference).sum(-2)
        h = F.silu(self.temporal_self(own) + self.temporal_message(context))
        base = (weights * delta).sum(-1)
        q = self.q(torch.cat([h, x[..., -8:]], -1))
        logits = torch.einsum("bid,bjd->bij", q, self.k(h)) / math.sqrt(cfg.hidden)
        prior = prior.to(x)
        if mode != "no_prior":
            logits = logits + cfg.attention_prior * prior.clamp_min(1e-6).log()[None]
        if mode == "uniform":
            logits = torch.zeros_like(logits)
        allowed = query.available[:, None, :].expand(b, n, n).clone()
        if mode == "gate":
            allowed = allowed & ~torch.eye(n, dtype=torch.bool)[None]
        spatial_weights = torch.softmax(logits.masked_fill(~allowed, -1e9), -1) * allowed
        spatial_weights = spatial_weights / spatial_weights.sum(-1, keepdim=True).clamp_min(1e-12)
        spatial = torch.einsum("bij,bjd->bid", spatial_weights, self.v(h))
        local = base + cfg.correction_scale * self.head(h).squeeze(-1)
        if mode == "local":
            z, prediction = h, local
            spatial_weights = spatial_weights * 0
        elif mode == "gate":
            external = cfg.correction_scale * self.spatial_head(spatial).squeeze(-1)
            gate = torch.sigmoid(self.output_gate(torch.cat([h, spatial], -1))).squeeze(-1)
            no_external = ~allowed.any(-1)
            gate = torch.where(no_external, torch.ones_like(gate), gate)
            prediction = gate * local + (1 - gate) * external
            z = h
        else:
            z = self.norm(self.spatial_out(spatial) + (h if mode != "no_residual" else 0))
            prediction = base + cfg.correction_scale * self.head(z).squeeze(-1)
        if return_info:
            return prediction, {"spatial_weights": spatial_weights, "temporal_weights": weights,
                                "reference_indices": index, "representation": z, "retrieval_delta": base}
        return prediction


def prediction_loss(prediction, query, cfg, mask=None):
    truth = query.y - query.x[..., cfg.window - 1]
    mask = query.labeled if mask is None else mask & query.labeled
    if not mask.any():
        raise ValueError("No valid supervised queries")
    return F.huber_loss(prediction[mask] / cfg.delta_scale, truth[mask] / cfg.delta_scale, delta=1.)
