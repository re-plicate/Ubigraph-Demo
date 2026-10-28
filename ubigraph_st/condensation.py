"""GroC adaptation: match gradients of the complete synchronized ST regression task.

Fixed continuous Y, learned historical rates and temporal edge generator; CoSimHeat
and calendar/masks stay fixed. Localized raw-gradient ascent follows the supplied
GroC snippet, with explicit physical/trust projections for this regression task.
"""
import math
import time

import numpy as np
import torch
from torch import nn

from ubigraph_core.gradients import gradient_distance
from ubigraph_core.temporal import CausalEdgeGenerator
from ubigraph_core.utils import log
from .memory import STBank, reference_indices
from .model import JointGraphPredictor, prediction_loss


def initial_bank(real, cfg, seed, random=False):
    rng = np.random.default_rng(seed)
    size = min(cfg.synthetic_nodes, len(real))
    context = np.arange(cfg.neighbors)
    remaining = np.arange(cfg.neighbors, len(real))
    if random:
        selected = rng.choice(remaining, size-len(context), replace=False)
    else:
        change = (real.y - real.x[..., cfg.window-1]).abs()
        score = (change * real.labeled).sum(1) / real.labeled.sum(1).clamp_min(1)
        ordered = remaining[np.argsort(score[remaining].numpy(), kind="stable")]
        selected = np.array([rng.choice(part) for part in np.array_split(ordered, size-len(context))])
    ids = np.sort(np.r_[context, selected])
    return STBank(real.subset(ids), kind="random" if random else "initialized"), ids


def gradients(loss, model, create_graph=False):
    params = tuple(model.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=create_graph, allow_unused=True)
    return tuple(g if g is not None else p * 0 for g, p in zip(grads, params, strict=True))


def matching(model, bank, query_ids, real_gradients, prior, edge, cfg):
    query = bank.episodes.subset(query_ids)
    idx = reference_indices(bank, query, cfg)
    loss = prediction_loss(model(bank, query, idx, prior, edge), query, cfg)
    return gradient_distance(gradients(loss, model, True), real_gradients, cfg.magnitude_weight)


def localized_noise(model, bank, query_ids, real_gradients, prior, edge, cfg):
    clean = bank.episodes.x.detach()
    rates = clean[..., :cfg.window]
    eligible = (clean[..., cfg.window:2*cfg.window] > .5) & bank.episodes.available[..., None]
    if cfg.adversarial_epsilon == 0 or not eligible.any():
        return torch.zeros_like(clean), {"max_noise": 0., "active_features": 0}
    delta = (2 * torch.rand_like(rates) - 1) * cfg.adversarial_epsilon * eligible
    active = eligible
    for _ in range(cfg.adversarial_steps):
        delta = delta.detach().requires_grad_(True)
        mask = torch.ones_like(delta, requires_grad=True)
        x = torch.cat([(rates + delta*mask*active).clamp(0, 1), clean[..., cfg.window:]], -1)
        objective = matching(model, bank.with_x(x), query_ids, real_gradients, prior, edge, cfg)
        gd, gm = torch.autograd.grad(objective, (delta, mask))
        count = max(1, math.ceil(int(eligible.sum()) * cfg.vulnerable_fraction))
        positions = gm.detach().abs().masked_fill(~eligible, -torch.inf).flatten().topk(count).indices
        active = torch.zeros_like(delta, dtype=torch.bool).flatten()
        active[positions] = True
        active = active.reshape_as(delta)
        delta = ((delta.detach() + cfg.adversarial_lr * gd.detach())
                 .clamp(-cfg.adversarial_epsilon, cfg.adversarial_epsilon) * active)
        delta = ((rates + delta).clamp(0, 1) - rates).detach()
    noise = torch.cat([delta, torch.zeros_like(clean[..., cfg.window:])], -1)
    return noise, {"max_noise": float(delta.abs().max()), "active_features": int(active.sum())}


def condense(real, prior, cfg, seed, adversarial=True, tag=""):
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    bank, selected = initial_bank(real, cfg, seed)
    initial = bank.episodes.x[..., :cfg.window].clone()
    rates = nn.Parameter(initial.clone())
    fixed = bank.episodes.x[..., cfg.window:]
    edge = CausalEdgeGenerator(cfg.window, cfg.hidden//2, cfg.edge_logit_scale)
    initial_edge = [p.detach().clone() for p in edge.parameters()]
    opt_x = torch.optim.Adam([rates], lr=cfg.feature_lr)
    opt_edge = torch.optim.Adam(edge.parameters(), lr=cfg.adjacency_lr)
    real_bank = STBank(real)
    real_index = reference_indices(real_bank, real, cfg)
    rq_pool = np.flatnonzero(real.labeled.any(1).numpy())[cfg.neighbors:]
    sq_pool = np.arange(cfg.neighbors, len(bank.episodes))
    trace = []
    for step in range(cfg.condense_steps):
        if step % cfg.reset_interval == 0:
            torch.manual_seed(seed+10000+step//cfg.reset_interval)
            model = JointGraphPredictor(cfg)
            opt_theta = torch.optim.SGD(model.parameters(), lr=cfg.theta_lr)
        rq = np.sort(rng.choice(rq_pool, min(cfg.real_batch, len(rq_pool)), replace=False))
        sq = np.sort(rng.choice(sq_pool, min(cfg.real_batch, len(sq_pool)), replace=False))
        query = real.subset(rq)
        real_loss = prediction_loss(model(real_bank, query, real_index[rq], prior), query, cfg)
        real_grads = tuple(g.detach() for g in gradients(real_loss, model))
        clean = bank.with_x(torch.cat([rates, fixed], -1))
        if adversarial:
            noise, noise_info = localized_noise(model, clean, sq, real_grads, prior, edge, cfg)
        else:
            noise, noise_info = torch.zeros_like(clean.episodes.x), {"max_noise": 0., "active_features": 0}
        match = matching(model, clean.with_x(clean.episodes.x+noise), sq, real_grads, prior, edge, cfg)
        retrieval_ids = reference_indices(clean, query, cfg)
        retrieval = prediction_loss(model(clean, query, retrieval_ids, prior, edge), query, cfg)
        valid_rates = bank.episodes.available[..., None].expand_as(rates)
        anchor = (((rates-initial)/cfg.trust_radius).square()*valid_rates).sum()/valid_rates.sum().clamp_min(1)
        shape = ((rates.diff(dim=-1)-initial.diff(dim=-1))/cfg.trust_radius).square().mean()
        objective = match + cfg.retrieval_weight*retrieval + cfg.anchor_weight*anchor + cfg.shape_weight*shape
        feature_phase = step % (cfg.feature_steps+cfg.adjacency_steps) < cfg.feature_steps
        parameters = [rates] if feature_phase else list(edge.parameters())
        opt = opt_x if feature_phase else opt_edge
        opt.zero_grad(set_to_none=True)
        grads = torch.autograd.grad(objective, parameters)
        if not torch.isfinite(objective) or not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError(f"Nonfinite ST condensation: {tag}/{step}")
        for p, g in zip(parameters, grads, strict=True):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(parameters, 5.)
        opt.step()
        with torch.no_grad():
            rates.copy_(torch.maximum(torch.minimum(rates, initial+cfg.trust_radius), initial-cfg.trust_radius).clamp(0, 1))
            rates.copy_(torch.where(valid_rates, rates, initial))
        updated = bank.with_x(torch.cat([rates.detach(), fixed], -1))
        synq = updated.episodes.subset(sq)
        si = reference_indices(updated, synq, cfg)
        opt_theta.zero_grad(set_to_none=True)
        theta_loss = prediction_loss(model(updated, synq, si, prior, edge), synq, cfg)
        gs = gradients(theta_loss, model)
        for p, g in zip(model.parameters(), gs, strict=True):
            p.grad = g.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt_theta.step()
        trace.append({"step": step, "matching": float(match.detach()), "retrieval_loss": float(retrieval.detach()),
                      "anchor": float(anchor.detach()), "objective": float(objective.detach()),
                      "phase": "features" if feature_phase else "temporal_edges", **noise_info})
        if (step+1) % max(20, cfg.condense_steps//2) == 0:
            log(f"ST {tag}: condensation {step+1}/{cfg.condense_steps}")
    final = bank.with_x(torch.cat([rates.detach(), fixed], -1))
    final.kind = "groc" if adversarial else "gcond"
    final.edge_state = {k: v.detach().clone() for k, v in edge.state_dict().items()}
    summary = {"kind": final.kind, "seconds": time.perf_counter()-started,
               "original_episodes": len(real), "synthetic_episodes": len(final.episodes),
               "node_count": len(real.names), "initial_indices": selected.tolist(),
               "fixed_absolute_targets": True, "fixed_calendar_masks_and_spatial_prior": True,
               "feature_change_l2": float((rates-initial).detach().norm()),
               "max_feature_displacement": float((rates-initial).detach().abs().max()),
               "edge_change_l2": float(torch.sqrt(sum((p.detach()-v).square().sum() for p, v in zip(edge.parameters(), initial_edge, strict=True)))),
               "max_adversarial_noise": max(t["max_noise"] for t in trace),
               "model_reinitializations": (cfg.condense_steps-1)//cfg.reset_interval+1}
    return final, summary, trace
