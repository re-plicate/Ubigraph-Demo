import copy
import time

import numpy as np
import torch
from torch import nn

from ubigraph_core.temporal import CausalEdgeGenerator
from .config import STConfig
from .data import Episodes
from .memory import STBank, reference_indices, extend_for_target
from .model import JointGraphPredictor, prediction_loss


def load_edge(bank, cfg):
    if bank.edge_state is None:
        return None
    edge = CausalEdgeGenerator(cfg.window, cfg.hidden//2, cfg.edge_logit_scale)
    edge.load_state_dict(bank.edge_state)
    edge.requires_grad_(False)
    return edge


@torch.no_grad()
def evaluate_model(model, bank, query, prior, cfg, mode, index=None, adapter=None, target=None, info=False):
    index = reference_indices(bank, query, cfg) if index is None else index
    predictions, details = [], []
    edge = load_edge(bank, cfg)
    for start in range(0, len(query), cfg.batch_size):
        ids = np.arange(start, min(start+cfg.batch_size, len(query)))
        part = query.subset(ids)
        delta, detail = model(bank, part, index[ids], prior, edge, mode, True)
        if adapter is not None:
            j = query.names.index(target)
            delta[:, j] += cfg.correction_scale*adapter(detail["representation"][:, j]).squeeze(-1)
        predictions.append((part.x[..., cfg.window-1]+delta).clamp(0, 1))
        if info:
            details.append({k: v.detach() for k, v in detail.items()})
    prediction = torch.cat(predictions)
    return (prediction, {k: torch.cat([d[k] for d in details]) for k in details[0]}) if info else prediction


def fit(bank, validation, prior, cfg, seed, mode="attention"):
    started = time.perf_counter()
    torch.manual_seed(seed)
    model = JointGraphPredictor(cfg)  # fresh theta; never export condensation theta
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    edge = load_edge(bank, cfg)
    train = bank.episodes.subset(np.arange(cfg.neighbors, len(bank.episodes)))
    train = train.subset(train.labeled.any(1))
    ti, vi = reference_indices(bank, train, cfg), reference_indices(bank, validation, cfg)
    rng = np.random.default_rng(seed)
    best_score, best_epoch, best, stale = float("inf"), 0, None, 0
    curve = []
    for epoch in range(1, cfg.epochs+1):
        order = rng.permutation(len(train))
        losses = []
        model.train()
        for start in range(0, len(order), cfg.batch_size):
            ids = order[start:start+cfg.batch_size]
            q = train.subset(ids)
            optimizer.zero_grad(set_to_none=True)
            loss = prediction_loss(model(bank, q, ti[ids], prior, edge, mode), q, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite joint prediction loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % cfg.validation_interval == 0 or epoch == cfg.epochs:
            model.eval()
            prediction = evaluate_model(model, bank, validation, prior, cfg, mode, vi)
            score = float((prediction-validation.y).abs()[validation.labeled].mean())
            curve.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_nmae": score})
            if score < best_score-1e-7:
                best_score, best_epoch, best, stale = score, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += cfg.validation_interval
            if stale >= cfg.patience:
                break
    model.load_state_dict(best)
    return model.eval(), {"mode": mode, "fit_seconds": time.perf_counter()-started,
                          "selected_epoch": best_epoch, "epochs_run": epoch, "validation_nmae": best_score,
                          "training_episodes": len(train), "training_labeled_node_slots": int(train.labeled.sum()),
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "fresh_predictor_initialization": True}, curve


def adapt(model, source_bank, support, target, prior, cfg, mode):
    """One target budget, chronological fit/selection, no target September labels."""
    started = time.perf_counter()
    nfit = int(len(support)*cfg.adaptation_fraction)
    fitting = support.subset(np.arange(nfit))
    validation = support.subset(np.arange(nfit, len(support)))
    fit_bank = extend_for_target(source_bank, fitting, target)
    j = support.names.index(target)
    with torch.no_grad():
        _, train_info = evaluate_model(model, fit_bank, fitting, prior, cfg, mode, info=True)
        _, val_info = evaluate_model(model, fit_bank, validation, prior, cfg, mode, info=True)
    zt, zv = train_info["representation"][:, j], val_info["representation"][:, j]
    # Reconstruct unclipped deltas: clipping must not bias the adaptation objective.
    edge = load_edge(fit_bank, cfg)
    with torch.no_grad():
        dt = model(fit_bank, fitting, reference_indices(fit_bank, fitting, cfg), prior, edge, mode)[:, j]
        dv = model(fit_bank, validation, reference_indices(fit_bank, validation, cfg), prior, edge, mode)[:, j]
    adapter = nn.Linear(cfg.hidden, 1)
    nn.init.zeros_(adapter.weight)
    nn.init.zeros_(adapter.bias)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=cfg.adaptation_lr, weight_decay=cfg.weight_decay)
    yt = fitting.y[:, j]-fitting.x[:, j, cfg.window-1]
    # Earliest K target observations provide context, not own-answer supervised input.
    take = torch.arange(len(fitting)) >= cfg.neighbors
    best = copy.deepcopy(adapter.state_dict())
    best_score = float(((validation.x[:, j, cfg.window-1]+dv).clamp(0, 1)-validation.y[:, j]).abs().mean())
    best_epoch, trace = 0, []
    for epoch in range(1, cfg.adaptation_epochs+1):
        optimizer.zero_grad(set_to_none=True)
        p = dt+cfg.correction_scale*adapter(zt).squeeze(-1)
        loss = torch.nn.functional.huber_loss(p[take]/cfg.delta_scale, yt[take]/cfg.delta_scale)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pval = (validation.x[:, j, cfg.window-1]+dv+cfg.correction_scale*adapter(zv).squeeze(-1)).clamp(0, 1)
            score = float((pval-validation.y[:, j]).abs().mean())
        trace.append({"epoch": epoch, "fit_loss": float(loss.detach()), "support_validation_nmae": score})
        if score < best_score-1e-7:
            best_score, best_epoch, best = score, epoch, copy.deepcopy(adapter.state_dict())
    adapter.load_state_dict(best)
    final_bank = extend_for_target(source_bank, support, target)
    return final_bank, adapter.eval(), {"target": target, "label_budget": len(support),
        "fit_labels": nfit, "selection_labels": len(support)-nfit, "selected_epoch": best_epoch,
        "support_validation_nmae": best_score, "target_full_history_used": False,
        "target_september_labels_used": 0, "support_first_label_ns": int(support.label_ns.min()),
        "support_last_label_ns": int(support.label_ns.max()), "seconds": time.perf_counter()-started}, trace


def make_bundle(model, bank, prior, cfg, mode, floor, adapter=None, target=None, provenance=None):
    return {"format_version": 3, "config": cfg.asdict(), "model_state": model.state_dict(),
            "bank": bank.asdict(), "prior": prior.detach(), "mode": mode,
            "minimum_serving_anchor_ns": int(floor), "adapter_state": None if adapter is None else adapter.state_dict(),
            "target": target, "provenance": provenance or {}}


class STForecaster:
    def __init__(self, bundle):
        self.bundle = bundle
        self.cfg = STConfig(**bundle["config"])
        self.bank = STBank.fromdict(bundle["bank"])
        self.model = JointGraphPredictor(self.cfg)
        self.model.load_state_dict(bundle["model_state"])
        self.model.eval()
        self.adapter = None
        if bundle["adapter_state"] is not None:
            self.adapter = nn.Linear(self.cfg.hidden, 1)
            self.adapter.load_state_dict(bundle["adapter_state"])
            self.adapter.eval()

    @classmethod
    def load(cls, path):
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    def predict(self, query, info=False):
        if query.names != self.bank.episodes.names:
            raise ValueError("Parking node order differs from checkpoint contract")
        if not torch.equal(query.capacities, self.bank.episodes.capacities):
            raise ValueError("Capacity differs from checkpoint")
        if int(query.anchor_ns.min()) < self.bundle["minimum_serving_anchor_ns"]:
            raise ValueError("Prediction origin predates the model/selection cutoff")
        if not torch.isfinite(query.x).all():
            raise ValueError("Nonfinite inference features")
        return evaluate_model(self.model, self.bank, query, self.bundle["prior"], self.cfg,
                              self.bundle["mode"], adapter=self.adapter, target=self.bundle["target"], info=info)
