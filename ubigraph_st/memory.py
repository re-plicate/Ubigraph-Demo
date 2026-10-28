"""A synchronous bank with independently masked, causal temporal reference edges."""
from dataclasses import dataclass

import numpy as np
import torch

from ubigraph_core.temporal import LabelBank, causal_knn, tensor_key
from .data import Episodes


@dataclass
class STBank:
    episodes: Episodes
    edge_state: dict | None = None
    kind: str = "real"

    def asdict(self):
        return {"episodes": self.episodes.asdict(), "edge_state": self.edge_state, "kind": self.kind}

    @classmethod
    def fromdict(cls, d):
        return cls(Episodes.fromdict(d["episodes"]), d["edge_state"], d["kind"])

    def with_x(self, x):
        d = dict(self.episodes.__dict__)
        d["x"] = x
        return STBank(Episodes(**d), self.edge_state, self.kind)


def reference_indices(bank, queries, cfg):
    """Non-differentiable exact selection; features/edge weights remain differentiable.

    Padding is -1, so early/sparse history does not require invented outcomes.
    KD-tree is used for large banks; small learned banks use squared distances.
    """
    memory = bank.episodes
    result = torch.full((len(queries), len(queries.names), cfg.neighbors), -1, dtype=torch.long)
    for j, name in enumerate(queries.names):
        if name not in memory.names:
            continue
        m = memory.names.index(name)
        rows = torch.where(memory.labeled[:, m])[0]
        if not len(rows):
            continue
        mx = memory.x[rows, m].detach()
        qx = queries.x[:, j].detach()
        allowed = ((memory.label_ns[rows][None] <= queries.anchor_ns[:, None])
                   & (memory.anchor_ns[rows][None] != queries.anchor_ns[:, None]))
        k = min(cfg.neighbors, len(rows))
        if len(rows) > 512:
            enough = torch.where(allowed.sum(1) >= cfg.neighbors)[0]
            lb = LabelBank(mx, torch.zeros(len(rows)), memory.anchor_ns[rows], memory.label_ns[rows],
                           memory.anchor_ns[rows])
            if len(enough):
                picked = causal_knn(qx[enough].numpy(), queries.anchor_ns[enough].numpy(),
                                    queries.anchor_ns[enough].numpy(), lb, cfg.window, cfg.neighbors)
                result[enough, j] = rows[picked]
            short = torch.where(allowed.sum(1) < cfg.neighbors)[0]
        else:
            short = torch.arange(len(queries))
        if len(short):
            qkey, mkey = tensor_key(qx[short], cfg.window), tensor_key(mx, cfg.window)
            distance = (qkey[:, None] - mkey[None]).square().sum(-1)
            ids = distance.masked_fill(~allowed[short], torch.inf).argsort(dim=1, stable=True)[:, :k]
            chosen = torch.where(allowed[short].gather(1, ids), rows[ids], -1)
            result[short, j, :k] = chosen
    return result


def gather_memory(bank, queries, index, cfg):
    """Gather K references only, rather than encoding the entire bank each batch."""
    e = bank.episodes
    features, changes, valid = [], [], []
    for j, name in enumerate(queries.names):
        if name not in e.names:
            features.append(torch.zeros(len(queries), cfg.neighbors, queries.x.shape[-1]))
            changes.append(torch.zeros(len(queries), cfg.neighbors))
            valid.append(torch.zeros(len(queries), cfg.neighbors, dtype=torch.bool))
            continue
        m = e.names.index(name)
        ids = index[:, j].clamp_min(0)
        ok = index[:, j] >= 0
        if ((e.label_ns[ids] > queries.anchor_ns[:, None]) & ok).any():
            raise ValueError("Future outcome in temporal messages")
        if ((e.anchor_ns[ids] == queries.anchor_ns[:, None]) & ok).any():
            raise ValueError("Current episode outcome in temporal messages")
        if (~e.labeled[ids, m] & ok).any():
            raise ValueError("Unlabeled reference")
        features.append(e.x[ids, m])
        changes.append(e.y[ids, m] - e.x[ids, m, cfg.window - 1])
        valid.append(ok)
    return torch.stack(features, 1), torch.stack(changes, 1), torch.stack(valid, 1)


def extend_for_target(source_bank, support, target):
    """Target has only its budgeted REAL memory; no full-target checkpoint or bank."""
    a = source_bank.episodes
    names = support.names
    t = sorted(set(a.anchor_ns.tolist()) | set(support.anchor_ns.tolist()))
    index = {v: i for i, v in enumerate(t)}
    n, f = len(names), a.x.shape[-1]
    x = torch.zeros(len(t), n, f)
    y = torch.zeros(len(t), n)
    available = torch.zeros(len(t), n, dtype=torch.bool)
    labeled = available.clone()
    labels = torch.zeros(len(t), dtype=torch.long)
    for data, members in ((a, a.names), (support, [target])):
        ids = torch.tensor([index[v] for v in data.anchor_ns.tolist()])
        labels[ids] = data.label_ns
        for name in members:
            src, dest = data.names.index(name), names.index(name)
            x[ids, dest], y[ids, dest] = data.x[:, src], data.y[:, src]
            available[ids, dest], labeled[ids, dest] = data.available[:, src], data.labeled[:, src]
    e = Episodes(x, y, available, labeled, torch.tensor(t), labels, names, support.capacities)
    return STBank(e, source_bank.edge_state, source_bank.kind + "+budgeted_target")


def bank_audit(bank, cfg):
    e = bank.episodes
    ids = reference_indices(bank, e, cfg)
    _, _, valid = gather_memory(bank, e, ids, cfg)
    slots = int(e.labeled.sum())
    return {"episodes": len(e), "node_slots": int(e.x.shape[0] * e.x.shape[1]),
            "labeled_node_slots": slots, "per_node_labeled": dict(zip(e.names, e.labeled.sum(0).tolist())),
            "future_edges": 0, "self_answer_edges": 0, "reference_edges": int(valid.sum()),
            "rates_physical": bool(((e.x[..., :cfg.window] >= 0) & (e.x[..., :cfg.window] <= 1)).all()),
            "labels_physical": bool(((e.y >= 0) & (e.y <= 1)).all()), "kind": bank.kind}
