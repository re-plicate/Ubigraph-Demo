"""Causal retrieval keys, reference banks and trainable temporal edge scores."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn


def graph_key(x: np.ndarray, window: int) -> np.ndarray:
    # Only past states and known calendar encodings; never future regression labels.
    return np.concatenate([x[:, :window], 0.2 * x[:, 3 * window:3 * window + 4]], axis=1)



@dataclass
class LabelBank:
    x: torch.Tensor
    delta: torch.Tensor
    anchor_ns: torch.Tensor
    label_ns: torch.Tensor
    uid: torch.Tensor
    edge_state: dict | None = None

    @classmethod
    def from_windows(cls, windows):
        return cls(torch.tensor(windows.x),
                   torch.tensor(windows.y - windows.x[:, (windows.x.shape[1] - 8) // 3 - 1]),
                   torch.tensor(windows.anchor_ns), torch.tensor(windows.label_ns),
                   torch.arange(len(windows), dtype=torch.long))

    def subset(self, index):
        return LabelBank(self.x[index].clone(), self.delta[index].clone(), self.anchor_ns[index].clone(),
                         self.label_ns[index].clone(), self.uid[index].clone(), self.edge_state)

    def asdict(self):
        return {"x": self.x.detach(), "delta": self.delta.detach(), "anchor_ns": self.anchor_ns,
                "label_ns": self.label_ns, "uid": self.uid, "edge_state": self.edge_state}

    @classmethod
    def fromdict(cls, d):
        return cls(**d)



def causal_knn(query_x, query_time, query_uid, bank: LabelBank, window: int, k: int):
    """Exact nearest *available* outcomes; adaptive KD-tree expansion handles ties."""
    q = graph_key(np.asarray(query_x), window)
    r = graph_key(bank.x.detach().numpy(), window)
    times = np.asarray(query_time, dtype=np.int64)
    uid = np.asarray(query_uid, dtype=np.int64)
    release, ref_uid = bank.label_ns.numpy(), bank.uid.numpy()
    if len(r) < k:
        raise ValueError("Reference bank is smaller than k")
    tree = cKDTree(r)
    result = np.empty((len(q), k), dtype=np.int64)
    remaining = np.arange(len(q))
    candidates = min(len(r), max(32, 4 * k))
    while len(remaining):
        distances, index = tree.query(q[remaining], k=candidates, workers=1)
        index = np.asarray(index).reshape(len(remaining), candidates)
        distances = np.asarray(distances).reshape(len(remaining), candidates)
        allowed = ((release[index] <= times[remaining, None])
                   & ((ref_uid[index] != uid[remaining, None]) | (uid[remaining, None] < 0)))
        good = allowed.sum(axis=1) >= k
        for row in np.flatnonzero(good):
            qi = remaining[row]
            kth_distance = distances[row, allowed[row]][k - 1]
            # Stable UID tie handling makes adding unavailable future records inert.
            pool = np.asarray(tree.query_ball_point(q[qi], kth_distance + 1e-10), dtype=int)
            valid = (release[pool] <= times[qi]) & ((ref_uid[pool] != uid[qi]) | (uid[qi] < 0))
            pool = pool[valid]
            d2 = ((r[pool] - q[qi]) ** 2).sum(axis=1)
            ordering = np.lexsort((ref_uid[pool], d2))
            result[qi] = pool[ordering[:k]]
        remaining = remaining[~good]
        if not len(remaining):
            break
        if candidates == len(r):
            raise ValueError("Query has fewer than k already-released, non-self reference answers")
        candidates = min(len(r), candidates * 2)
    return torch.tensor(result)



def tensor_key(x, window):
    return torch.cat([x[:, :window], .2 * x[:, 3 * window:3 * window + 4]], dim=1)



class CausalEdgeGenerator(nn.Module):
    """Symmetric pair MLP scores, directed causal mask, positive row normalization.

    The learned score is bounded and added to historical-distance kernel logits.
    No diagonal/self-label edge and no post-training adjacency threshold exists.
    """
    def __init__(self, window, hidden=16, scale=.5):
        super().__init__()
        self.window, self.scale = window, scale
        features = window + 4
        self.network = nn.Sequential(nn.Linear(2 * features, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, query_key, ref_key):
        q = query_key[:, None, :].expand_as(ref_key)
        forward = self.network(torch.cat([q, ref_key], -1)).squeeze(-1)
        backward = self.network(torch.cat([ref_key, q], -1)).squeeze(-1)
        return self.scale * torch.tanh((forward + backward) / 2)

