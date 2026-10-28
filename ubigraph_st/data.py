"""Historical availability is computed independently of future-label availability."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ubigraph_core.data import encode_windows
from ubigraph_core.utils import uniform_indices


@dataclass
class Episodes:
    x: torch.Tensor               # [T,N,3L+8]
    y: torch.Tensor               # [T,N], next rate, zero when masked
    available: torch.Tensor       # [T,N], based on HISTORY ONLY
    labeled: torch.Tensor         # [T,N], history and future truth available
    anchor_ns: torch.Tensor        # [T]
    label_ns: torch.Tensor         # [T]
    names: list[str]
    capacities: torch.Tensor      # [N]

    def __len__(self):
        return len(self.x)

    def subset(self, rows):
        rows = torch.as_tensor(np.asarray(rows)) if not isinstance(rows, torch.Tensor) else rows
        return Episodes(self.x[rows], self.y[rows], self.available[rows], self.labeled[rows],
                        self.anchor_ns[rows], self.label_ns[rows], self.names, self.capacities)

    def nodes(self, names):
        ids = [self.names.index(n) for n in names]
        return Episodes(self.x[:, ids], self.y[:, ids], self.available[:, ids], self.labeled[:, ids],
                        self.anchor_ns, self.label_ns, list(names), self.capacities[ids])

    def capped(self, count):
        return self.subset(uniform_indices(len(self), count))

    def asdict(self):
        return {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                for k, v in self.__dict__.items()}

    @classmethod
    def fromdict(cls, d):
        return cls(**d)


def build_episodes(prepared, cfg):
    names = sorted(prepared.splits)
    first = min(prepared.grids[n].index.min() for n in names)
    last = max(prepared.grids[n].index.max() for n in names)
    grid_times = pd.date_range(first, last, freq=f"{cfg.freq_minutes}min")
    ends = np.arange(cfg.window - 1, len(grid_times) - cfg.horizon)
    anchors, labels = grid_times[ends], grid_times[ends + cfg.horizon]
    x, y, available, labeled, capacities = [], [], [], [], []
    for n in names:
        grid = prepared.grids[n].reindex(grid_times)
        def sliding(col):
            return np.lib.stride_tricks.sliding_window_view(grid[col].to_numpy(), cfg.window)[ends-cfg.window+1]
        values, age = sliding("available_rate"), sliding("age")
        masks = np.nan_to_num(sliding("observed").astype(float), nan=0)
        # Crucially, no label column participates in this calculation.
        valid = (np.isfinite(values).all(1) & np.isfinite(age).all(1)
                 & (masks.mean(1) >= cfg.min_observed_fraction) & (masks[:, -1] == 1))
        truth = grid.label_rate.to_numpy()[ends + cfg.horizon]
        x.append(encode_windows(np.nan_to_num(values), masks, np.nan_to_num(age, nan=1), labels))
        y.append(np.nan_to_num(truth).astype(np.float32))
        available.append(valid)
        labeled.append(valid & np.isfinite(truth))
        capacities.append(prepared.splits[n]["train"].capacity)
    all_episodes = Episodes(torch.tensor(np.stack(x, 1)), torch.tensor(np.stack(y, 1)),
                            torch.tensor(np.stack(available, 1)), torch.tensor(np.stack(labeled, 1)),
                            torch.tensor(anchors.as_unit("ns").asi8), torch.tensor(labels.as_unit("ns").asi8),
                            names, torch.tensor(capacities))
    keep = all_episodes.available.any(1)
    all_episodes = all_episodes.subset(keep)
    t = all_episodes.label_ns
    boundaries = (pd.Timestamp(cfg.train_end).value, pd.Timestamp(cfg.val_end).value)
    return {"train": all_episodes.subset(t < boundaries[0]).capped(cfg.train_cap),
            "val": all_episodes.subset((t >= boundaries[0]) & (t < boundaries[1])).capped(cfg.eval_cap),
            "test": all_episodes.subset(t >= boundaries[1]).capped(cfg.eval_cap)}


def only_nodes_with_labels(data):
    return data.subset(data.labeled.any(1))


def budgeted_support(train, target, count):
    node = train.names.index(target)
    rows = torch.where(train.labeled[:, node])[0][-count:]
    if len(rows) != count:
        raise ValueError("Insufficient budgeted target labels")
    return train.subset(rows)
