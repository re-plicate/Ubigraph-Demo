"""Apply a fixed evaluation cohort without changing historical observations."""
import numpy as np
import torch

from .data import Episodes


def apply_layout(data, path):
    with np.load(path, allow_pickle=False) as layout:
        names = layout["names"].tolist()
        anchors = layout["anchor_ns"]
        available = torch.tensor(layout["available"], dtype=torch.bool)
        labeled = torch.tensor(layout["labeled"], dtype=torch.bool)
    if data.names != names or not len(data):
        raise ValueError("Evaluation parking nodes differ from the prepared data")
    ids = np.searchsorted(data.anchor_ns.numpy(), anchors)
    if (ids >= len(data)).any() or not np.array_equal(data.anchor_ns.numpy()[ids], anchors):
        raise ValueError("An evaluation origin is missing from the prepared data")
    part = data.subset(ids)
    if available.shape != part.available.shape or labeled.shape != part.labeled.shape:
        raise ValueError("Evaluation masks have an unexpected shape")
    if (available & ~part.available).any() or (labeled & ~part.labeled).any():
        raise ValueError("Evaluation layout cannot introduce missing inputs or labels")
    fields = dict(part.__dict__)
    fields.update(available=available, labeled=labeled)
    return Episodes(**fields)
