import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ubigraph_core.data import history_to_feature
from ubigraph_core.utils import seed_all
from .data import Episodes
from .training import STForecaster


def predict_saved(run, target, histories, protocol="full_history", seed=None, as_of=None):
    run = Path(run).resolve()
    registry = json.loads((run/"serving.json").read_text())
    seed = int(next(iter(registry["seeds"]))) if seed is None else seed
    try:
        relative = registry["seeds"][str(seed)][protocol][target if protocol == "fewshot" else "all"]
    except KeyError as exc:
        raise ValueError("Unknown protocol/seed/target in serving registry") from exc
    predictor = STForecaster.load(run/relative)
    cfg = predictor.cfg
    seed_all(seed, cfg.torch_threads)
    if "garagecode" not in histories or "timestamp" not in histories:
        raise ValueError("History CSV needs garagecode and timestamp")
    if target not in predictor.bank.episodes.names:
        raise ValueError("Unknown target parking lot")
    names = predictor.bank.episodes.names
    if set(histories.garagecode)-set(names):
        raise ValueError("Unexpected parking lot in history CSV")
    times = pd.to_datetime(histories.timestamp, format="mixed")
    as_of = pd.Timestamp(times.max() if as_of is None else as_of)
    if as_of.tzinfo is not None or times.dt.tz is not None:
        raise ValueError("Use the dataset's naive time convention")
    if (times > as_of).any():
        raise ValueError("History includes observations after as_of")
    f = 3*cfg.window+8
    x = torch.zeros(1, len(names), f)
    available = torch.zeros(1, len(names), dtype=torch.bool)
    statuses = {}
    when = as_of+pd.Timedelta(cfg.freq_minutes*cfg.horizon, unit="min")
    for j, name in enumerate(names):
        table = histories.loc[histories.garagecode == name].copy()
        if not len(table):
            statuses[name] = "missing"
            continue
        if pd.Timestamp(pd.to_datetime(table.timestamp, format="mixed").max()) != as_of:
            if name == target:
                raise ValueError("Target history must end at as_of")
            statuses[name] = "stale; masked"
            continue
        try:
            values, prediction_time, capacity = history_to_feature(table, cfg)
        except ValueError as exc:
            if name == target:
                raise
            statuses[name] = "invalid; masked: "+str(exc)
            continue
        if capacity != float(predictor.bank.episodes.capacities[j]):
            raise ValueError(f"Capacity mismatch for {name}")
        x[0, j] = torch.tensor(values[0])
        available[0, j] = True
        statuses[name] = "available"
    j = names.index(target)
    if not available[0, j]:
        raise ValueError("A valid current target history is required")
    query = Episodes(x, torch.zeros(1, len(names)), available, torch.zeros_like(available),
                     torch.tensor([as_of.value]), torch.tensor([when.value]), names, predictor.bank.episodes.capacities)
    prediction, detail = predictor.predict(query, info=True)
    references = {}
    bank = predictor.bank.episodes
    for node, name in enumerate(names):
        entries = []
        for slot in range(cfg.neighbors):
            row = int(detail["reference_indices"][0, node, slot])
            if row < 0:
                continue
            delta = float(bank.y[row, node]-bank.x[row, node, cfg.window-1])
            entries.append({"template_index": row, "anchor": str(pd.Timestamp(int(bank.anchor_ns[row]))),
                            "outcome_time": str(pd.Timestamp(int(bank.label_ns[row]))),
                            "weight": float(detail["temporal_weights"][0, node, slot]), "delta_rate": delta})
        references[name] = entries
    rate, capacity = float(prediction[0, j]), float(query.capacities[j])
    return {"garage": target, "as_of": str(as_of), "prediction_time": str(when),
            "predicted_available": rate*capacity, "predicted_available_rate": rate, "capacity": capacity,
            "protocol": protocol, "seed": seed, "method": "V3 synchronized "+predictor.bundle["provenance"].get("synthetic_kind", "unknown")+" + CoSimHeat spatial feature attention",
            "current_input_status": statuses, "spatial_attention": dict(zip(names, detail["spatial_weights"][0, j].tolist())),
            "spatial_prior": dict(zip(names, predictor.bundle["prior"][j].tolist())),
            "temporal_references": references, "memory_kind": predictor.bank.kind,
            "attention_note": "Attention weights describe this computation, not confidence or causal effects",
            "run_hash": registry["run_hash"], "checkpoint": relative,
            "provenance": predictor.bundle["provenance"]}
