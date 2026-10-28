"""Causal 30-minute snapshots, physical validation, and historical-window nodes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .utils import file_hash, uniform_indices, write_json


@dataclass
class Windows:
    x: np.ndarray
    y: np.ndarray
    anchor_ns: np.ndarray
    label_ns: np.ndarray
    garage: str
    capacity: float

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, index) -> "Windows":
        return Windows(self.x[index], self.y[index], self.anchor_ns[index],
                       self.label_ns[index], self.garage, self.capacity)

    def capped(self, cap: int) -> "Windows":
        return self.subset(uniform_indices(len(self), cap))

    def save(self, path: Path) -> None:
        np.savez_compressed(path, x=self.x, y=self.y, anchor_ns=self.anchor_ns,
                            label_ns=self.label_ns, garage=self.garage, capacity=self.capacity)


@dataclass
class Prepared:
    splits: dict[str, dict[str, Windows]]
    grids: dict[str, pd.DataFrame]
    metadata: pd.DataFrame
    audit: dict


def calendar_features(times) -> np.ndarray:
    t = pd.DatetimeIndex(times)
    hour = np.asarray(t.hour + t.minute / 60)
    weekday = np.asarray(t.dayofweek)
    year_day = np.asarray(t.dayofyear)
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7),
        np.sin(2 * np.pi * year_day / 366), np.cos(2 * np.pi * year_day / 366),
        (weekday >= 5).astype(float), np.ones(len(t)),
    ]).astype(np.float32)


def encode_windows(values: np.ndarray, observed: np.ndarray, age: np.ndarray, times) -> np.ndarray:
    return np.concatenate([values, observed.astype(np.float32), age, calendar_features(times)],
                          axis=1).astype(np.float32)


def causal_grid(rows: pd.DataFrame, capacity: float, cfg: Config) -> pd.DataFrame:
    """Records in (previous grid, grid] may only appear at grid; no nearest rounding."""
    freq = pd.Timedelta(cfg.freq_minutes, unit="min")
    rows = rows.sort_values(["updatetime", "_id"]).copy()
    rows["grid_time"] = rows.updatetime.dt.ceil(freq)
    rows = rows.drop_duplicates("grid_time", keep="last").set_index("grid_time")
    grid = pd.date_range(rows.index.min(), rows.index.max(), freq=freq)
    rows = rows.reindex(grid)
    valid = (rows.vehiclecount.ge(0) & rows.vehiclecount.le(capacity)
             & rows.totalspaces.eq(capacity) & rows.updatetime.notna())
    raw_free = ((capacity - rows.vehiclecount) / capacity).where(valid)
    filled = raw_free.ffill(limit=cfg.max_fill_steps) if cfg.max_fill_steps else raw_free
    source = rows.updatetime.where(valid)
    source = source.ffill(limit=cfg.max_fill_steps) if cfg.max_fill_steps else source
    # source is the original event time, including its sub-minute precision.
    age_minutes = (pd.Series(grid, index=grid) - source).dt.total_seconds() / 60
    age_scale = cfg.freq_minutes * (cfg.max_fill_steps + 1)
    return pd.DataFrame({"available_rate": filled, "label_rate": raw_free,
                         "observed": valid.astype(bool),
                         "age": (age_minutes / age_scale).clip(0, 1),
                         "source_time": source, "capacity": capacity}, index=grid)


def grid_to_windows(grid: pd.DataFrame, garage: str, capacity: float, cfg: Config) -> Windows:
    length, horizon = cfg.window, cfg.horizon
    values = grid.available_rate.to_numpy()
    observed = grid.observed.to_numpy()
    ages = grid.age.to_numpy()
    labels = grid.label_rate.to_numpy()
    starts = np.arange(length - 1, len(grid) - horizon)
    if not len(starts):
        raise ValueError(f"{garage}: too few grid points")
    windows = np.lib.stride_tricks.sliding_window_view(values, length)[starts - length + 1]
    masks = np.lib.stride_tricks.sliding_window_view(observed, length)[starts - length + 1]
    age_windows = np.lib.stride_tricks.sliding_window_view(ages, length)[starts - length + 1]
    keep = (np.isfinite(windows).all(axis=1) & np.isfinite(age_windows).all(axis=1)
            & (masks.mean(axis=1) >= cfg.min_observed_fraction) & observed[starts]
            & np.isfinite(labels[starts + horizon]))
    starts = starts[keep]
    label_times = grid.index[starts + horizon]
    features = encode_windows(windows[keep], masks[keep], age_windows[keep], label_times)
    return Windows(features, labels[starts + horizon].astype(np.float32),
                   grid.index[starts].as_unit("ns").asi8, label_times.as_unit("ns").asi8,
                   garage, float(capacity))


def split_windows(w: Windows, cfg: Config) -> dict[str, Windows]:
    train_end = pd.Timestamp(cfg.train_end).value
    val_end = pd.Timestamp(cfg.val_end).value
    # Split by prediction target, never by shuffled window index.
    return {
        "train": w.subset(w.label_ns < train_end),
        "val": w.subset((w.label_ns >= train_end) & (w.label_ns < val_end)),
        "test": w.subset(w.label_ns >= val_end),
    }


def fewshot_windows(w: Windows, count: int) -> Windows:
    """Latest supervised training windows. The budget is WINDOWS, not independent points."""
    if len(w) < count:
        raise ValueError(f"{w.garage}: {len(w)} training windows < fewshot={count}")
    return w.subset(np.arange(len(w) - count, len(w)))


def prepare(cfg: Config, output: Path | None = None) -> Prepared:
    raw_path, metadata_path = cfg.path("raw_csv"), cfg.path("metadata_csv")
    raw = pd.read_csv(raw_path)
    required = {"vehiclecount", "updatetime", "_id", "totalspaces", "garagecode", "streamtime"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing raw columns: {required - set(raw.columns)}")
    raw["updatetime"] = pd.to_datetime(raw.updatetime, format="mixed", errors="raise")
    if raw.updatetime.dt.tz is not None:
        raise ValueError("Expected naive source time; explicitly handle timezone before importing")
    meta = pd.read_csv(metadata_path).set_index("garagecode").sort_index()
    if set(meta.index) != set(raw.garagecode) or meta.index.has_duplicates:
        raise ValueError("Garage IDs in metadata and observations must match exactly")
    for col in ("latitude", "longitude"):
        if not np.isfinite(meta[col]).all():
            raise ValueError(f"Non-finite {col}")
    invalid = ~((raw.vehiclecount >= 0) & (raw.vehiclecount <= raw.totalspaces)
                & (raw.totalspaces > 0))
    duplicates = int(raw.duplicated(["garagecode", "updatetime"]).sum())
    # Keep last _id at a duplicated event time; report count instead of silently ignoring it.
    raw = raw.sort_values(["updatetime", "_id"]).drop_duplicates(["garagecode", "updatetime"], keep="last")
    unique_times = raw.updatetime.drop_duplicates().sort_values()
    audit = {
        "raw_rows": int(len(invalid)), "deduplicated_rows": len(raw),
        "duplicate_garage_timestamps": duplicates, "raw_invalid_rows": int(invalid.sum()),
        "distinct_event_times": len(unique_times),
        "first_event": str(unique_times.min()), "last_event": str(unique_times.max()),
        "median_gap_minutes": float(unique_times.diff().dt.total_seconds().median() / 60),
        "max_gap_hours": float(unique_times.diff().dt.total_seconds().max() / 3600),
        "raw_sha256": file_hash(raw_path), "metadata_sha256": file_hash(metadata_path),
        "time_axis": "updatetime; original timezone unspecified; right-closed causal ceil bins",
        "train_end_exclusive": cfg.train_end, "val_end_exclusive": cfg.val_end,
        "garages": [],
    }
    splits, grids = {}, {}
    for garage, rows in raw.groupby("garagecode", sort=True):
        train_raw = rows[rows.updatetime < pd.Timestamp(cfg.train_end)]
        capacities = train_raw.totalspaces.unique()
        if len(capacities) != 1 or capacities[0] <= 0:
            raise ValueError(f"{garage}: need one positive training-period capacity, got {capacities}")
        capacity = float(capacities[0])
        bad_train = ~train_raw.vehiclecount.between(0, capacity)
        invalid_fraction = float(bad_train.mean())
        grid = causal_grid(rows, capacity, cfg)
        w = grid_to_windows(grid, garage, capacity, cfg)
        all_splits = split_windows(w, cfg)
        reason = None
        if invalid_fraction > cfg.max_train_invalid_fraction:
            reason = "training-period out-of-capacity fraction exceeds threshold"
        elif len(all_splits["train"]) < max(cfg.min_train_windows, cfg.fewshot):
            reason = "too few valid training windows"
        # No test/validation outcome can decide the source set. Fail rather than adapt it.
        included = reason is None
        entry = {
            "garage": garage, "capacity": capacity, "raw_rows": len(rows),
            "invalid_all": int((~rows.vehiclecount.between(0, capacity)).sum()),
            "invalid_train": int(bad_train.sum()), "train_raw_rows": len(train_raw),
            "invalid_train_fraction": invalid_fraction,
            "valid_grid_observations": int(grid.observed.sum()), "grid_points": len(grid),
            "all_valid_windows": len(w), "windows": {k: len(v) for k, v in all_splits.items()},
            "included": included, "exclusion_reason": reason,
        }
        if included:
            for split in ("val", "test"):
                if len(all_splits[split]) < cfg.min_eval_windows:
                    raise ValueError(f"{garage}: too few {split} windows; protocol cannot be evaluated")
            splits[garage] = all_splits
        grids[garage] = grid
        audit["garages"].append(entry)
    if len(splits) < 3:
        raise ValueError("Need at least three eligible garages for held-out-source transfer")
    if output:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "data_audit.json", audit)
        flat = [{**row, **{f"windows_{k}": v for k, v in row["windows"].items()}}
                for row in audit["garages"]]
        pd.DataFrame(flat).drop(columns="windows").to_csv(output / "data_audit.csv", index=False)
        for garage, parts in splits.items():
            folder = output / garage
            folder.mkdir(exist_ok=True)
            for name, part in parts.items():
                part.save(folder / f"{name}.npz")
        pd.DataFrame({g: grid.label_rate for g, grid in grids.items()}).to_csv(
            output / "aligned_observed_rates.csv", index_label="grid_time")
    return Prepared(splits, grids, meta, audit)


def history_to_feature(table: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, pd.Timestamp, float]:
    """Public inference contract: exactly L successive aligned snapshots, oldest first."""
    required = {"timestamp", "available", "capacity"}
    if not required.issubset(table):
        raise ValueError(f"History needs columns {sorted(required)}")
    if len(table) != cfg.window:
        raise ValueError(f"Expected exactly {cfg.window} historical grid rows")
    t = pd.DatetimeIndex(pd.to_datetime(table.timestamp, format="mixed"))
    step = pd.Timedelta(cfg.freq_minutes, unit="min")
    if t.tz is not None or not (np.diff(t.as_unit("ns").asi8) == step.value).all():
        raise ValueError("History times must be naive, strictly consecutive configured grid times")
    if not t.equals(t.ceil(step)):
        raise ValueError("History timestamps must lie on the configured grid")
    capacities = table.capacity.to_numpy(float)
    if not np.isfinite(capacities).all() or not (capacities == capacities[0]).all() or capacities[0] <= 0:
        raise ValueError("History capacity must be constant and positive")
    capacity = float(capacities[0])
    values = table.available.to_numpy(float) / capacity
    observed = table.get("observed", pd.Series(np.ones(len(t)))).to_numpy(float)
    age = table.get("age", pd.Series(np.zeros(len(t)))).to_numpy(float)
    if (not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()
            or not np.isin(observed, [0, 1]).all() or observed.mean() < cfg.min_observed_fraction
            or observed[-1] != 1 or not np.isfinite(age).all() or ((age < 0) | (age > 1)).any()):
        raise ValueError("History violates physical range, observation-mask or age requirements")
    prediction_time = t[-1] + cfg.horizon * step
    x = encode_windows(values[None], observed[None], age[None], [prediction_time])
    return x, prediction_time, capacity
