"""All experiment choices are serialized; paths are relative to project root."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    raw_csv: str = "data/raw/aarhus_parking.csv"
    metadata_csv: str = "data/raw/aarhus_parking_address.csv"
    freq_minutes: int = 30
    window: int = 24
    horizon: int = 1
    train_end: str = "2014-09-01 00:00:00"
    val_end: str = "2014-10-01 00:00:00"
    max_fill_steps: int = 1
    min_observed_fraction: float = 0.8
    max_train_invalid_fraction: float = 0.05
    min_train_windows: int = 300
    min_eval_windows: int = 24
    train_cap: int = 0
    eval_cap: int = 0
    fewshot: int = 96
    neighbors: int = 8
    hidden: int = 32
    residual_scale: float = 0.2
    revin: bool = False
    epochs: int = 160
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    patience: int = 40
    synthetic_nodes: int = 48
    condense_steps: int = 160
    reset_interval: int = 20
    real_batch: int = 32
    feature_steps: int = 4
    adjacency_steps: int = 1
    feature_lr: float = 0.01
    adjacency_lr: float = 0.005
    theta_lr: float = 0.02
    adjacency_threshold: float = 0.5
    magnitude_weight: float = 0.05
    anchor_weight: float = 0.01
    adversarial_steps: int = 3
    adversarial_epsilon: float = 0.03
    adversarial_lr: float = 0.1
    vulnerable_fraction: float = 0.25
    spatial_neighbors: int = 3
    spatial_lambda: float = 1.0
    spatial_theta: float = 0.5
    spatial_topk: int = 2
    poi_csv: str | None = None
    poi_cluster_radius_m: float = 200.0
    poi_min_samples: int = 2
    calibration_prior: float = 32.0
    calibration_max_bias: float = 0.15
    methods: tuple[str, ...] = ("full", "random", "gcond", "groc")
    seeds: tuple[int, ...] = (7,)
    torch_threads: int = 1

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        data = json.loads(Path(path).read_text()) if path else {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        for name in ("methods", "seeds"):
            if name in data:
                data[name] = tuple(data[name])
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        import pandas as pd
        positive = ("freq_minutes", "window", "horizon", "fewshot", "neighbors", "hidden",
                    "epochs", "synthetic_nodes", "condense_steps", "reset_interval",
                    "real_batch", "feature_steps", "adjacency_steps", "spatial_topk",
                    "spatial_neighbors", "torch_threads", "adversarial_steps", "poi_min_samples",
                    "patience", "min_train_windows", "min_eval_windows")
        for key in positive:
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        for key in ("min_observed_fraction", "max_train_invalid_fraction", "vulnerable_fraction",
                    "spatial_theta", "adjacency_threshold"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if self.vulnerable_fraction == 0:
            raise ValueError("vulnerable_fraction must be > 0")
        for key in ("learning_rate", "feature_lr", "adjacency_lr", "theta_lr",
                    "residual_scale", "poi_cluster_radius_m"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in ("max_fill_steps", "train_cap", "eval_cap", "adversarial_epsilon",
                    "spatial_lambda", "magnitude_weight", "anchor_weight", "weight_decay",
                    "calibration_prior", "calibration_max_bias"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be non-negative")
        if pd.Timestamp(self.train_end) >= pd.Timestamp(self.val_end):
            raise ValueError("train_end must precede val_end")
        if not self.seeds or not self.methods or set(self.methods) - {"full", "random", "gcond", "groc"}:
            raise ValueError("Need seeds and valid methods: full/random/gcond/groc")

    def path(self, name: str) -> Path:
        p = Path(getattr(self, name))
        return p if p.is_absolute() else ROOT / p

    def asdict(self) -> dict:
        return asdict(self)
