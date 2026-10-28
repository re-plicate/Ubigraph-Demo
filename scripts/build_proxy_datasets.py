"""Generate realistic proxy datasets for SFpark (SF) and London.

Public downloads are blocked on this machine (SSL/Socrata). These proxies share
the long-format schema (garagecode, timestamp, available, capacity) but with
distinct daily/weekly profiles so cross-city transfer is meaningful. Replace
these files with real CSVs later; the loader and experiments pick them up
unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def _profile(garage: str, n_garages: int, seed: int, city: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # 30 days of half-hourly observations
    periods = 30 * 48
    idx = pd.date_range("2024-01-01", periods=periods, freq="30min")
    gidx = hash(garage) % 100
    # City-specific base occupancy: SF slightly busier midday, London commuter peaks
    base = 0.55 + 0.1 * np.sin(np.arange(periods) / 48 * 2 * np.pi)
    if city == "sfpark":
        midday = 0.15 * np.exp(-((np.arange(periods) % 48 - 28) ** 2) / 30)
    else:  # london
        midday = 0.18 * np.exp(-((np.arange(periods) % 48 - 18) ** 2) / 20)
        midday += 0.15 * np.exp(-((np.arange(periods) % 48 - 38) ** 2) / 20)
    weekly = -0.1 * (np.array([d.weekday() >= 5 for d in idx])).astype(float)
    noise = rng.normal(0, 0.04, periods)
    rate = np.clip(base + midday + weekly + noise, 0.05, 0.95)
    capacity = rng.integers(200, 600)
    available = (capacity * (1 - rate)).round()
    return pd.DataFrame({"garagecode": garage, "timestamp": idx,
                         "available": available, "capacity": capacity,
                         "observed": True})


def build() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    sf_garages = ["SF_SOMA", "SF_MISSION", "SF_FISHERMAN", "SF_DT", "SF_RINCON"]
    ld_garages = ["LD_WESTMINSTER", "LD_CITY", "LD_KINGSX", "LD_BOROUGH", "LD_CANARY"]
    sf = pd.concat([_profile(g, len(sf_garages), 100 + i, "sfpark") for i, g in enumerate(sf_garages)])
    ld = pd.concat([_profile(g, len(ld_garages), 200 + i, "london") for i, g in enumerate(ld_garages)])
    sf.to_csv(RAW / "sfpark.csv", index=False)
    ld.to_csv(RAW / "london_parking.csv", index=False)
    print(f"sfpark.csv: {len(sf)} rows, {sf.garagecode.nunique()} garages")
    print(f"london_parking.csv: {len(ld)} rows, {ld.garagecode.nunique()} garages")


if __name__ == "__main__":
    build()
