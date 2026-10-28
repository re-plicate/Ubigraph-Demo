"""Unified parking-time-series loader.

Normalizes CityPulse (Aarhus), SFpark, and London Parking to a common long format:

    garagecode (str), timestamp (naive datetime), available (float),
    capacity (float), observed (bool)

and a metadata frame indexed by garagecode with latitude/longitude.

Datasets are optional: if a raw file is missing, ``load_*`` raises FileNotFoundError
with a download hint so the experiment runner can skip gracefully.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def _out_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["observed"] = df.get("observed", True)
    return df[["garagecode", "timestamp", "available", "capacity", "observed"]]


def load_aarhus(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CityPulse / Aarhus. Already ships in data/raw."""
    raw = Path(raw_dir) if raw_dir else RAW
    df = pd.read_csv(raw / "aarhus_parking.csv")
    meta = pd.read_csv(raw / "aarhus_parking_address.csv")
    df = df.rename(columns={c: c.lower() for c in df.columns})
    meta = meta.rename(columns={c: c.lower() for c in meta.columns}).set_index("garagecode")
    return _out_long(df), meta


def load_sfpark(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SFpark (San Francisco). Expected columns: garagelongtype/count,
    or the common export `facilityname,occupancydatetime,occupancyavailable,totalspaces`."""
    raw = Path(raw_dir) if raw_dir else RAW
    cand = list(raw.glob("*sfpark*")) + list(raw.glob("*sf_park*"))
    if not cand:
        raise FileNotFoundError(
            "SFpark CSV 未找到。请从 https://data.sfgov.org/Transportation/SFpark-Parking-Occupancy/ 下载，"
            "存为 data/raw/sfpark.csv")
    df = pd.read_csv(cand[0])
    # normalize column names
    df.columns = [c.lower() for c in df.columns]
    if "occupancydatetime" in df.columns:
        df = df.rename(columns={"facilityname": "garagecode",
                                "occupancydatetime": "timestamp",
                                "occupancyavailable": "available",
                                "totalspaces": "capacity"})
    df = df[["garagecode", "timestamp", "available", "capacity"]]
    meta = pd.DataFrame(index=sorted(df.garagecode.unique()),
                        columns=["latitude", "longitude"])
    return _out_long(df), meta


def load_london(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """London Parking. Common TfL/open format: `Bays,Date,Status,Source`."""
    raw = Path(raw_dir) if raw_dir else RAW
    cand = list(raw.glob("*london*park*")) + list(raw.glob("*tfl*park*"))
    if not cand:
        raise FileNotFoundError(
            "London Parking CSV 未找到。请从 https://data.london.gov.uk/dataset/car-parking 下载，"
            "存为 data/raw/london_parking.csv")
    df = pd.read_csv(cand[0])
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df = df.rename(columns={"bays": "garagecode", "date": "timestamp"})
        # Status-like: occupied/free -> derive available
        if "status" in df.columns:
            df["available"] = (df["status"].str.lower().str.contains("free", na=False)).astype(float)
            df["capacity"] = 1.0
    return _out_long(df), pd.DataFrame(index=sorted(df.garagecode.unique()),
                                       columns=["latitude", "longitude"])


DATASETS = {
    "aarhus": load_aarhus,
    "sfpark": load_sfpark,
    "london": load_london,
}


def available_datasets() -> dict[str, bool]:
    raw = RAW
    return {
        "aarhus": (raw / "aarhus_parking.csv").exists(),
        "sfpark": bool(list(raw.glob("*sfpark*")) or list(raw.glob("*sf_park*"))),
        "london": bool(list(raw.glob("*london*park*"))),
    }
