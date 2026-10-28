"""Transfer agent: borrow predictions from a source city/region to a target
region that has little or no labels, using CoSimHeat-style similarity.

This is the paper's "transferable prediction of parking states in areas with
limited data via model reuse across spatially heterogeneous regions".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Agent, Envelope


class TransferAgent(Agent):
    name = "transfer"

    def __init__(self, bus=None) -> None:
        super().__init__(bus)

    def _register(self) -> None:
        self.handlers = {
            "transfer_predict": self._transfer_predict,
            "transfer_gain": self._transfer_gain,
        }

    @staticmethod
    def _similarity_matrix(source_series: pd.DataFrame, target_series: pd.DataFrame) -> np.ndarray:
        """Cosine similarity between source garages and target garages based on
        normalized temporal profile (mean occupancy by half-hour-of-day)."""
        def profile(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["ts"] = pd.to_datetime(df["timestamp"], format="mixed")
            df["rate"] = df["available"] / df["capacity"].clip(lower=1)
            df["hh"] = df["ts"].dt.hour * 2 + (df["ts"].dt.minute // 30)
            return df.groupby("garagecode")["rate"].mean()  # simplest robust profile
        # Build (garage, halfhour) mean occupancy table
        def grid(df):
            df = df.copy()
            df["ts"] = pd.to_datetime(df["timestamp"], format="mixed")
            df["rate"] = df["available"] / df["capacity"].clip(lower=1)
            df["hh"] = df["ts"].dt.hour * 2 + (df["ts"].dt.minute // 30)
            return df.pivot_table(index="garagecode", columns="hh", values="rate", aggfunc="mean").fillna(0)
        s = grid(source_series)
        t = grid(target_series)
        common = sorted(set(s.columns) & set(t.columns))
        s = s[common].to_numpy(); t = t[common].to_numpy()
        s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-8)
        t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)
        return t @ s.T  # (n_target, n_source)

    def _transfer_predict(self, env: Envelope) -> dict:
        """Borrow the next-step availability rate from the most similar source
        garage (nearest-neighbor transfer). Returns a rate, not slot count."""
        source = env.payload.get("source")
        target = env.payload.get("target")
        if source is None or target is None:
            return {"error": "需要 source 和 target 长表 DataFrame"}
        sim = self._similarity_matrix(source, target)
        target_codes = sorted(target["garagecode"].unique())
        source_codes = sorted(source["garagecode"].unique())
        out = {}
        for i, code in enumerate(target_codes):
            j = int(np.argmax(sim[i]))
            nearest = source_codes[j]
            weight = float(sim[i, j])
            # Borrow source garage's latest observed rate as the cold-start prediction
            src_latest = source[source["garagecode"] == nearest].sort_values("timestamp")
            rate = float(src_latest["available"].iloc[-1] / src_latest["capacity"].iloc[-1])
            out[code] = {"borrowed_from": nearest, "similarity": weight,
                         "predicted_rate": rate}
        return {"method": "cosine-profile nearest-source transfer", "predictions": out}

    def _transfer_gain(self, env: Envelope) -> dict:
        """Given arrays y_true, y_no_transfer (persistence baseline), y_transfer,
        compute MAE-based transfer gain."""
        y = np.asarray(env.payload["y_true"], dtype=float)
        base = np.asarray(env.payload["y_baseline"], dtype=float)
        tr = np.asarray(env.payload["y_transfer"], dtype=float)
        mae_base = float(np.mean(np.abs(y - base)))
        mae_tr = float(np.mean(np.abs(y - tr)))
        gain = (mae_base - mae_tr) / mae_base if mae_base > 0 else 0.0
        return {"mae_baseline": mae_base, "mae_transfer": mae_tr, "transfer_gain": gain}
