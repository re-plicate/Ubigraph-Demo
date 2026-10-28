"""Temporal agent: wraps the saved V3 spatiotemporal forecaster."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import Agent, Envelope


class TemporalAgent(Agent):
    name = "temporal"

    def __init__(self, bus=None, run="models/parking") -> None:
        self.run = Path(run)
        self._predictor = None
        super().__init__(bus)

    def _register(self) -> None:
        self.handlers = {
            "predict": self._predict,
            "names": self._names,
        }

    def _names(self, env: Envelope) -> dict:
        try:
            from ubigraph_st.serving import predict_saved  # noqa: F401
            registry = self.run / "serving.json"
            if not registry.exists():
                return {"garages": [], "error": "serving.json 不存在"}
            import json
            reg = json.loads(registry.read_text())
            seeds = list(reg.get("seeds", {}).keys())
            names = sorted({name for entry in reg.get("seeds", {}).values()
                            for name in entry.get("fewshot", {})})
            return {"garages": names, "seeds": seeds}
        except Exception as exc:
            return {"error": repr(exc)}

    def _predict(self, env: Envelope) -> dict:
        from ubigraph_st.serving import predict_saved
        garage = env.payload.get("garage")
        histories = env.payload.get("histories")
        protocol = env.payload.get("protocol", "full_history")
        seed = env.payload.get("seed", 7)
        as_of = env.payload.get("as_of")
        if garage is None:
            return {"error": "predict 需要 garage"}
        if histories is None:
            return {"error": "predict 需要 histories (DataFrame 或 CSV 路径)"}
        if isinstance(histories, (str, Path)):
            histories = pd.read_csv(histories)
        try:
            result = predict_saved(self.run, garage, histories,
                                   protocol=protocol, seed=seed, as_of=as_of)
            return result
        except Exception as exc:
            return {"garage": garage, "error": repr(exc)}
