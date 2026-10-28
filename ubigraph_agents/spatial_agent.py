"""Spatial agent: CoSimHeat similarity between garages."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import Agent, Envelope


class SpatialAgent(Agent):
    name = "spatial"

    def __init__(self, bus=None, run="models/parking") -> None:
        self.run = Path(run)
        self._prior = None
        super().__init__(bus)

    def _register(self) -> None:
        self.handlers = {
            "similarity": self._similarity,
            "neighbors": self._neighbors,
        }

    def _load(self) -> tuple[list[str], np.ndarray]:
        if self._prior is None:
            from ubigraph_st.training import STForecaster
            registry = json.loads((self.run / "serving.json").read_text())
            seed = str(min(int(k) for k in registry["seeds"]))
            path = registry["seeds"][seed]["full_history"]["all"]
            predictor = STForecaster.load(self.run / path)
            names = list(predictor.bank.episodes.names)
            sim = np.asarray(predictor.bundle["prior"], dtype=float)
            if sim.shape != (len(names), len(names)) or not np.isfinite(sim).all():
                raise ValueError("检查点空间关系矩阵无效")
            self._prior = (names, sim)
        return self._prior

    def _similarity(self, env: Envelope) -> dict:
        try:
            names, sim = self._load()
            g = env.payload.get("garage")
            if g is not None and g not in names:
                return {"error": f"未知停车场: {g}", "garages": names}
            if g is None:
                return {"garages": names, "similarity": sim.round(4).tolist(),
                        "source": "registered full_history checkpoint prior"}
            row = sim[names.index(g)]
            ranked = sorted([(names[i], float(row[i])) for i in range(len(names))
                             if names[i] != g], key=lambda x: -x[1])
            return {"garage": g, "most_similar": ranked,
                    "note": "CoSimHeat heat-fusion similarity"}
        except Exception as exc:
            return {"error": repr(exc)}

    def _neighbors(self, env: Envelope) -> dict:
        topk = int(env.payload.get("topk", 3))
        result = self._similarity(Envelope("similarity", {"garage": env.payload.get("garage")}))
        if "most_similar" not in result:
            return result
        result["topk"] = result["most_similar"][:topk]
        return result
