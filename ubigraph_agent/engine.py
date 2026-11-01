"""Read-only adapters to V3. No online training, new fusion, or acquisition."""
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import torch

from ubigraph_core.data import prepare, history_to_feature
from ubigraph_core.utils import seed_all
from ubigraph_st.config import STConfig
from ubigraph_st.data import Episodes
from ubigraph_st.training import STForecaster


class V3Engine:
    def __init__(self, run_dir):
        self.run = Path(run_dir).resolve()
        self.cfg = STConfig.load(self.run/"config.json")
        seed_all(7, self.cfg.torch_threads)
        self.registry = json.loads((self.run/"serving.json").read_text())
        self.protocol_configs = {
            protocol: STConfig.load(self.run / filename)
            for protocol, filename in self.registry.get("protocol_configs", {}).items()
        }
        self.prepared = prepare(self.cfg)
        self.names = sorted(self.prepared.splits)
        self.cache, self.lock = {}, threading.RLock()
        self.default_as_of = "2014-09-30 23:30:00"
        self.minimum = pd.Timestamp(self.cfg.val_end)-pd.Timedelta(30, unit="min")
        self.maximum = max(self.prepared.grids[g].index.max() for g in self.names)
        self.seed_values = sorted(int(k) for k in self.registry["seeds"])

    def normalize_garage(self, name):
        if not isinstance(name, str):
            raise ValueError('Garage code must be a string')
        key = name.strip().upper().replace(" ", "").replace('Garage', "")
        aliases = {"BRUUN": "BRUUNS", "KALK": "KALKVAERKSVEJ", "SCAN": "SCANDCENTER"}
        key = aliases.get(key, key)
        if key not in self.names:
            raise ValueError('Unsupported garage. Choose from:'+", ".join(self.names))
        return key

    def context(self, value=None):
        value = value or {}
        if not isinstance(value, dict):
            raise ValueError('Task context must be an object')
        stamp = pd.Timestamp(value.get("as_of") or self.default_as_of)
        if stamp.tzinfo is not None or stamp != stamp.ceil("30min"):
            raise ValueError('Choose a timezone-free time on a 30-minute grid')
        if stamp < self.minimum or stamp > self.maximum:
            raise ValueError(f"Current model replay range:{self.minimum} to {self.maximum}")
        protocol, seed = value.get("protocol", "full_history"), value.get("seed", 7)
        if protocol not in self.registry["protocol_configs"] or type(seed) is not int or seed not in self.seed_values:
            raise ValueError('Invalid training protocol or random seed')
        revision = self.registry["run_hash"]
        if value.get("model_revision", revision) != revision:
            raise ValueError('Forecast model updated. Start a new conversation to keep model evidence consistent.')
        return {"as_of": str(stamp), "protocol": protocol, "seed": seed, "model_revision": revision}

    def protocol_config(self, context):
        return self.protocol_configs.get(context["protocol"], self.cfg)

    def input_spec(self, context=None):
        cfg = self.protocol_config(context or self.context())
        return {"history_points": cfg.window, "history_frequency_minutes": cfg.freq_minutes,
                "history_hours": cfg.window * cfg.freq_minutes / 60,
                "temporal_neighbors": cfg.neighbors, "history_feature_dim": 3 * cfg.window + 8,
                "node_input_dim": 3 * cfg.window + 10, "hidden_dim": cfg.hidden,
                "synthetic_episodes": cfg.synthetic_nodes,
                "horizon_minutes": cfg.horizon * cfg.freq_minutes}

    def capabilities(self):
        return {"version": "V3", "garages": self.names,
                **self.input_spec(), "horizon_minutes": [30], "protocols": list(self.registry["protocol_configs"]),
                "protocol_inputs": {p: self.input_spec({"protocol": p}) for p in self.registry["protocol_configs"]},
                "default_protocol": "full_history", "experimental_protocols": [],
                "seeds": self.seed_values, "default_as_of": self.default_as_of,
                "as_of_min": str(self.minimum), "as_of_max": str(self.maximum),
                "acquisition_enabled": False, "training_enabled": False,
                "interval_available": False, "data_mode": 'Aarhus historical replay, not live city data',
                "run_hash": self.registry["run_hash"]}

    def forecaster(self, target, context):
        key = (target, context["protocol"], context["seed"])
        with self.lock:
            if key not in self.cache:
                path = self.registry["seeds"][str(context["seed"])][context["protocol"]][target if context["protocol"] == "fewshot" else "all"]
                predictor = STForecaster.load(self.run/path)
                cfg = self.protocol_config(context)
                if any(getattr(predictor.cfg, k) != getattr(cfg, k) for k in ("window", "neighbors", "hidden", "freq_minutes", "horizon")):
                    raise ValueError('Registered input specification does not match forecast checkpoint. Use a matching configuration.')
                self.cache[key] = predictor
            return self.cache[key]

    def history(self, name, context):
        name = self.normalize_garage(name)
        stamp = pd.Timestamp(context["as_of"])
        cfg = self.protocol_config(context)
        times = pd.date_range(end=stamp, periods=cfg.window, freq=f"{cfg.freq_minutes}min")
        # Every feature is obtained from rows at/before as_of. Future label_rate is never read.
        grid = self.prepared.grids[name].reindex(times)
        cap = float(self.prepared.splits[name]["train"].capacity)
        table = pd.DataFrame({"timestamp": times, "available": grid.available_rate.to_numpy()*cap,
                              "capacity": cap, "observed": grid.observed.fillna(False).to_numpy().astype(int),
                              "age": grid.age.to_numpy()})
        return table

    def inspect(self, names, context):
        output = []
        for name in names:
            name = self.normalize_garage(name)
            table = self.history(name, context)
            error = None
            try:
                history_to_feature(table, self.protocol_config(context))
            except ValueError as exc:
                error = str(exc)
            points = [{"time": str(r.timestamp), "available": float(r.available) if np.isfinite(r.available) else None,
                       "observed": bool(r.observed), "age": float(r.age) if np.isfinite(r.age) else None}
                      for r in table.itertuples()]
            output.append({"garage": name, "capacity": float(table.capacity.iloc[0]), "valid": error is None,
                           "quality_note": error or 'History meets V3 input requirements.', "observed_points": int(table.observed.sum()),
                           "history_points": len(table), "history": points,
                           "current_available": points[-1]["available"], "latest_observed": points[-1]["observed"]})
        return {"kind": "history", "as_of": context["as_of"], "items": output, "input_spec": self.input_spec(context),
                "note": 'Historical input is constructed at the selected cutoff on a 30-minute grid.'}

    def spatial(self, target, context, top_k=5):
        target = self.normalize_garage(target)
        f = self.forecaster(target, context)
        e = f.bank.episodes
        j = e.names.index(target)
        scores = f.bundle["prior"][j].tolist()
        rows = sorted([{"garage": n, "similarity": float(scores[i])} for i, n in enumerate(e.names) if n != target],
                      key=lambda r: -r["similarity"])[:top_k]
        return {"kind": "spatial", "target": target, "items": rows,
                "garages": e.names, "matrix": f.bundle["prior"].tolist(),
                "coordinates": [{"garage": n, "latitude": float(self.prepared.metadata.loc[n, "latitude"]),
                                 "longitude": float(self.prepared.metadata.loc[n, "longitude"])} for n in e.names],
                "note": 'The checkpoint’s CoSimHeat geographic prior. Similarity is not prediction confidence or current attention.'}

    def predict(self, targets, context, horizon=30, include_other=True):
        if type(horizon) is not int or horizon != 30:
            raise ValueError('V3 supports only a single 30-minute forecast step. Other horizons were not evaluated.')
        if type(include_other) is not bool:
            raise ValueError('include_other_parking must be a boolean')
        targets = list(dict.fromkeys(self.normalize_garage(n) for n in targets))
        if not targets or len(targets) > 6:
            raise ValueError('Select between 1 and 6 garages per request')
        results = []
        with self.lock:
            for target in targets:
                f = self.forecaster(target, context)
                cfg, names = f.cfg, f.bank.episodes.names
                inspected = self.inspect(names, context)
                x = torch.zeros(1, len(names), 3*cfg.window+8)
                available = torch.zeros(1, len(names), dtype=torch.bool)
                statuses = {}
                for j, name in enumerate(names):
                    if not include_other and name != target:
                        statuses[name] = 'Input masked as requested'
                        continue
                    try:
                        feature, when, cap = history_to_feature(self.history(name, context), cfg)
                    except ValueError as exc:
                        statuses[name] = 'Data requirements not met; input masked'
                        if name == target:
                            raise ValueError(f"{target} does not have history meeting the V3 input requirements: {exc}. Simulated collection does not fill parking observations. Select another cutoff.") from None
                        continue
                    if cap != float(f.bank.episodes.capacities[j]):
                        raise ValueError(f"{name} has a capacity that differs from the V3 checkpoint. Use matching data and models.")
                    x[0, j] = torch.tensor(feature[0])
                    available[0, j] = True
                    statuses[name] = 'Available'
                anchor = pd.Timestamp(context["as_of"])
                when = anchor+pd.Timedelta(30, unit="min")
                query = Episodes(x, torch.zeros(1, len(names)), available, torch.zeros_like(available),
                                 torch.tensor([anchor.value]), torch.tensor([when.value]), names, f.bank.episodes.capacities)
                prediction, detail = f.predict(query, info=True)
                node = names.index(target)
                cap = float(query.capacities[node])
                rate = float(prediction[0, node])
                history = inspected["items"][node]["history"]
                refs = []
                bank = f.bank.episodes
                for k in range(cfg.neighbors):
                    row = int(detail["reference_indices"][0, node, k])
                    if row < 0:
                        continue
                    refs.append({"anchor": str(pd.Timestamp(int(bank.anchor_ns[row]))),
                                 "outcome_time": str(pd.Timestamp(int(bank.label_ns[row]))),
                                 "weight": float(detail["temporal_weights"][0, node, k]),
                                 "delta_rate": float(bank.y[row, node]-bank.x[row, node, cfg.window-1])})
                results.append({"garage": target, "capacity": cap, "as_of": str(anchor), "prediction_time": str(when),
                     "predicted_available": rate*cap, "predicted_rate": rate,
                     "current_available": history[-1]["available"], "change_spaces": rate*cap-history[-1]["available"],
                     "history": history, "spatial_attention": dict(zip(names, detail["spatial_weights"][0, node].tolist())),
                     "spatial_prior": dict(zip(names, f.bundle["prior"][node].tolist())),
                     "temporal_neighbors": refs, "input_status": statuses,
                     "data_quality": [{k: v for k, v in item.items() if k != "history"} for item in inspected["items"]],
                     "memory_kind": f.bank.kind, "prediction_interval": None})
        return {"kind": "forecast", "items": results, **context, "model": "V3 / GroC / CoSimHeat / spatial attention",
                "input_spec": self.input_spec(context),
                "include_other_parking": include_other, "run_hash": self.registry["run_hash"],
                "note": 'Point forecast; attention is not confidence. Own-only input masks external data in the same attention checkpoint, rather than a separately trained local model.'}

    def model_info(self):
        return {"kind": "model_info", "name": "UbiGraph", "model": "V3",
                "input_spec": self.input_spec(), "garages": self.names,
                "seeds": self.seed_values, "data_mode": "Historical replay",
                "training_enabled": False, "acquisition_enabled": False,
                "note": "Forecasts run locally with the included pretrained model. Fleet missions use motion simulation and return action receipts."}
