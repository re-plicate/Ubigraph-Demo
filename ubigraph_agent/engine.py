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
            raise ValueError("停车场代码必须是字符串")
        key = name.strip().upper().replace(" ", "").replace("停车场", "")
        aliases = {"BRUUN": "BRUUNS", "KALK": "KALKVAERKSVEJ", "SCAN": "SCANDCENTER"}
        key = aliases.get(key, key)
        if key not in self.names:
            raise ValueError("不支持的停车场。可选："+"、".join(self.names))
        return key

    def context(self, value=None):
        value = value or {}
        if not isinstance(value, dict):
            raise ValueError("任务上下文必须是对象")
        stamp = pd.Timestamp(value.get("as_of") or self.default_as_of)
        if stamp.tzinfo is not None or stamp != stamp.ceil("30min"):
            raise ValueError("请选择无时区的30分钟网格时刻")
        if stamp < self.minimum or stamp > self.maximum:
            raise ValueError(f"当前模型回放范围：{self.minimum} 至 {self.maximum}")
        protocol, seed = value.get("protocol", "full_history"), value.get("seed", 7)
        if protocol not in ("fewshot", "full_history") or type(seed) is not int or seed not in self.seed_values:
            raise ValueError("无效的训练协议或随机种子")
        revision = self.registry["run_hash"]
        if value.get("model_revision", revision) != revision:
            raise ValueError("预测模型已更新，请开始新会话，避免混用旧模型结果。")
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
                **self.input_spec(), "horizon_minutes": [30], "protocols": ["full_history", "fewshot"],
                "protocol_inputs": {p: self.input_spec({"protocol": p}) for p in ("full_history", "fewshot")},
                "default_protocol": "full_history", "experimental_protocols": ["fewshot"],
                "seeds": self.seed_values, "default_as_of": self.default_as_of,
                "as_of_min": str(self.minimum), "as_of_max": str(self.maximum),
                "acquisition_enabled": False, "training_enabled": False,
                "interval_available": False, "data_mode": "CityPulse历史回放，不是实时城市数据",
                "run_hash": self.registry["run_hash"]}

    def forecaster(self, target, context):
        key = (target, context["protocol"], context["seed"])
        with self.lock:
            if key not in self.cache:
                path = self.registry["seeds"][str(context["seed"])][context["protocol"]][target if context["protocol"] == "fewshot" else "all"]
                predictor = STForecaster.load(self.run/path)
                cfg = self.protocol_config(context)
                if any(getattr(predictor.cfg, k) != getattr(cfg, k) for k in ("window", "neighbors", "hidden", "freq_minutes", "horizon")):
                    raise ValueError("登记的输入规格与预测检查点不一致，请使用匹配的模型配置。")
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
                           "quality_note": error or "历史符合V3输入要求", "observed_points": int(table.observed.sum()),
                           "history_points": len(table), "history": points,
                           "current_available": points[-1]["available"], "latest_observed": points[-1]["observed"]})
        return {"kind": "history", "as_of": context["as_of"], "items": output, "input_spec": self.input_spec(context),
                "note": "按数据截止时刻构造历史输入，时间间隔为30分钟。"}

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
                "note": "读取V3检查点中实际使用的CoSimHeat地理图先验；相似度不是预测置信度，也不是本次注意力。"}

    def predict(self, targets, context, horizon=30, include_other=True):
        if type(horizon) is not int or horizon != 30:
            raise ValueError("当前V3只支持30分钟单步预测；未执行其他跨度的预测。")
        if type(include_other) is not bool:
            raise ValueError("include_other_parking必须是布尔值")
        targets = list(dict.fromkeys(self.normalize_garage(n) for n in targets))
        if not targets or len(targets) > 6:
            raise ValueError("一次请选择1至6个停车场")
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
                        statuses[name] = "本次按要求屏蔽"
                        continue
                    try:
                        feature, when, cap = history_to_feature(self.history(name, context), cfg)
                    except ValueError as exc:
                        statuses[name] = "数据不满足要求，已屏蔽"
                        if name == target:
                            raise ValueError(f"{target} 当前历史不满足V3要求：{exc}。模拟采集不会补齐停车观测，请选择其他时刻。") from None
                        continue
                    if cap != float(f.bank.episodes.capacities[j]):
                        raise ValueError(f"{name} 数据容量与V3检查点不一致，请使用匹配的数据和模型。")
                    x[0, j] = torch.tensor(feature[0])
                    available[0, j] = True
                    statuses[name] = "可用"
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
                "note": "点预测；注意力不是置信度。仅本场输入指同一个attention检查点屏蔽外部输入，不是另训local模型。"}

    def benchmarks(self):
        data = pd.read_csv(self.run/self.registry.get("metrics_dir", ".")/"metrics_across_seeds.csv")
        data = data[~data["mode"].str.contains("others_missing")]
        data = data.where(pd.notna(data), None)
        # Convert NaN explicitly rather than serializing invalid JSON.
        rows = json.loads(data.to_json(orient="records"))
        for row in rows:
            joint = row["method"] == "groc" and row["mode"] == "attention"
            row["is_primary"] = (row["model_role"] == "primary" if "model_role" in row
                                 else joint and row["protocol"] == "full_history")
            row["is_additional_experiment"] = joint and row["protocol"] == "fewshot"
        primary = [r for r in rows if r["is_primary"]]
        evaluation = self.registry.get("evaluation", {
            p: {"labels_per_seed": 9820, "history_points": self.input_spec({"protocol": p})["history_points"]}
            for p in ("full_history", "fewshot")})
        return {"kind": "benchmarks", "items": rows, "primary_results": primary,
                "additional_experiments": [r for r in rows if r["is_additional_experiment"]],
                "primary_definition": f"以is_primary=true的行作为主方案；当前加载完整历史、{self.input_spec()['history_hours']:g}小时窗口、{self.cfg.neighbors}个时间邻居、GroC+attention。fewshot仅为附加试验；method=full是未凝缩训练数据对照。",
                "seeds": self.seed_values, "test_labels_per_seed": evaluation["full_history"]["labels_per_seed"],
                "evaluation": evaluation,
                "note": "同一协议内使用相同测试标签作对照；完整历史与少样本的评估范围分别列出。MAE与RMSE单位为车位。"}
