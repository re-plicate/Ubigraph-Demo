"""Synchronous analysis and optional simulation dispatch.

No acquisition ingestion or second prediction is implemented here.
"""
from __future__ import annotations

import uuid

from .base import Agent, Envelope


class OrchestratorAgent(Agent):
    name = "orchestrator"

    def _register(self) -> None:
        self.handlers = {
            "analyze": self._analyze,
            "run_mission": self._run_mission,
        }

    def _analyze(self, env: Envelope) -> dict:
        garage = env.payload["garage"]
        histories = env.payload["histories"]
        seed = env.payload.get("seed", 7)

        # 1. spatial prior
        spatial = self.call("spatial", Envelope("neighbors", {"garage": garage, "topk": 3}))

        # 2. temporal prediction
        pred = self.call("temporal", Envelope("predict",
                         {"garage": garage, "histories": histories, "seed": seed}))

        if "error" in pred:
            return {"garage": garage, "error": pred["error"], "prediction": pred,
                    "spatial_prior": spatial, "status": "prediction_failed",
                    "gate": {"trigger_human": True, "reason": "prediction_failed"}, "round": 1}

        # 3. human-in-the-loop gate
        gate = self.call("hic", Envelope("evaluate", {"prediction": pred,
                                                        "cost": env.payload.get("cost", 0.2)}))

        return {"garage": garage, "spatial_prior": spatial, "prediction": pred,
                "gate": gate, "round": 1}

    def _run_mission(self, env: Envelope) -> dict:
        """Analyze, then optionally submit a simulation mission; no reforecast."""
        first = self._analyze(env)
        out = {"first_round": first, "status": "analysis_only", "measurement_ingested": False}
        if "error" in first:
            return {**out, "error": first["error"], "status": "prediction_failed"}
        if first["gate"].get("trigger_human") and env.payload.get("allow_drone", True):
            targets = env.payload.get("drone_targets") or [env.payload["garage"]]
            plan = self.call("drone", Envelope("plan",
                             {"request_id": env.payload.get("request_id") or "orch_" + uuid.uuid4().hex[:20],
                              "targets": targets, "dwell_seconds": 3.0}))
            out["drone_plan"] = plan
            if "error" in plan:
                return {**out, "error": plan["error"], "status": "planning_failed"}
            if not plan.get("feasible"):
                return {**out, "status": "plan_infeasible"}
            if plan.get("feasible"):
                mission = self.call("drone", Envelope("start", {"plan_id": plan["id"]}))
                out["drone_mission"] = mission
                out["status"] = "dispatch_failed" if "error" in mission else "dispatched"
                if "error" in mission:
                    out["error"] = mission["error"]
                # In a real run the orchestrator would wait for receipts; here we
                # record that the补采 was dispatched (evidence chain).
                out["note"] = "模拟任务提交结果见 drone_mission；无真实测量入库，未进行二次预测"
        return out
