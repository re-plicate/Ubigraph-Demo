"""Human-in-the-loop agent.

Implements the project doc's trigger function:
    Trigger = 1 if U > t_u or R > t_r or C > t_c else 0
where U = prediction uncertainty (residual spread from temporal references),
R = data-quality risk (missing/stale garages), C = cost of a wrong call.
"""
from __future__ import annotations

from .base import Agent, Envelope


class HumanInLoopAgent(Agent):
    name = "hic"

    def __init__(self, bus=None, t_u: float = 0.15, t_r: float = 0.3, t_c: float = 0.5) -> None:
        self.t_u, self.t_r, self.t_c = t_u, t_r, t_c
        self.feedback: list[dict] = []
        super().__init__(bus)

    def _register(self) -> None:
        self.handlers = {
            "evaluate": self._evaluate,
            "feedback": self._feedback,
            "feedback_log": self._feedback_log,
        }

    def _evaluate(self, env: Envelope) -> dict:
        pred = env.payload.get("prediction", {})
        if not isinstance(pred, dict):
            # Allow simple numeric inputs: prediction=rate, uncertainty=U
            pred = {}
        refs = pred.get("temporal_references", {})
        deltas = []
        if isinstance(refs, dict):
            for entries in refs.values():
                if isinstance(entries, list):
                    for e in entries:
                        if isinstance(e, dict):
                            deltas.append(float(e.get("delta_rate", 0.0)))
        if deltas:
            import statistics
            u = statistics.pstdev(deltas)
        else:
            u = float(env.payload.get("uncertainty", 0.0))
        status = pred.get("current_input_status", {})
        if isinstance(status, dict):
            missing = sum(1 for v in status.values() if v != "available")
            r = missing / max(1, len(status))
        else:
            r = float(env.payload.get("risk", 0.0))
        c = float(env.payload.get("cost", 0.2))
        trigger = (u > self.t_u) or (r > self.t_r) or (c > self.t_c)
        return {"uncertainty": round(u, 4), "risk": round(r, 4), "cost": c,
                "trigger_human": bool(trigger),
                "reason": ("high_uncertainty" if u > self.t_u else
                           "stale_inputs" if r > self.t_r else
                           "high_cost" if c > self.t_c else "auto_ok")}

    def _feedback(self, env: Envelope) -> dict:
        event = {
            "task_id": env.payload.get("task_id"),
            "action": env.payload.get("action", "approve"),  # approve|edit|reject
            "old": env.payload.get("old_value"),
            "new": env.payload.get("new_value"),
            "reason": env.payload.get("reason", ""),
        }
        self.feedback.append(event)
        return {"recorded": True, "n_feedback": len(self.feedback), "event": event}

    def _feedback_log(self, env: Envelope) -> dict:
        return {"count": len(self.feedback), "events": self.feedback[-50:]}
