"""In-process agent bus with ordered evidence chain."""
from __future__ import annotations

from collections import OrderedDict
from time import time
import threading

from .base import Agent, Envelope, Message


class AgentBus:
    def __init__(self) -> None:
        self.agents: "OrderedDict[str, Agent]" = OrderedDict()
        self.logs: list[Message] = []
        self._seq = 0
        self._lock = threading.RLock()

    def register(self, agent: Agent) -> None:
        agent.bus = self
        self.agents[agent.name] = agent

    def log(self, src: str, dst: str, action: str, payload: dict,
            status: str = "ok", note: str = "") -> Message:
        with self._lock:
            self._seq += 1
            msg = Message(self._seq, time(), src, dst, action, self._summary(payload), status, note)
            self.logs.append(msg)
            return msg

    def dispatch(self, src: str, dst: str, env: Envelope) -> dict:
        target = self.agents.get(dst)
        if target is None:
            self.log(src, dst, env.action, env.payload, status="error", note="no such agent")
            return {"error": f"agent {dst} 不存在"}
        try:
            result = target.handle(env)
            status = "ok" if "error" not in result else "error"
            self.log(src, dst, env.action, env.payload, status=status,
                     note=str(result.get("error", ""))[:200])
            return result
        except Exception as exc:  # keep the bus alive across agent failures
            self.log(src, dst, env.action, env.payload, status="error", note=repr(exc)[:200])
            return {"error": f"{dst} 执行异常: {exc!r}"}

    @staticmethod
    def _summary(value):
        # Evidence describes inputs without retaining non-JSON DataFrames.
        if isinstance(value, dict):
            return {str(k): AgentBus._summary(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [AgentBus._summary(v) for v in value]
        if hasattr(value, "columns") and hasattr(value, "shape"):
            return {"type": "table", "rows": int(value.shape[0]),
                    "columns": [str(c) for c in value.columns]}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        return str(value)

    def evidence(self, since: int = 0) -> list[dict]:
        with self._lock:
            return [dict(m.__dict__) for m in self.logs if m.seq > since]

