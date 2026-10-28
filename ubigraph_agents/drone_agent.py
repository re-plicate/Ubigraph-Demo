"""Drone fleet agent: wraps the simulated DroneService and the MAVLink adapter.

Plan/start/cancel go through DroneService (which already models energy,
separation, spatiotemporal A*). A parallel MAVLinkAdapter exposes the real
command surface so the same fleet can be pointed at SITL or a vehicle later.
"""
from __future__ import annotations

from pathlib import Path

from ubigraph_drone.mavlink_adapter import MAVLinkAdapter

from .base import Agent, Envelope


class DroneFleetAgent(Agent):
    name = "drone"

    def __init__(self, bus=None, drone_service=None, mavlink_backend: str = "sim",
                 mavlink_url: str | None = None) -> None:
        self.service = drone_service  # a DroneService instance (or None to lazy-create)
        self.mav = MAVLinkAdapter(backend=mavlink_backend, connection_url=mavlink_url)
        super().__init__(bus)

    def _register(self) -> None:
        self.handlers = {
            "state": self._state,
            "plan": self._plan,
            "start": self._start,
            "cancel": self._cancel,
            "mavlink_connect": self._mav_connect,
            "mavlink_command": self._mav_command,
        }

    def _svc(self):
        if self.service is None:
            from ubigraph_drone.service import DroneService
            self.service = DroneService(Path("runtime/agent"))
        return self.service

    def _state(self, env: Envelope) -> dict:
        snap = self._svc().snapshot()
        snap["mavlink"] = self.mav.snapshot()
        return snap

    def _plan(self, env: Envelope) -> dict:
        try:
            return self._svc().plan(
                request_id=env.payload.get("request_id", "req_" + str(int(__import__("time").time()))),
                targets=env.payload["targets"],
                drone_ids=env.payload.get("drone_ids"),
                priority_targets=env.payload.get("priority_targets"),
                dwell_seconds=env.payload.get("dwell_seconds", 3.0),
                return_to_base=env.payload.get("return_to_base", True),
            )
        except Exception as exc:
            return {"error": repr(exc)}

    def _start(self, env: Envelope) -> dict:
        try:
            return self._svc().start(env.payload["plan_id"])
        except Exception as exc:
            return {"error": repr(exc)}

    def _cancel(self, env: Envelope) -> dict:
        try:
            return self._svc().cancel(env.payload["mission_id"],
                                      reason=env.payload.get("reason", "orchestrator cancel"))
        except Exception as exc:
            return {"error": repr(exc)}

    def _mav_connect(self, env: Envelope) -> dict:
        self.mav = MAVLinkAdapter(backend=env.payload.get("backend", "sim"),
                                  connection_url=env.payload.get("url"))
        return self.mav.connect()

    def _mav_command(self, env: Envelope) -> dict:
        action = env.payload.get("action")
        args = env.payload.get("args", {})
        fn = getattr(self.mav, action, None)
        if fn is None or action in ("connect", "disconnect", "snapshot"):
            return {"error": f"unsupported mav action {action}"}
        return fn(**args)
