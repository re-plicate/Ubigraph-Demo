"""FastAPI surface for the CityClaw multi-agent system.

Run with::

    uvicorn ubigraph_agents.api:app --port 8766

Endpoints mirror the project doc's serving layer:
  GET  /api/agents                 list registered agents
  POST /api/agents/{name}/call     call one agent directly
  POST /api/orchestrator/analyze   run the analysis->gate loop
  POST /api/orchestrator/mission   run analyze->drone补采 loop
  GET  /api/evidence               the decision evidence chain
  POST /api/hic/feedback           record human feedback
  GET  /api/drone/mavlink          MAVLink adapter snapshot
  POST /api/drone/mavlink/command  issue arm/takeoff/goto/land
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .base import Envelope
from .system import build_system

app = FastAPI(title="CityClaw Multi-Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ROOT = Path(__file__).resolve().parents[1]
_RUN = str((_ROOT / os.environ.get("UBIGRAPH_RUN", "models/parking")).resolve())
_bus_lock = threading.Lock()
_bus = None


def _bus_lazy():
    global _bus
    with _bus_lock:
        if _bus is None:
            _bus = build_system(run=_RUN,
                                mavlink_backend=os.environ.get("MAVLINK_BACKEND", "sim"),
                                mavlink_url=os.environ.get("MAVLINK_URL"))
    return _bus


class CallReq(BaseModel):
    action: str
    payload: dict = {}


class AnalyzeReq(BaseModel):
    garage: str
    seed: int = 7
    histories: str | None = None  # path to CSV; if None uses examples/history_24.csv
    cost: float = 0.2


class MissionReq(AnalyzeReq):
    request_id: str | None = None
    allow_drone: bool = True
    drone_targets: list[str] | None = None


class FeedbackReq(BaseModel):
    task_id: str | None = None
    action: str = "approve"
    old_value: Any = None
    new_value: Any = None
    reason: str = ""


class MavCommandReq(BaseModel):
    action: str
    args: dict = {}


@app.get("/api/health")
def health():
    return {"status": "ok", "agents": list(_bus_lazy().agents)}


@app.get("/api/agents")
def list_agents():
    return {"agents": [
        {"name": n, "handlers": sorted(a.handlers)}
        for n, a in _bus_lazy().agents.items()
    ]}


@app.post("/api/agents/{name}/call")
def call_agent(name: str, req: CallReq):
    bus = _bus_lazy()
    if name not in bus.agents:
        raise HTTPException(404, f"unknown agent {name}")
    result = bus.dispatch("api", name, Envelope(req.action, req.payload))
    return result


@app.post("/api/orchestrator/analyze")
def analyze(req: AnalyzeReq):
    bus = _bus_lazy()
    histories = req.histories or str(Path(_RUN).parent.parent / "examples" / "history_24.csv")
    import pandas as pd
    df = pd.read_csv(histories)
    return bus.dispatch("api", "orchestrator", Envelope("analyze", {
        "garage": req.garage, "histories": df, "seed": req.seed, "cost": req.cost,
    }))


@app.post("/api/orchestrator/mission")
def mission(req: MissionReq):
    bus = _bus_lazy()
    import pandas as pd
    df = pd.read_csv(req.histories or Path(_RUN).parent.parent / "examples" / "history_24.csv")
    return bus.dispatch("api", "orchestrator", Envelope("run_mission", {
        "garage": req.garage, "histories": df, "seed": req.seed,
        "cost": req.cost, "allow_drone": req.allow_drone,
        "drone_targets": req.drone_targets, "request_id": req.request_id,
    }))


@app.get("/api/evidence")
def evidence(since: int = 0):
    return {"count": len(_bus_lazy().logs), "chain": _bus_lazy().evidence(since)}


@app.post("/api/hic/feedback")
def feedback(req: FeedbackReq):
    return _bus_lazy().dispatch("api", "hic", Envelope("feedback", req.model_dump()))


@app.get("/api/drone/mavlink")
def mavlink_state():
    return _bus_lazy().dispatch("api", "drone", Envelope("state", {}))["mavlink"]


@app.post("/api/drone/mavlink/command")
def mavlink_command(req: MavCommandReq):
    return _bus_lazy().dispatch("api", "drone",
                                Envelope("mavlink_command", {"action": req.action, "args": req.args}))
