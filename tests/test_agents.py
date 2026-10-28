"""Smoke tests for the multi-agent layer. Run with `pytest tests/test_agents.py`."""
from __future__ import annotations

from pathlib import Path

import pytest

from ubigraph_agents.base import Envelope
from ubigraph_agents.bus import AgentBus
from ubigraph_agents.system import build_system


def test_bus_routes_messages():
    bus = AgentBus()

    class Echo:
        name = "echo"
        def handle(self, env):
            return {"got": env.action, "n": len(env.payload)}

    bus.register(Echo())
    out = bus.dispatch("t", "echo", Envelope("ping", {"x": 1}))
    assert out == {"got": "ping", "n": 1}
    assert bus.logs[-1].status == "ok"


def test_build_system_has_all_agents():
    bus = build_system(run="models/parking")
    for name in ("temporal", "spatial", "transfer", "detection", "drone", "hic", "orchestrator"):
        assert name in bus.agents, f"missing agent {name}"


def test_detection_stub_without_weights():
    bus = build_system(run="models/parking")
    out = bus.dispatch("t", "detection", Envelope("detect", {"image": "x.jpg"}))
    assert out["backend"] == "stub"
    assert isinstance(out["cars_detected"], int)


def test_hic_trigger_logic():
    from ubigraph_agents.hic_agent import HumanInLoopAgent
    hic = HumanInLoopAgent(t_u=0.1)
    # high-spread references should trigger
    pred = {"temporal_references": {"G": [{"delta_rate": 0.5}, {"delta_rate": -0.5}]},
            "current_input_status": {"a": "available"}}
    out = hic.handle(Envelope("evaluate", {"prediction": pred}))
    assert out["trigger_human"] is True
    # flat references should not
    pred2 = {"temporal_references": {"G": [{"delta_rate": 0.01}, {"delta_rate": 0.02}]},
             "current_input_status": {"a": "available"}}
    out2 = hic.handle(Envelope("evaluate", {"prediction": pred2}))
    assert out2["trigger_human"] is False


def test_mavlink_shell_sim_backend():
    from ubigraph_drone.mavlink_adapter import MAVLinkAdapter
    m = MAVLinkAdapter(backend="sim")
    assert m.connect()["ok"]
    assert m.arm()["ok"]
    assert m.takeoff(10.0)["ok"]
    assert m.goto(40.1, -88.2, 10.0)["ok"]
    assert m.land()["ok"]
    snap = m.snapshot()
    assert snap["mode"] == "LANDED"
