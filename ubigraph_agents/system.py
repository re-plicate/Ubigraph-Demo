"""Wire the multi-agent system together.

Usage::

    from ubigraph_agents.system import build_system
    sys = build_system()
    result = sys.bus.dispatch("user", "orchestrator", Envelope("analyze", {...}))
"""
from __future__ import annotations

from pathlib import Path

from .base import Agent, Envelope
from .bus import AgentBus
from .detection_agent import DetectionAgent
from .drone_agent import DroneFleetAgent
from .hic_agent import HumanInLoopAgent
from .orchestrator import OrchestratorAgent
from .requirement_agent import RequirementAgent
from .spatial_agent import SpatialAgent
from .temporal_agent import TemporalAgent
from .transfer_agent import TransferAgent


def build_system(run: str = "models/parking",
                 drone_service=None,
                 mavlink_backend: str = "sim",
                 mavlink_url: str | None = None) -> AgentBus:
    bus = AgentBus()
    RequirementAgent(bus)
    TemporalAgent(bus, run=run)
    SpatialAgent(bus, run=run)
    TransferAgent(bus)
    DetectionAgent(bus)
    DroneFleetAgent(bus, drone_service=drone_service,
                     mavlink_backend=mavlink_backend, mavlink_url=mavlink_url)
    HumanInLoopAgent(bus)
    OrchestratorAgent(bus)
    return bus


__all__ = ["build_system", "Envelope", "Agent"]
