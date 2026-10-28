"""CityClaw multi-agent orchestration layer.

This package wraps the existing UbiGraph capabilities (spatiotemporal prediction,
CoSimHeat spatial similarity, simulated fleet) as first-class agents and adds the
missing pieces the paper/design doc calls for: a transfer agent, a detection agent,
a human-in-the-loop gate, and an evidence-keeping message bus.
"""
from .base import Agent, Envelope, Message
from .bus import AgentBus
from .orchestrator import OrchestratorAgent
from .temporal_agent import TemporalAgent
from .spatial_agent import SpatialAgent
from .transfer_agent import TransferAgent
from .detection_agent import DetectionAgent
from .drone_agent import DroneFleetAgent
from .hic_agent import HumanInLoopAgent

__all__ = [
    "Agent", "Envelope", "Message", "AgentBus",
    "OrchestratorAgent", "TemporalAgent", "SpatialAgent", "TransferAgent",
    "DetectionAgent", "DroneFleetAgent", "HumanInLoopAgent",
]
