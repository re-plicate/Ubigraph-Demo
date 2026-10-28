"""Minimal agent primitives: envelope, message, and agent base.

Agents are intentionally synchronous in-process callers on a shared bus. They do
not own threads; the bus keeps an ordered evidence chain so every decision can be
audited (per the project doc's "decision evidence chain").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Callable


@dataclass
class Envelope:
    """A typed request handed from one agent to another."""
    action: str
    payload: dict = field(default_factory=dict)
    reply_to: str | None = None


@dataclass
class Message:
    """One bus record; doubles as the evidence-chain entry."""
    seq: int
    ts: float
    src: str
    dst: str
    action: str
    payload: dict
    status: str = "ok"          # ok | error | routed
    note: str = ""


class Agent:
    """Base class. Subclasses register handlers for the actions they accept."""

    name: str = "agent"

    def __init__(self, bus: "AgentBus | None" = None) -> None:  # noqa: F821
        self.bus = bus
        self.handlers: dict[str, Callable[[Envelope], dict]] = {}
        self._register()
        if bus is not None:
            bus.register(self)

    def _register(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def handle(self, env: Envelope) -> dict:
        handler = self.handlers.get(env.action)
        if handler is None:
            return {"error": f"{self.name} 不支持动作 {env.action}"}
        return handler(env)

    def call(self, dst: str, env: Envelope) -> dict:
        if self.bus is None:
            return {"error": f"{self.name} 未连接 bus"}
        return self.bus.dispatch(src=self.name, dst=dst, env=env)

    def emit(self, action: str, **payload) -> None:
        if self.bus is not None:
            self.bus.log(self.name, "bus", action, payload)
