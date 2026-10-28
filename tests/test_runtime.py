"""Delivery regressions: configuration, adapter contracts and startup failures."""
import sqlite3
import time

import pytest

from ubigraph_core.config import ROOT
from ubigraph_agent import config as config_module
from ubigraph_agent.config import AgentConfig
from ubigraph_agent.engine import V3Engine
from ubigraph_agent.runtime import AgentRuntime
from ubigraph_drone import DroneService


class StubEngine:
    names = ["BRUUNS"]

    def context(self, value=None):
        return value or {"as_of": "2014-09-30 23:30:00", "protocol": "fewshot", "seed": 7}


def test_utf8_bom_environment_file_loads_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "ROOT", tmp_path)
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_REASONING_EFFORT"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env.local").write_text(
        '# 本机模型配置\nOPENAI_API_KEY=local-fixture-value\n'
        'OPENAI_MODEL=gpt-6.1-sol\nOPENAI_REASONING_EFFORT=medium\n',
        encoding="utf-8-sig",
    )
    cfg = AgentConfig.load()
    assert cfg.api_key == "local-fixture-value"
    assert cfg.model == "gpt-6.1-sol" and cfg.reasoning_effort == "medium"
    # A BOM on the actual first key must also be accepted.
    (tmp_path / ".env.local").write_text('OPENAI_API_KEY=first-line-value\n', encoding="utf-8-sig")
    assert AgentConfig.load().api_key == "first-line-value"


@pytest.fixture(scope="module")
def engine():
    return V3Engine(ROOT / "models/parking")


@pytest.mark.parametrize("changed_garage", ["BRUUNS", "SALLING"])
def test_agent_rejects_input_capacity_that_differs_from_checkpoint(engine, changed_garage):
    context = engine.context()
    original = engine.prepared.splits[changed_garage]["train"].capacity
    baseline = engine.predict(["BRUUNS"], context)["items"][0]["predicted_available"]
    try:
        engine.prepared.splits[changed_garage]["train"].capacity *= 2
        with pytest.raises(ValueError, match="容量.*检查点不一致"):
            engine.predict(["BRUUNS"], context)
    finally:
        engine.prepared.splits[changed_garage]["train"].capacity = original
    assert engine.predict(["BRUUNS"], context)["items"][0]["predicted_available"] == baseline


def test_session_setup_exception_has_terminal_failure_and_event(tmp_path, monkeypatch):
    runtime = AgentRuntime(AgentConfig(runtime_dir=tmp_path), StubEngine())
    try:
        session = runtime.memory.create_session(runtime.engine.context())

        def fail(_):
            raise ValueError("injected session setup failure")

        monkeypatch.setattr(runtime.memory, "artifacts", fail)
        task = runtime.submit(session["id"], "检查历史", session["context"])
        deadline = time.monotonic() + 3
        while runtime.memory.task(task)["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(.01)
        assert runtime.memory.task(task)["status"] == "failed"
        assert runtime.memory.events(task)[-1]["type"] == "error"
        assert "injected session setup failure" in runtime.memory.task(task)["error"]
    finally:
        runtime.drones.close()


def test_failed_agent_initialization_releases_fleet_worker_and_ownership(tmp_path):
    cfg = AgentConfig(runtime_dir=tmp_path)
    (tmp_path / "memory.sqlite3").write_bytes(b"invalid database for regression")
    drones = DroneService(tmp_path / "drone", autostart=True)
    with pytest.raises(sqlite3.DatabaseError):
        AgentRuntime(cfg, StubEngine(), drone_service=drones)
    assert drones.stopping.is_set() and not drones.thread.is_alive()
    # A failed startup must not hold the simulation directory indefinitely.
    replacement = DroneService(tmp_path / "drone", autostart=False)
    replacement.close()
