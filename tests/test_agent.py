"""Agent contracts: frozen V3 parity, causal inputs, real tool continuation and cancellation."""
import copy
import json
import io
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import pytest

from ubigraph_core.config import ROOT
from ubigraph_st.serving import predict_saved
from ubigraph_agent.config import AgentConfig
from ubigraph_agent.engine import V3Engine
from ubigraph_agent.memory import MemoryStore
from ubigraph_agent.responses import sse_messages, ResponsesGateway, GatewayError
from ubigraph_agent.runtime import AgentRuntime
from ubigraph_agent.tools import Skills, schemas, validate


@pytest.fixture(scope="module")
def engine():
    return V3Engine(ROOT/"models/parking")


@pytest.mark.parametrize("protocol,seed,target", [
    ("fewshot", 7, "BRUUNS"), ("fewshot", 17, "SALLING"), ("full_history", 7, "BRUUNS")])
def test_agent_forecast_matches_frozen_public_v3(engine, protocol, seed, target):
    context = engine.context({"protocol": protocol, "seed": seed})
    tables = [engine.history(n, context).assign(garagecode=n) for n in engine.names]
    expected = predict_saved(engine.run, target, pd.concat(tables), protocol=protocol, seed=seed)
    actual = engine.predict([target], context)["items"][0]
    assert actual["predicted_available"] == pytest.approx(expected["predicted_available"], abs=1e-6)
    assert actual["spatial_attention"] == expected["spatial_attention"]
    assert actual["spatial_prior"] == expected["spatial_prior"]
    assert len(actual["temporal_neighbors"]) == 8
    assert all(pd.Timestamp(n["outcome_time"]) <= pd.Timestamp(context["as_of"]) for n in actual["temporal_neighbors"])


def test_agent_reading_is_causal_and_masks_invalid_inputs(engine):
    context = engine.context()
    before = engine.predict(["BRUUNS"], context)["items"][0]
    grids = {n: g.copy() for n, g in engine.prepared.grids.items()}
    try:
        for grid in engine.prepared.grids.values():
            grid.loc[grid.index > pd.Timestamp(context["as_of"]), "available_rate"] = .12345
            grid.loc[:, "label_rate"] = .98765  # labels never used by the serving adapter
        after = engine.predict(["BRUUNS"], context)["items"][0]
        assert after["predicted_available"] == before["predicted_available"]
        stamp = pd.Timestamp(context["as_of"])
        engine.prepared.grids["SALLING"].loc[stamp, "observed"] = False
        masked = engine.predict(["BRUUNS"], context)["items"][0]
        assert masked["spatial_attention"]["SALLING"] == 0
        engine.prepared.grids["BRUUNS"].loc[stamp, "observed"] = False
        with pytest.raises(ValueError, match="历史不满足"):
            engine.predict(["BRUUNS"], context)
    finally:
        engine.prepared.grids = grids


def test_agent_input_modes_and_scope(engine):
    context = engine.context()
    result = engine.predict(["BRUUNS"], context, include_other=False)["items"][0]
    assert result["spatial_attention"]["BRUUNS"] == 1
    assert sum(result["spatial_attention"].values()) == 1
    with pytest.raises(ValueError, match="30分钟"):
        engine.predict(["BRUUNS"], context, horizon=60)
    with pytest.raises(ValueError, match="停车场"):
        engine.predict(["NOREPORT"], context)
    with pytest.raises(ValueError):
        engine.context({"as_of": "2014-10-01 00:15:00"})
    with pytest.raises(ValueError):
        engine.context({"as_of": "2026-10-25 12:00:00"})
    with pytest.raises(ValueError):
        engine.context({"seed": True})
    assert not engine.capabilities()["acquisition_enabled"]


def test_main_experiment_is_unambiguous(engine):
    result = engine.benchmarks()
    assert len(result["primary_results"]) == 1
    row = result["primary_results"][0]
    assert row["protocol"] == "full_history"
    assert row["method"] == "groc" and row["mode"] == "attention"
    assert row["mae_spaces_mean"] == pytest.approx(11.22899, abs=.001)
    experiment = result["additional_experiments"][0]
    assert experiment["protocol"] == "fewshot" and not experiment["is_primary"]
    assert experiment["mae_spaces_mean"] == pytest.approx(17.0569, abs=.001)


def test_default_prediction_uses_full_history_main_checkpoint(engine):
    context = engine.context()
    assert context["protocol"] == "full_history"
    assert engine.capabilities()["default_protocol"] == "full_history"
    prediction = engine.predict(["BRUUNS"], context)
    f = engine.forecaster("BRUUNS", context)
    assert f.bundle["target"] is None and f.adapter is None
    assert f.bank.kind == "groc"
    assert prediction["protocol"] == "full_history"
    assert len(prediction["items"][0]["spatial_attention"]) == 6


def test_strict_skill_parameters_and_no_acquisition():
    definitions = schemas(["BRUUNS", "SALLING"])
    spec = next(s for s in definitions if s["name"] == "predict_parking")
    valid = {"targets": ["BRUUNS"], "horizon_minutes": 30, "include_other_parking": True}
    validate(valid, spec)
    for invalid in [dict(valid, horizon_minutes=60), dict(valid, horizon_minutes=True),
                    dict(valid, targets=["UNKNOWN"]), dict(valid, targets=[]),
                    dict(valid, include_other_parking="true"), dict(valid, extra=1)]:
        with pytest.raises(ValueError):
            validate(invalid, spec)
    assert not any("drone" in s["name"] or "acquir" in s["name"] for s in definitions)


def test_sse_multiline_comments_and_done():
    lines = [b": keepalive\n", b"event: example\n", b'data: {"type": "test",\n',
             b'data: "value": 1}\n', b"\n", b"data: [DONE]\n", b"\n"]
    assert list(sse_messages(lines)) == [{"type": "test", "value": 1}]
    assert list(sse_messages([b'data: {"tail":true}'])) == [{"tail": True}]


def test_gateway_shows_only_provider_summary_and_preserves_length_limited_answer():
    final = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
             "output": [{"type": "message", "content": [{"type": "output_text", "text": "部分说明"}]}]}
    events = [{"type": "response.reasoning_text.delta", "delta": "private should be ignored"},
              {"type": "response.reasoning_summary_text.delta", "delta": "provider summary"},
              {"type": "response.incomplete", "response": final}]
    stream = b"".join(("data: "+json.dumps(e)+"\n\n").encode() for e in events)
    gateway = ResponsesGateway(AgentConfig())
    gateway._request = lambda *a, **k: io.BytesIO(stream)
    visible = []
    result = gateway.complete([], [], "", lambda k, d: visible.append((k, d)), lambda: False)
    assert result == final
    assert "private should be ignored" not in json.dumps(visible)
    assert ("reasoning_summary", {"text": "provider summary"}) in visible
    assert any(k == "warning" for k, _ in visible)
    final["output"] = [{"type": "function_call", "name": "predict_parking", "arguments": "{"}]
    broken = ("data: "+json.dumps({"type": "response.incomplete", "response": final})+"\n\n").encode()
    gateway._request = lambda *a, **k: io.BytesIO(broken)
    with pytest.raises(GatewayError, match="未完成"):
        gateway.complete([], [], "", lambda *a: None, lambda: False)


class StubEngine:
    names = ["BRUUNS", "SALLING"]

    def context(self, value=None):
        return value or {"as_of": "2014-09-30 23:30:00", "protocol": "fewshot", "seed": 7}

    def inspect(self, names, context):
        return {"kind": "history", "items": [{"garage": n, "observed_points": 12} for n in names]}


class ScriptedGateway:
    """Deterministic protocol stub, never used by the production application."""
    def __init__(self):
        self.requests = []

    def complete(self, inputs, tools, instructions, emit, cancelled, model):
        self.requests.append(copy.deepcopy(inputs))
        if len(self.requests) == 1:
            return {"output": [{"type": "reasoning", "id": "rs_test", "summary": []},
                  {"type": "function_call", "call_id": "call_test", "name": "inspect_history",
                   "arguments": '{"targets":["BRUUNS"]}'}]}
        emit("text_delta", {"text": "历史数据可用。"})
        return {"output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": "历史数据可用。"}]}]}


def wait_terminal(runtime, task_id):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        task = runtime.memory.task(task_id)
        if task["status"] not in ("queued", "running"):
            return task
        time.sleep(.01)
    pytest.fail("Agent did not finish")


def test_responses_continuation_memory_and_terminal_event(tmp_path):
    memory = MemoryStore(tmp_path/"memory.sqlite3")
    config = AgentConfig(runtime_dir=tmp_path)
    gateway = ScriptedGateway()
    runtime = AgentRuntime(config, StubEngine(), gateway=gateway, memory=memory)
    session = memory.create_session(runtime.engine.context())
    task = runtime.submit(session["id"], "检查BRUUNS历史", session["context"])
    assert wait_terminal(runtime, task)["status"] == "completed"
    continued = gateway.requests[1]
    assert any(x.get("type") == "reasoning" for x in continued)
    call_output = next(x for x in continued if x.get("type") == "function_call_output")
    assert call_output["call_id"] == "call_test"
    assert json.loads(call_output["output"])["result"]["items"][0]["garage"] == "BRUUNS"
    assert memory.events(task)[-1]["type"] == "done"  # status and event are atomic
    assert memory.history(session["id"])[-1]["content"] == "历史数据可用。"
    other = memory.create_session(runtime.engine.context())
    assert not memory.history(other["id"]) and not memory.artifacts(other["id"])
    with pytest.raises(ValueError, match="上下文"):
        runtime.submit(session["id"], "检查", {**session["context"], "seed": 17})


def test_cancel_is_immediate_and_late_provider_output_is_discarded(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class BlockingGateway(ScriptedGateway):
        def complete(self, *args):
            entered.set()
            release.wait(4)
            return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "不应保存"}]}]}

    runtime = AgentRuntime(AgentConfig(runtime_dir=tmp_path), StubEngine(), gateway=BlockingGateway())
    session = runtime.memory.create_session(runtime.engine.context())
    task = runtime.submit(session["id"], "检查", session["context"])
    assert entered.wait(3)
    with pytest.raises(ValueError, match="执行中"):
        runtime.submit(session["id"], "重复提交", session["context"])
    runtime.cancel(task)
    assert runtime.memory.task(task)["status"] == "cancelled"
    assert runtime.memory.events(task)[-1]["type"] == "cancelled"
    release.set()
    time.sleep(.05)
    assert not any(m["role"] == "assistant" for m in runtime.memory.history(session["id"]))
    assert runtime.memory.events(task)[-1]["type"] == "cancelled"


def test_gateway_errors_are_redacted_and_not_fake_success(tmp_path):
    class BrokenGateway(ScriptedGateway):
        def complete(self, *args):
            raise RuntimeError("endpoint rejected fixture-placeholder-credential")

    runtime = AgentRuntime(AgentConfig(runtime_dir=tmp_path, api_key="fixture-placeholder-credential"),
                           StubEngine(), gateway=BrokenGateway())
    session = runtime.memory.create_session(runtime.engine.context())
    task = runtime.submit(session["id"], "预测", session["context"])
    assert wait_terminal(runtime, task)["status"] == "failed"
    events = runtime.memory.events(task)
    assert events[-1]["type"] == "error" and "[redacted]" in events[-1]["data"]["message"]
    assert "secret-credential" not in json.dumps(events)
    assert not runtime.memory.artifacts(session["id"])

