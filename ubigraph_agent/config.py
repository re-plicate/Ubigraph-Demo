from dataclasses import dataclass
import os
from pathlib import Path
import re

from ubigraph_core.config import ROOT


def env_file(path):
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip().strip("\"'")
    return result


@dataclass
class AgentConfig:
    api_key: str = ""
    detection_weights: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-6.1-sol"
    reasoning_effort: str = "medium"
    api_style: str = "auto"
    host: str = "127.0.0.1"
    port: int = 8765
    run_dir: Path = ROOT / "models/parking"
    runtime_dir: Path = ROOT / "runtime/agent"
    timeout: int = 90
    max_rounds: int = 6
    max_tool_calls: int = 12
    max_output_tokens: int = 4096

    @classmethod
    def load(cls):
        values = {**env_file(ROOT/".env.local"), **os.environ}
        cfg = cls(api_key=values.get("OPENAI_API_KEY", ""),
                  detection_weights=values.get("UBIGRAPH_DETECTION_WEIGHTS", ""),
                  base_url=values.get("OPENAI_BASE_URL", cls.base_url).rstrip("/"),
                  model=values.get("OPENAI_MODEL") or cls.model,
                  reasoning_effort=values.get("OPENAI_REASONING_EFFORT") or cls.reasoning_effort,
                  api_style=values.get("UBIGRAPH_API_STYLE", "auto"),
                  host=values.get("UBIGRAPH_AGENT_HOST", "127.0.0.1"),
                  port=int(values.get("UBIGRAPH_AGENT_PORT", "8765")),
                  run_dir=ROOT / values.get("UBIGRAPH_RUN_DIR", "models/parking"))
        if not cfg.base_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError("Model endpoint must be HTTPS, or localhost for testing")
        if cfg.reasoning_effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError('OPENAI_REASONING_EFFORT must be low, medium, high, xhigh, or max')
        if cfg.api_style not in ("auto", "responses", "chat_completions"):
            raise ValueError("UBIGRAPH_API_STYLE must be auto, responses, or chat_completions")
        return cfg

    def wire_api(self):
        if self.api_style != "auto":
            return self.api_style
        return "chat_completions" if "api.deepseek.com" in self.base_url else "responses"

    def redact(self, value):
        value = str(value)
        if self.api_key:
            value = value.replace(self.api_key, "[redacted]")
        return re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[redacted]", value)

    def public(self):
        return {"base_url": self.base_url, "model": self.model, "configured": bool(self.api_key),
                "reasoning_effort": self.reasoning_effort, "wire_api": self.wire_api(), "acquisition_enabled": False,
                "drone_simulation_enabled": True, "hardware_control_enabled": False}
