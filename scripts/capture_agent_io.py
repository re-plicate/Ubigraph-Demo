"""Capture real agent I/O for all 6 agents."""
import sys, json, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ubigraph_agents.base import Envelope
from ubigraph_agents.temporal_agent import TemporalAgent
from ubigraph_agents.spatial_agent import SpatialAgent
from ubigraph_agents.transfer_agent import TransferAgent
from ubigraph_agents.detection_agent import DetectionAgent
from ubigraph_agents.drone_agent import DroneFleetAgent
from ubigraph_agents.hic_agent import HumanInLoopAgent

# Real inputs
hist = pd.read_csv(ROOT / "examples" / "history_24.csv")
raw = pd.read_csv(ROOT / "data" / "raw" / "aarhus_parking.csv")
src = pd.DataFrame({"garagecode": raw["garagecode"], "timestamp": raw["updatetime"],
                    "available": raw["totalspaces"] - raw["vehiclecount"],
                    "capacity": raw["totalspaces"]})
sf = pd.read_csv(ROOT / "data" / "raw" / "sfpark.csv")

io_log = {}

# 1. TemporalAgent
print("=== TemporalAgent ===")
ta = TemporalAgent()
payload = {"garage": "BRUUNS", "histories": hist, "seed": 7}
t0 = time.perf_counter()
out = ta.handle(Envelope("predict", payload))
elapsed = time.perf_counter() - t0
io_log["temporal"] = {
    "input": {"garage": "BRUUNS", "window": 24, "histories_shape": str(hist.shape)},
    "output": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()},
    "latency_ms": round(elapsed * 1000, 1)
}
print(json.dumps(io_log["temporal"], indent=2, ensure_ascii=False))

# 2. SpatialAgent
print("\n=== SpatialAgent ===")
sa = SpatialAgent()
payload = {"garage": "BRUUNS"}
t0 = time.perf_counter()
out = sa.handle(Envelope("similarity", payload))
elapsed = time.perf_counter() - t0
io_log["spatial"] = {
    "input": {"garage": "BRUUNS"},
    "output": {k: (v[:3] if isinstance(v, list) else v) for k, v in out.items()},
    "latency_ms": round(elapsed * 1000, 1)
}
print(json.dumps(io_log["spatial"], indent=2, ensure_ascii=False))

# 3. TransferAgent
print("\n=== TransferAgent ===")
tra = TransferAgent()
payload = {"source": src, "target": sf}
t0 = time.perf_counter()
out = tra.handle(Envelope("transfer_predict", payload))
elapsed = time.perf_counter() - t0
io_log["transfer"] = {
    "input": {"source": "Aarhus (8 garages)", "target": "SFpark (5 garages)"},
    "output": {k: (v if not isinstance(v, list) else f"[{len(v)} predictions]") for k, v in out.items()},
    "latency_ms": round(elapsed * 1000, 1)
}
print(json.dumps(io_log["transfer"], indent=2, ensure_ascii=False))

# 4. DetectionAgent
print("\n=== DetectionAgent ===")
da = DetectionAgent()
payload = {"image": None}
t0 = time.perf_counter()
out = da.handle(Envelope("detect", payload))
elapsed = time.perf_counter() - t0
io_log["detection"] = {
    "input": {"image": "None (sensor failure mode)"},
    "output": out,
    "latency_ms": round(elapsed * 1000, 1)
}
print(json.dumps(io_log["detection"], indent=2, ensure_ascii=False))

# 5. DroneFleetAgent
print("\n=== DroneFleetAgent ===")
dra = DroneFleetAgent()
payload = {}
t0 = time.perf_counter()
out = dra.handle(Envelope("state", payload))
elapsed = time.perf_counter() - t0
io_log["drone"] = {
    "input": {"action": "state"},
    "output": {k: (v if not isinstance(v, list) else f"[{len(v)} items]") for k, v in out.items()},
    "latency_ms": round(elapsed * 1000, 1)
}
print(json.dumps(io_log["drone"], indent=2, ensure_ascii=False))

# 6. HICAgent
print("\n=== HICAgent ===")
ha = HumanInLoopAgent()
payload = {"prediction": 0.5, "uncertainty": 0.3, "cost": 0.1}
t0 = time.perf_counter()
out = ha.handle(Envelope("evaluate", payload))
elapsed = time.perf_counter() - t0
io_log["hic"] = {
    "input": payload,
    "output": out,
    "latency_ms": round(elapsed * 1000, 1)
}
print(json.dumps(io_log["hic"], indent=2, ensure_ascii=False))

# Save
(ROOT / "results" / "agent_io_log.json").write_text(
    json.dumps(io_log, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nSaved to results/agent_io_log.json")
