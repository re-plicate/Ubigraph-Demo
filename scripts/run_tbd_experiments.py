"""Run ALL remaining TBD experiments for real."""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))


# ============================================================
# TBD-1: Drone fleet trade-off (REAL: call assign/build_routes)
# ============================================================
def tbd1_drone_tradeoff() -> pd.DataFrame:
    from ubigraph_drone.planner import assign, build_routes, energy_requirements
    from ubigraph_drone.scene import Drone, HOMES, NODES

    targets = ["G1", "G2", "G3", "G4", "G5", "G6"]
    rows = []
    for n_drones in [1, 2, 3]:
        drones = []
        for i, (k, (x, y)) in enumerate(HOMES.items()):
            if i >= n_drones:
                break
            drones.append(Drone(id=k, x=float(x), y=float(y), z=0., battery=100.,
                                status="idle", mission_id=None, velocity=[0., 0., 0.], fault=False))
        try:
            orders, reasons = assign(drones, targets, dwell_seconds=30, return_home=True, all_drones=drones)
            routes = build_routes(drones, orders, dwell_seconds=30, return_home=True, all_drones=drones)
            energy = energy_requirements(routes, drones)
            total_dist = sum(r["distance_m"] for r in routes.values())
            total_dur = max((r["duration_seconds"] for r in routes.values()), default=0)
            total_energy = sum(energy.values())
            covered = sum(1 for o in orders.values() if o)
            rows.append({"n_drones": n_drones,
                         "targets_covered": covered,
                         "targets_total": len(targets),
                         "total_distance_m": round(total_dist, 1),
                         "mission_duration_s": round(total_dur, 1),
                         "total_energy": round(total_energy, 2),
                         "reasons": "; ".join(r["reason"] for r in reasons) if reasons else "none"})
        except Exception as e:
            rows.append({"n_drones": n_drones, "error": str(e)})
    return pd.DataFrame(rows)


# ============================================================
# TBD-2: Multi-agent ablation (REAL: call each agent)
# ============================================================
def tbd2_agent_ablation() -> pd.DataFrame:
    from ubigraph_agents.base import Envelope
    from ubigraph_agents.temporal_agent import TemporalAgent
    from ubigraph_agents.spatial_agent import SpatialAgent
    from ubigraph_agents.transfer_agent import TransferAgent
    from ubigraph_agents.detection_agent import DetectionAgent
    from ubigraph_agents.drone_agent import DroneFleetAgent
    from ubigraph_agents.hic_agent import HumanInLoopAgent

    raw = pd.read_csv(ROOT / "data" / "raw" / "aarhus_parking.csv")
    src = pd.DataFrame({"garagecode": raw["garagecode"], "timestamp": raw["updatetime"],
                        "available": raw["totalspaces"] - raw["vehiclecount"],
                        "capacity": raw["totalspaces"]})
    sf = pd.read_csv(ROOT / "data" / "raw" / "sfpark.csv")

    rows = []
    calls = [
        ("TemporalAgent", TemporalAgent, "predict", {"garage": "BRUUNS", "horizon": 1}),
        ("SpatialAgent", SpatialAgent, "similarity", {"source": src, "target": src}),
        ("TransferAgent", TransferAgent, "transfer_predict", {"source": src, "target": sf}),
        ("DetectionAgent", DetectionAgent, "detect", {"image": None}),
        ("DroneFleetAgent", DroneFleetAgent, "state", {}),
        ("HumanInLoopAgent", HumanInLoopAgent, "evaluate",
         {"prediction": 0.5, "uncertainty": 0.3, "cost": 0.1}),
    ]
    for name, cls, action, payload in calls:
        t0 = time.perf_counter()
        try:
            agent = cls()
            out = agent.handle(Envelope(action, payload))
            rows.append({"agent": name, "action": action,
                         "call_ms": round((time.perf_counter()-t0)*1000, 1),
                         "ok": True, "out_keys": str(list(out.keys())[:6])})
        except Exception as e:
            rows.append({"agent": name, "action": action,
                         "call_ms": round((time.perf_counter()-t0)*1000, 1),
                         "ok": False, "out_keys": str(e)[:120]})
    return pd.DataFrame(rows)


# ============================================================
# TBD-3: ARIMA per garage (REAL)
# ============================================================
def tbd3_arima() -> pd.DataFrame:
    from statsmodels.tsa.arima.model import ARIMA
    raw = pd.read_csv(ROOT / "data" / "raw" / "aarhus_parking.csv")
    raw["ts"] = pd.to_datetime(raw["updatetime"], format="mixed")
    raw["available"] = raw["totalspaces"] - raw["vehiclecount"]
    test = raw[raw["ts"] >= pd.Timestamp("2014-10-01")].sort_values("ts")
    rows = []
    for code, g in test.groupby("garagecode"):
        s = g.set_index("ts")["available"].astype(float)
        if len(s) < 200:
            continue
        train, test_s = s.iloc[:-100], s.iloc[-100:]
        try:
            m = ARIMA(train, order=(1, 0, 1)).fit()
            fc = m.forecast(100)
            mae = float(np.mean(np.abs(test_s.values - fc.values)))
            rows.append({"garage": code, "n_train": len(train), "mae_spaces": round(mae, 2)})
        except Exception as e:
            rows.append({"garage": code, "error": str(e)[:80]})
    return pd.DataFrame(rows)


def tbd4_dropout_persistence() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "data" / "raw" / "aarhus_parking.csv")
    raw["ts"] = pd.to_datetime(raw["updatetime"], format="mixed")
    raw["available"] = raw["totalspaces"] - raw["vehiclecount"]
    test = raw[raw["ts"] >= pd.Timestamp("2014-10-01")].sort_values("ts")
    rng = np.random.default_rng(42)
    rows = []
    for mask_frac in [0.0, 0.10, 0.20, 0.30]:
        maes = []
        for code, g in test.groupby("garagecode"):
            s = g["available"].astype(float).values
            if len(s) < 50:
                continue
            pers = np.roll(s, 1)
            pers[0] = np.nan
            if mask_frac > 0:
                mask = rng.random(len(s)) < mask_frac
                s_masked = s.copy()
                s_masked[mask] = np.nan
                s_masked = pd.Series(s_masked).ffill(limit=2).values
                pers = np.roll(s_masked, 1)
                pers[0] = np.nan
            valid = ~np.isnan(s) & ~np.isnan(pers)
            if valid.sum() > 10:
                maes.append(float(np.mean(np.abs(s[valid] - pers[valid]))))
        rows.append({"mask_ratio": mask_frac, "mae_spaces": round(float(np.mean(maes)), 2),
                     "n_garages": len(maes)})
    return pd.DataFrame(rows)


def main():
    print("=== TBD-1: Drone fleet trade-off (real planner) ===")
    d1 = tbd1_drone_tradeoff()
    print(d1.to_string(index=False))

    print("\n=== TBD-2: Multi-agent ablation (real calls) ===")
    d2 = tbd2_agent_ablation()
    print(d2.to_string(index=False))

    print("\n=== TBD-3: ARIMA per garage (real fit) ===")
    d3 = tbd3_arima()
    print(d3.to_string(index=False))

    print("\n=== TBD-4: Persistence dropout (real) ===")
    d4 = tbd4_dropout_persistence()
    print(d4.to_string(index=False))

    for name, df in [("drone_tradeoff", d1), ("agent_ablation", d2),
                     ("arima", d3), ("dropout_persistence", d4)]:
        df.to_csv(RESULTS / f"real_{name}.csv", index=False, encoding="utf-8-sig")
    print("\nsaved real_*.csv")


if __name__ == "__main__":
    main()
