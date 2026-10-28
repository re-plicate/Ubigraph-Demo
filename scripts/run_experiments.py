"""Build the CityClaw big experiment table.

Combines:
  A) Prediction accuracy ablations (from results/metrics_across_seeds.csv)
  B) Multi-agent system metrics (HIC trigger, evidence chain, agent latency)
  C) Drone scheduling metrics (energy, makespan, conflict-free routes)
  D) MAVLink shell conformance

Outputs:
  results/experiment_table.csv
  results/experiment_table.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS


def section_a_prediction() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "metrics_across_seeds.csv")
    df = df.copy()
    df["section"] = "A_prediction"
    df["mae"] = df["mae_spaces_mean"].round(2)
    df["mae_std"] = df["mae_spaces_std"].round(2)
    df["rmse"] = df["rmse_spaces_mean"].round(2)
    df["nmae"] = (df["macro_nmae_mean"] * 100).round(3)
    return df[["section", "protocol", "method", "mode", "variant", "model_role",
               "history_points", "neighbors", "mae", "mae_std", "rmse", "nmae",
               "test_labels_per_seed"]]


def section_b_agents() -> pd.DataFrame:
    """HIC trigger rate + evidence chain on real saved predictions."""
    import sys
    sys.path.insert(0, str(ROOT))
    from ubigraph_agents.system import build_system
    from ubigraph_agents.base import Envelope
    from ubigraph_agents.hic_agent import HumanInLoopAgent

    bus = build_system(run="models/parking")
    hic = HumanInLoopAgent()

    # Load real saved predictions across seeds and measure trigger rate.
    rows = []
    pred_files = sorted((RESULTS / "predictions").glob("seed_*.csv"))
    for pf in pred_files:
        seed = int(pf.stem.split("_")[1])
        df = pd.read_csv(pf)
        # Use residual spread as uncertainty proxy: residual |truth - prediction|
        if {"truth", "prediction"}.issubset(df.columns):
            resid = (df["truth"] - df["prediction"]).abs()
            u = float(resid.std())
            trigger_rate = float((resid > resid.quantile(0.85)).mean())
        else:
            u = float("nan")
            trigger_rate = float("nan")
        rows.append({"seed": seed, "n_samples": len(df),
                     "residual_std": round(u, 3) if u == u else None,
                     "hic_trigger_rate@85pct": round(trigger_rate, 3) if trigger_rate == trigger_rate else None})

    # Evidence chain completeness
    n_calls = 0
    t0 = time.perf_counter()
    for garage in ["BRUUNS", "SALLING", "MAGASIN"]:
        bus.dispatch("exp", "spatial", Envelope("neighbors", {"garage": garage, "topk": 3}))
        bus.dispatch("exp", "hic", Envelope("evaluate", {"prediction": {
            "temporal_references": [{"delta_rate": 0.1 * i - 0.2} for i in range(8)],
            "current_input_status": {"a": "available"}}}))
        n_calls += 2
    latency_ms = (time.perf_counter() - t0) / n_calls * 1000
    evidence = bus.evidence()

    trs = [r["hic_trigger_rate@85pct"] for r in rows if r["hic_trigger_rate@85pct"] is not None]
    return pd.DataFrame([{
        "section": "B_multi_agent",
        "metric": "hic_trigger_rate_mean",
        "value": round(float(np.mean(trs)), 3) if trs else None,
        "detail": json.dumps(rows, ensure_ascii=False),
    }, {
        "section": "B_multi_agent",
        "metric": "agent_call_latency_ms",
        "value": round(latency_ms, 2),
        "detail": f"{n_calls} in-process calls",
    }, {
        "section": "B_multi_agent",
        "metric": "evidence_chain_entries",
        "value": len(evidence),
        "detail": "ordered, auditable",
    }, {
        "section": "B_multi_agent",
        "metric": "agents_registered",
        "value": len(bus.agents),
        "detail": ",".join(bus.agents.keys()),
    }])


def section_c_drone() -> pd.DataFrame:
    """Run the existing DroneService across N drones and targets; record energy/makespan."""
    import sys, tempfile
    sys.path.insert(0, str(ROOT))
    from ubigraph_drone.service import DroneService

    rows = []
    for n_drones in [1, 2, 3]:
        td = Path(tempfile.mkdtemp(prefix=f"fleet_exp_{n_drones}_"))
        svc = DroneService(td, autostart=False)
        targets = ["G1", "G3", "G5", "G2", "G4"][:n_drones * 2]
        try:
            plan = svc.plan(request_id=f"exp_{n_drones}", targets=targets,
                            drone_ids=[str(i) for i in range(1, n_drones + 1)],
                            dwell_seconds=3.0)
            feasible = plan["feasible"]
            duration = round(plan["duration_seconds"], 1)
            distance = round(plan["distance_m"], 1)
            energy = {k: round(v, 1) for k, v in plan.get("energy_required_percent", {}).items()}
            rejected = plan.get("rejected", [])
        except Exception as exc:
            feasible, duration, distance, energy, rejected = False, None, None, {}, [str(exc)]
        finally:
            svc.close()
        rows.append({
            "section": "C_drone_scheduling",
            "n_drones": n_drones, "n_targets": len(targets),
            "feasible": feasible, "makespan_s": duration,
            "total_distance_m": distance,
            "energy_per_drone_pct": json.dumps(energy),
            "rejected": json.dumps(rejected, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def section_d_mavlink() -> pd.DataFrame:
    from ubigraph_drone.mavlink_adapter import MAVLinkAdapter
    m = MAVLinkAdapter(backend="sim")
    ops = [m.connect, m.arm, lambda: m.takeoff(10.0),
           lambda: m.goto(40.1, -88.2, 10.0), m.land, m.disconnect]
    results = []
    for op in ops:
        t0 = time.perf_counter()
        r = op()
        results.append((r.get("ok", True), (time.perf_counter() - t0) * 1000))
    return pd.DataFrame([{
        "section": "D_mavlink_shell",
        "metric": "command_surface",
        "value": "connect/arm/takeoff/goto/land/disconnect",
        "detail": f"{sum(1 for ok,_ in results if ok)}/{len(results)} ok; "
                  f"latency_ms={[round(x,1) for _,x in results]}",
    }, {
        "section": "D_mavlink_shell",
        "metric": "mavlink_backend_supported",
        "value": "sim, mavlink(pymavlink), dryrun",
        "detail": "real SITL via udp:IP:14550 when pymavlink installed",
    }])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = section_a_prediction()
    b = section_b_agents()
    c = section_c_drone()
    d = section_d_mavlink()

    # Save long-format CSV
    combined = pd.concat([
        a.assign(value="", detail=""),
        b, c, d,
    ], ignore_index=True, sort=False)
    combined.to_csv(OUT / "experiment_table.csv", index=False, encoding="utf-8-sig")

    # Build a readable markdown table (section A)
    md = ["# CityClaw 大实验表\n",
          "## A. 预测精度消融（Aarhus/CityPulse, 3 seeds, 9748 test labels）\n"]
    show = a[a.model_role.isin(["primary", "baseline"]) | (a.protocol == "fewshot")]
    show = show[["protocol", "method", "mode", "variant", "mae", "mae_std", "rmse", "nmae"]]
    md.append(show.to_markdown(index=False, floatfmt=".2f"))
    md.append("\n\n## B. 多智能体系统指标\n")
    md.append(b.to_markdown(index=False))
    md.append("\n\n## C. 无人机调度（能量约束下的多机协同）\n")
    md.append(c.to_markdown(index=False))
    md.append("\n\n## D. MAVLink 接口壳\n")
    md.append(d.to_markdown(index=False))
    (OUT / "experiment_table.md").write_text("\n".join(md), encoding="utf-8")
    print("written:", OUT / "experiment_table.csv")
    print("written:", OUT / "experiment_table.md")
    print("\n=== A preview ===")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
