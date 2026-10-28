"""Export experiment settings and numeric results."""
import json
from pathlib import Path

from ubigraph_core.utils import write_json
import pandas as pd


def build_report(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    config = manifest["identity"]["config"]
    metrics = pd.read_csv(output / "metrics_across_seeds.csv")
    rows = json.loads(metrics.to_json(orient="records"))
    result = {
        "run_hash": manifest["run_hash"], "device": manifest["device"],
        "history_points": config["window"], "history_hours": config["window"] * config["freq_minutes"] / 60,
        "temporal_neighbors": config["neighbors"], "horizon_minutes": config["horizon"] * config["freq_minutes"],
        "seeds": config["seeds"], "garages": manifest["eligible_garages"],
        "train_end": config["train_end"], "validation_end": config["val_end"],
        "mae_rmse_unit": "parking spaces", "std": "sample standard deviation across seeds",
        "results": rows,
    }
    target = output / "evaluation.json"
    write_json(target, result)
    return target
