"""Evaluate the delivered main checkpoints on the fixed test cohort."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ubigraph_core.data import prepare
from ubigraph_core.utils import seed_all
from ubigraph_st.config import STConfig
from ubigraph_st.data import build_episodes
from ubigraph_st.evaluation import apply_layout
from ubigraph_st.training import STForecaster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "models/parking")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation")
    args = parser.parse_args()
    cfg = STConfig.load(args.run / "config.json")
    seed_all(7, cfg.torch_threads)
    data = build_episodes(prepare(cfg), cfg)["test"]
    data = apply_layout(data, ROOT / "data/evaluation/test.npz")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in cfg.seeds:
        predictor = STForecaster.load(args.run / f"seed_{seed}/all/models/groc_attention.pt")
        prediction = predictor.predict(data).numpy() * data.capacities.numpy()[None]
        truth = data.y.numpy() * data.capacities.numpy()[None]
        error = (prediction - truth)[data.labeled.numpy()]
        records.append({"seed": seed, "test_labels": len(error), "mae_spaces": float(np.abs(error).mean()),
                        "rmse_spaces": float(np.sqrt((error ** 2).mean()))})
        print(f"seed={seed}: n={len(error)}, MAE={records[-1]['mae_spaces']:.4f}, RMSE={records[-1]['rmse_spaces']:.4f}")
    table = pd.DataFrame(records)
    table.to_csv(args.output / "metrics_by_seed.csv", index=False)
    print(f"MAE={table.mae_spaces.mean():.4f} ± {table.mae_spaces.std():.4f} spaces")


if __name__ == "__main__":
    main()
