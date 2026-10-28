import argparse
import json
from pathlib import Path

import pandas as pd

from ubigraph_core.utils import write_json
from .config import STConfig


def main():
    parser = argparse.ArgumentParser(description="V3：同步图凝缩、空间特征交互与少样本迁移")
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("run")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--resume", action="store_true")
    prediction = sub.add_parser("predict")
    prediction.add_argument("--run", type=Path, required=True)
    prediction.add_argument("--garage", required=True)
    prediction.add_argument("--histories", type=Path, required=True)
    prediction.add_argument("--as-of")
    prediction.add_argument("--protocol", choices=("full_history", "fewshot"), default="full_history")
    prediction.add_argument("--seed", type=int)
    prediction.add_argument("--output", type=Path)
    reporting = sub.add_parser("report")
    reporting.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            from .experiment import run
            run(STConfig.load(args.config), args.output, args.resume)
        elif args.command == "predict":
            from .serving import predict_saved
            result = predict_saved(args.run, args.garage, pd.read_csv(args.histories), args.protocol, args.seed, args.as_of)
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            from .report import build_report
            print(build_report(args.run.resolve()))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
