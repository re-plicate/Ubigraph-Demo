#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python_bin="${UBIGRAPH_PYTHON:-.venv/bin/python}"
"$python_bin" -m ubigraph_st run --config configs/st_main.json --output outputs/st_main_retrain "$@"
