"""Contracts for the delivered checkpoints and fixed evaluation cohort."""
import numpy as np
import pandas as pd
import pytest
import torch

from ubigraph_core.config import ROOT
from ubigraph_st.data import Episodes
from ubigraph_st.evaluation import apply_layout
from ubigraph_agent.config import AgentConfig
from ubigraph_agent.engine import V3Engine


def test_main_and_optional_protocols_use_their_actual_input_lengths():
    engine = V3Engine(AgentConfig().run_dir)
    for protocol, points, dim in [("full_history", 24, 82), ("fewshot", 12, 46)]:
        context = engine.context({"protocol": protocol})
        result = engine.predict(["BRUUNS"], context)
        assert len(result["items"][0]["history"]) == points
        assert result["input_spec"]["node_input_dim"] == dim
        assert len(result["items"][0]["temporal_neighbors"]) == 8


def test_evaluation_layout_cannot_introduce_unavailable_inputs(tmp_path):
    source = Episodes(torch.zeros(1, 1, 80), torch.zeros(1, 1), torch.tensor([[False]]),
                      torch.tensor([[False]]), torch.tensor([30]), torch.tensor([60]), ["A"], torch.tensor([100.]))
    path = tmp_path / "layout.npz"
    np.savez_compressed(path, names=["A"], anchor_ns=[30], available=[[True]], labeled=[[False]])
    with pytest.raises(ValueError, match="cannot introduce"):
        apply_layout(source, path)


def test_evaluation_layout_requires_the_exact_prediction_origins(tmp_path):
    source = Episodes(torch.zeros(1, 1, 80), torch.zeros(1, 1), torch.tensor([[True]]),
                      torch.tensor([[True]]), torch.tensor([30]), torch.tensor([60]), ["A"], torch.tensor([100.]))
    path = tmp_path / "layout.npz"
    np.savez_compressed(path, names=["A"], anchor_ns=[31], available=[[True]], labeled=[[True]])
    with pytest.raises(ValueError, match="origin is missing"):
        apply_layout(source, path)


def test_delivered_metrics_match_the_per_prediction_results():
    metrics = pd.read_csv(ROOT / "results/metrics_by_seed.csv")
    main = metrics[metrics.model_role == "primary"]
    for row in main.itertuples():
        predictions = pd.read_csv(ROOT / f"results/predictions/seed_{row.seed}.csv")
        error = predictions.prediction - predictions.truth
        assert len(predictions) == 9748
        assert error.abs().mean() == pytest.approx(row.mae_spaces, abs=1e-5)
        assert np.sqrt((error ** 2).mean()) == pytest.approx(row.rmse_spaces, abs=1e-5)
