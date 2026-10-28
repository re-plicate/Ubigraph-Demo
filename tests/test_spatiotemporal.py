"""Behavioral tests for leakage boundaries, synchronization, and serving parity."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from ubigraph_core.utils import seed_all
from ubigraph_st.config import STConfig
from ubigraph_st.data import Episodes, build_episodes, budgeted_support
from ubigraph_st.memory import STBank, reference_indices, gather_memory, extend_for_target
from ubigraph_st.model import JointGraphPredictor, prediction_loss
from ubigraph_st.condensation import initial_bank, condense
from ubigraph_st.training import fit, adapt, STForecaster, make_bundle


def toy(n=40, names=None):
    names = names or ["A", "B", "C"]
    generator = torch.Generator().manual_seed(42)
    rates = .3+.3*torch.rand(n, len(names), 12, generator=generator)
    fixed = torch.cat([torch.ones(n, len(names), 12), torch.zeros(n, len(names), 12),
                       torch.rand(n, len(names), 8, generator=generator)], -1)
    x = torch.cat([rates, fixed], -1)
    # Small predictable target based on own last change, plus a cross-node signal.
    y = (rates[..., -1]+.15*(rates[..., -1]-rates[..., -2])+.02*rates[..., 0].roll(1, 1)).clamp(0, 1)
    return Episodes(x, y, torch.ones(n, len(names), dtype=torch.bool),
                    torch.ones(n, len(names), dtype=torch.bool), torch.arange(n)*30,
                    torch.arange(n)*30+30, names, torch.tensor([100.]*len(names)))


def config():
    return STConfig(window=12, folds=True, neighbors=2, synthetic_nodes=12, fewshot=16, hidden=12,
                    epochs=3, validation_interval=1, patience=3, real_batch=3,
                    condense_steps=6, reset_interval=3, adversarial_steps=1, adaptation_epochs=3)


def test_synchronous_selection_and_target_exclusion():
    e, cfg = toy(), config()
    bank, selected = initial_bank(e.nodes(["B", "C"]), cfg, 7)
    assert torch.equal(bank.episodes.x, e.x[selected][:, [1, 2]])
    assert "A" not in bank.episodes.names
    support = budgeted_support(e, "A", 16)
    extended = extend_for_target(bank, support, "A")
    assert int(extended.episodes.labeled[:, 0].sum()) == 16
    assert int(extended.episodes.labeled[:, 1].sum()) == 12


def test_future_and_own_answers_are_inert():
    seed_all(7)
    e, cfg = toy(), config()
    bank = STBank(e)
    q = e.subset([15])
    model = JointGraphPredictor(cfg)
    prior = torch.ones(3, 3)
    ids = reference_indices(bank, q, cfg)
    baseline = model(bank, q, ids, prior)
    changed = Episodes.fromdict(e.asdict())
    changed.y[15:] = torch.rand_like(changed.y[15:])
    other_bank = STBank(changed)
    assert torch.equal(ids, reference_indices(other_bank, q, cfg))
    torch.testing.assert_close(baseline, model(other_bank, q, ids, prior), rtol=0, atol=0)
    q.y[:] = .99
    torch.testing.assert_close(baseline, model(bank, q, ids, prior), rtol=0, atol=0)


def test_invalid_source_history_cannot_change_target():
    seed_all(7)
    e, cfg = toy(), config()
    q = e.subset([30])
    q.available[:, 1:] = False
    model = JointGraphPredictor(cfg)
    bank = STBank(e)
    base = model(bank, q, reference_indices(bank, q, cfg), torch.ones(3, 3))[:, 0]
    q.x[:, 1:] = torch.rand_like(q.x[:, 1:])
    after = model(bank, q, reference_indices(bank, q, cfg), torch.ones(3, 3))[:, 0]
    torch.testing.assert_close(base, after, rtol=0, atol=0)


def test_groc_physical_bounds_fixed_fields_and_changes():
    seed_all(7)
    e, cfg = toy(), config()
    initial, _ = initial_bank(e, cfg, 7)
    bank, summary, trace = condense(e, torch.eye(3)+.2, cfg, 7)
    torch.testing.assert_close(bank.episodes.x[..., 12:], initial.episodes.x[..., 12:], rtol=0, atol=0)
    torch.testing.assert_close(bank.episodes.y, initial.episodes.y, rtol=0, atol=0)
    assert summary["max_feature_displacement"] <= cfg.trust_radius+1e-6
    assert 0 < summary["max_adversarial_noise"] <= cfg.adversarial_epsilon+1e-6
    assert summary["edge_change_l2"] > 0
    assert summary["model_reinitializations"] == 2
    ids = reference_indices(bank, bank.episodes, cfg)
    gather_memory(bank, bank.episodes, ids, cfg)
    assert len(trace) == cfg.condense_steps


def test_history_availability_independent_of_future_label():
    from ubigraph_core.data import Prepared, Windows
    cfg = config()
    times = pd.date_range("2014-08-31", periods=60, freq="30min")
    grid = pd.DataFrame({"available_rate": .5, "label_rate": .5, "observed": True,
                         "age": 0., "capacity": 100.}, index=times)
    stub = Windows(np.empty((0, 44)), np.empty(0), np.empty(0), np.empty(0), "A", 100.)
    p = Prepared({n: {"train": stub} for n in ["A", "B", "C"]},
                 {n: grid.copy() for n in ["A", "B", "C"]}, pd.DataFrame(), {})
    before = build_episodes(p, cfg)
    # A missing FUTURE label must not remove B's current historical input.
    p.grids["B"].loc[times[30], "label_rate"] = np.nan
    after = build_episodes(p, cfg)
    for key in before:
        torch.testing.assert_close(before[key].x, after[key].x, rtol=0, atol=0)
        assert torch.equal(before[key].available, after[key].available)
    assert sum(int(v.labeled.sum()) for v in before.values()) == sum(int(v.labeled.sum()) for v in after.values())+1


def test_fit_adapt_reload_and_cutoff(tmp_path):
    seed_all(7)
    e, cfg = toy(70), config()
    bank, _ = initial_bank(e.subset(np.arange(40)).nodes(["B", "C"]), cfg, 7)
    prior = torch.eye(2)+.2
    model, meta, _ = fit(bank, e.subset(np.arange(40, 50)).nodes(["B", "C"]), prior, cfg, 7)
    support = budgeted_support(e.subset(np.arange(40)), "A", 16)
    final, adapter, budget, _ = adapt(model, bank, support, "A", torch.eye(3)+.2, cfg, "attention")
    assert budget["label_budget"] == 16 and budget["target_september_labels_used"] == 0
    assert budget["fit_labels"]+budget["selection_labels"] == 16
    bundle = make_bundle(model, final, torch.eye(3)+.2, cfg, "attention", 1500, adapter, "A")
    path = tmp_path/"model.pt"
    torch.save(bundle, path)
    q = e.subset([55, 56])
    a, b = STForecaster(bundle), STForecaster.load(path)
    torch.testing.assert_close(a.predict(q), b.predict(q), rtol=0, atol=0)
    with pytest.raises(ValueError, match="cutoff"):
        a.predict(e.subset([1]))


def test_attention_and_source_features_have_gradients():
    seed_all(7)
    e, cfg = toy(), config()
    q = e.subset([30])
    q.x.requires_grad_(True)
    model = JointGraphPredictor(cfg)
    bank = STBank(e)
    pred = model(bank, q, reference_indices(bank, q, cfg), torch.eye(3)+.2)
    pred[0, 0].backward()
    assert q.x.grad[0, 0].abs().sum() > 0
    assert q.x.grad[0, 1:].abs().sum() > 0
    assert model.q.weight.grad.abs().sum() > 0


def test_end_to_end_public_prediction_budget_and_future_rejection(tmp_path):
    import json
    from ubigraph_st.experiment import run
    from ubigraph_st.serving import predict_saved
    cfg = config()
    cfg.train_cap, cfg.eval_cap = 64, 8
    cfg.methods, cfg.seeds = ("groc",), (7,)
    cfg.ablations = False
    result = run(cfg, tmp_path/"run")
    assert result["status"] == "complete"
    output = tmp_path/"run"
    expected = json.loads((output/"examples/fewshot_BRUUNS_expected.json").read_text())
    history = pd.read_csv(output/"examples/fewshot_BRUUNS_histories.csv")
    prediction = predict_saved(output, "BRUUNS", history, protocol="fewshot")
    assert abs(prediction["predicted_available"]-expected["predicted_available"]) < 1e-3
    assert prediction["provenance"]["target_label_budget"] == 16
    with pytest.raises(ValueError, match="after as_of"):
        predict_saved(output, "BRUUNS", history, protocol="fewshot", as_of="2014-09-30 23:00")
    target_only = predict_saved(output, "BRUUNS", history[history.garagecode == "BRUUNS"], protocol="fewshot")
    assert target_only["spatial_attention"]["BRUUNS"] == 1.
    assert sum(target_only["spatial_attention"].values()) == 1.
    budget = pd.read_csv(output/"fewshot_budget.csv")
    assert (budget.label_budget == 16).all()
    assert (budget.target_september_labels_used == 0).all()
    assert (budget.source_target_node_overlap == 0).all()
