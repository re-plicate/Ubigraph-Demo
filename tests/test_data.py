import numpy as np
import pandas as pd
import pytest

from ubigraph_core.config import Config
from ubigraph_core.data import causal_grid, grid_to_windows, split_windows, fewshot_windows, history_to_feature


def rows():
    t = pd.date_range("2014-05-22 00:09", periods=30, freq="30min")
    return pd.DataFrame({"updatetime": t, "_id": np.arange(30),
                         "vehiclecount": np.arange(30), "totalspaces": 100})


def test_causal_ceil_never_moves_future_record_back():
    cfg = Config(window=3)
    grid = causal_grid(rows(), 100, cfg)
    assert grid.index[0] == pd.Timestamp("2014-05-22 00:30")
    assert (grid.source_time.dropna() <= grid.source_time.dropna().index).all()
    assert grid.iloc[0].label_rate == 1.0


def test_invalid_labels_not_clipped_and_long_gaps_not_filled():
    r = rows()
    r.loc[8, "vehiclecount"] = 101
    r = r.drop(index=[10, 11, 12, 13])
    cfg = Config(window=3, max_fill_steps=1)
    grid = causal_grid(r, 100, cfg)
    assert np.isnan(grid.iloc[8].label_rate)
    assert not grid.iloc[8].observed
    assert np.isnan(grid.iloc[11].available_rate)
    windows = grid_to_windows(grid, "A", 100, cfg)
    assert grid.index[8].value not in windows.label_ns
    assert not np.isin(grid.index[11:16].as_unit("ns").asi8, windows.anchor_ns).any()


def test_only_past_features_and_split_by_future_label():
    cfg = Config(window=3, horizon=2, train_end="2014-05-22 06:00", val_end="2014-05-22 10:00")
    r = rows()
    grid = causal_grid(r, 100, cfg)
    w = grid_to_windows(grid, "A", 100, cfg)
    assert ((w.label_ns - w.anchor_ns) == 60 * 60 * 10**9).all()
    splits = split_windows(w, cfg)
    assert splits["train"].label_ns.max() < pd.Timestamp(cfg.train_end).value
    assert splits["val"].label_ns.min() >= pd.Timestamp(cfg.train_end).value
    assert splits["val"].label_ns.max() < pd.Timestamp(cfg.val_end).value
    changed = r.copy()
    changed.loc[changed.updatetime >= pd.Timestamp(cfg.train_end), "vehiclecount"] = 80
    w2 = grid_to_windows(causal_grid(changed, 100, cfg), "A", 100, cfg)
    np.testing.assert_array_equal(splits["train"].x, split_windows(w2, cfg)["train"].x)
    np.testing.assert_array_equal(splits["train"].y, split_windows(w2, cfg)["train"].y)
    few = fewshot_windows(splits["train"], 3)
    assert len(few) == 3 and few.label_ns.max() == splits["train"].label_ns.max()


def test_history_contract_and_future_calendar():
    cfg = Config(window=3)
    table = pd.DataFrame({"timestamp": pd.date_range("2014-10-01", periods=3, freq="30min"),
                          "available": [20, 21, 22], "capacity": 100})
    x, when, capacity = history_to_feature(table, cfg)
    np.testing.assert_allclose(x[0, :3], [0.2, 0.21, 0.22])
    assert when == pd.Timestamp("2014-10-01 01:30") and capacity == 100
    table.loc[2, "available"] = 101
    with pytest.raises(ValueError):
        history_to_feature(table, cfg)


def test_config_rejects_silent_typos(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"horizonn": 3}')
    with pytest.raises(ValueError, match="Unknown"):
        Config.load(p)
