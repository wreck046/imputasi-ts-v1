import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("rank_val", Path(__file__).resolve().parents[1] / "scripts" / "rank_val_vs_test.py")
rv = importlib.util.module_from_spec(spec); spec.loader.exec_module(rv)


def _write(tmp_path, name, rows):
    pd.DataFrame(rows).to_csv(tmp_path / name, index=False)


def test_selection_and_spearman(tmp_path):
    for ms, v, t in [(0, 0.30, 0.31), (1, 0.31, 0.32), (2, 0.50, 0.47)]:
        _write(tmp_path, f"val_{ms}.csv", [{"model_seed": ms, "condition": "block24_sim", "seed": s,
                                           "nsample": 10, "mae": v} for s in (0, 1)])
        _write(tmp_path, f"test_{ms}.csv", [{"model_seed": ms, "condition": "block24_sim", "seed": s,
                                            "nsample": 50, "mae": t} for s in (0, 1, 2)])
    val = rv.load(str(tmp_path / "val_*.csv"), ["block24_sim"])
    test = rv.load(str(tmp_path / "test_*.csv"), ["block24_sim"], [0, 1, 2])
    res = rv.analyse(val, test)
    assert int(res.loc[0, "selected_model"]) == 0
    assert abs(res.loc[0, "spearman"] - 1.0) < 1e-9
    assert abs(res.loc[0, "test_worst"] - 0.47) < 1e-9


def test_guard_flags_models_worse_than_baseline(tmp_path, capsys):
    val = pd.DataFrame({"condition": ["c"] * 2, "model_seed": [0, 1], "mae": [0.30, 0.50]})
    test = pd.DataFrame({"condition": ["c"] * 2, "model_seed": [0, 1], "mae": [0.31, 0.52]})
    b = pd.DataFrame({"method": ["ridge_mv", "seasonal_interp"] * 2, "condition": "c",
                      "seed": [0, 0, 1, 1], "mae": [0.35, 0.40, 0.35, 0.40]})
    b.to_csv(tmp_path / "b.csv", index=False)
    m = rv.guard(val, test, str(tmp_path / "b.csv"), ["c"])
    assert m.set_index("model_seed")["lolos_val"].to_dict() == {0: True, 1: False}
    assert (m["baseline"] == "ridge_mv").all()
