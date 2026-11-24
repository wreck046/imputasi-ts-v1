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
