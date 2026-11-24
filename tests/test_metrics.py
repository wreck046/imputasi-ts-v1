import numpy as np
from imputasi import metrics as M
from imputasi.data import chronological_split, make_synthetic_ett
from imputasi.evaluate import run_one
from imputasi.baselines import linear
from imputasi.masks import make_mask


def test_mae_per_variable_matches_overall():
    rng = np.random.default_rng(0)
    y, p = rng.normal(size=(200, 3)), rng.normal(size=(200, 3))
    m = rng.random((200, 3)) < 0.3
    pv = M.mae_per_variable(y, p, m)
    assert pv.shape == (3,)
    # rata-rata berbobot jumlah titik hilang = MAE total
    w = m.sum(0) / m.sum()
    assert np.isclose((pv * w).sum(), M.mae(y, p, m))


def test_mae_per_variable_nan_when_no_missing():
    y = np.ones((10, 2)); p = np.zeros((10, 2))
    m = np.zeros((10, 2), bool); m[:3, 0] = True
    pv = M.mae_per_variable(y, p, m)
    assert pv[0] == 1.0 and np.isnan(pv[1])


def test_run_one_has_per_variable_columns():
    sp = chronological_split(make_synthetic_ett(n=3000), (0.7, 0.1, 0.2))
    cond = {"name": "random20", "kind": "random", "ratio": 0.2}
    row = run_one(lambda x: linear(x), "linear", cond, 0, sp.val, sp.test, 0.2, columns=["a", "b", "c", "d", "e", "f", "g"])
    assert {f"mae_var_{c}" for c in "abcdefg"} <= set(row)
    assert abs(row["missing_frac"] - 0.2) < 1e-3


def test_crps_gaussian_known_value():
    # CRPS(N(0,1), y=0) = sqrt(2/pi) - 1/sqrt(pi)  ~= 0.2337
    y = np.zeros((1, 1)); mu = np.zeros((1, 1)); sd = np.ones((1, 1)); m = np.ones((1, 1), bool)
    assert np.isclose(M.crps_gaussian(y, mu, sd, m), (np.sqrt(2) - 1) / np.sqrt(np.pi))
