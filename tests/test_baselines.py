import numpy as np
import pytest
from imputasi import baselines as B
from imputasi.data import chronological_split, make_synthetic_ett
from imputasi.masks import block_mask


def _setup(seed=0):
    x = chronological_split(make_synthetic_ett(n=3000), (0.7, 0.1, 0.2)).test
    mask = block_mask(x.shape, 24, 0.2, seed)
    obs = x.copy(); obs[mask] = np.nan
    return x, obs, mask


@pytest.mark.parametrize("fn", [B.linear, B.ffill, B.seasonal_interp])
def test_no_nan_and_no_leakage(fn):
    x, obs, mask = _setup()
    mu, _ = fn(obs)
    assert not np.isnan(mu).any()
    # ubah ground truth di titik hilang -> keluaran harus identik
    x2 = x.copy(); x2[mask] += 100.0
    obs2 = x2.copy(); obs2[mask] = np.nan
    mu2, _ = fn(obs2)
    assert np.allclose(mu, mu2)


def test_observed_points_untouched():
    x, obs, mask = _setup()
    for fn in (B.linear, B.ffill, B.seasonal_interp):
        mu, _ = fn(obs)
        assert np.allclose(mu[~mask], x[~mask])


def test_seasonal_beats_linear_on_seasonal_data():
    x, obs, mask = _setup()
    e_lin = np.abs(B.linear(obs)[0][mask] - x[mask]).mean()
    e_sea = np.abs(B.seasonal_interp(obs)[0][mask] - x[mask]).mean()
    assert e_sea < e_lin


def test_ffill_leading_nan():
    obs = np.array([[np.nan], [np.nan], [1.0], [np.nan], [3.0]])
    assert np.allclose(B.ffill(obs)[0].ravel(), [1, 1, 1, 1, 3])


def test_linear_edges():
    obs = np.array([[np.nan], [1.0], [np.nan], [3.0], [np.nan]])
    assert np.allclose(B.linear(obs)[0].ravel(), [1, 1, 2, 3, 3])


# Butuh statsmodels: dilewati per-tes (bukan satu modul) bila belum terpasang
needs_sm = pytest.mark.skipif(
    __import__("importlib").util.find_spec("statsmodels") is None,
    reason="statsmodels belum terpasang",
)


@needs_sm
def test_stl_runs_and_no_leakage():
    x, obs, mask = _setup()
    mu, _ = B.stl(obs, period=24, iterations=2)
    assert not np.isnan(mu).any()
    x2 = x.copy(); x2[mask] += 100.0
    obs2 = x2.copy(); obs2[mask] = np.nan
    assert np.allclose(mu, B.stl(obs2, period=24, iterations=2)[0])


@needs_sm
def test_kalman_runs_gives_positive_sd():
    sp = chronological_split(make_synthetic_ett(n=3000), (0.7, 0.1, 0.2))
    imp = B.KalmanImputer(24, 3, maxiter=10).fit(sp.train[:, :2])
    x = sp.test[:, :2]
    mask = block_mask(x.shape, 24, 0.2, 0)
    obs = x.copy(); obs[mask] = np.nan
    mu, sd = imp.impute(obs)
    assert not np.isnan(mu).any() and (sd > 0).all()
    # sd di titik hilang harus >= sd di titik terobservasi (lebih tidak pasti)
    assert sd[mask].mean() > sd[~mask].mean()
