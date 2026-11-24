import numpy as np
from imputasi import baselines as B
from imputasi.masks import block_mask
from imputasi.mv_baselines import RidgeMV


def _corr_data(n=4000, k=5, seed=0, walk=0.05):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    common = np.sin(2 * np.pi * t / 24) + np.cumsum(rng.normal(0, walk, n))
    x = np.stack([common * rng.uniform(0.5, 1.5) + rng.normal(0, 0.15, n) for _ in range(k)], axis=1)
    return (x - x.mean(0)) / x.std(0)


def _split(x):
    return x[:2800], x[3200:]


def test_beats_seasonal_when_others_observed_and_no_leak():
    tr, te = _split(_corr_data())
    imp = RidgeMV(24, (), 1.0, 50, 2, None, 0.2).fit(tr)
    mask = block_mask(te.shape, 24, 0.2, 0, simultaneous=False)
    obs = te.copy(); obs[mask] = np.nan
    mu, sd = imp.impute(obs)
    assert not np.isnan(mu).any() and (sd > 0).all()
    e_mv = np.abs(mu[mask] - te[mask]).mean()
    e_seas = np.abs(B.seasonal_interp(obs)[0][mask] - te[mask]).mean()
    assert e_mv < e_seas
    te2 = te.copy(); te2[mask] += 50.0
    obs2 = te2.copy(); obs2[mask] = np.nan
    assert np.allclose(mu, imp.impute(obs2)[0])          # tidak ada kebocoran


def test_mv_beats_uv_when_others_observed_on_correlated_data():
    tr, te = _split(_corr_data(walk=0.4))     # faktor bersama tak terprediksi dari deret sendiri
    uv = RidgeMV(24, (), 1.0, 50, 2, None, 0.2, use_others=False).fit(tr)
    mv = RidgeMV(24, (), 1.0, 50, 2, None, 0.2).fit(tr)
    mask = block_mask(te.shape, 24, 0.2, 2, simultaneous=False)
    obs = te.copy(); obs[mask] = np.nan
    assert np.abs(mv.impute(obs)[0][mask] - te[mask]).mean() < 0.8 * np.abs(uv.impute(obs)[0][mask] - te[mask]).mean()


def test_level_shift_between_train_and_test_does_not_bias():
    tr, te = _split(_corr_data())
    te = te + 3.0                                           # pergeseran level test
    imp = RidgeMV(24, (), 1.0, 50, 2, None, 0.2).fit(tr)
    mask = block_mask(te.shape, 24, 0.2, 0, simultaneous=False)
    obs = te.copy(); obs[mask] = np.nan
    mu, _ = imp.impute(obs)
    assert abs((mu[mask] - te[mask]).mean()) < 0.3


def test_simultaneous_blocks_fall_back_to_pattern0_without_crash():
    tr, te = _split(_corr_data())
    imp = RidgeMV(24, (24,), 1.0, 50, 2, None, 0.2).fit(tr)
    mask = block_mask(te.shape, 72, 0.2, 1, simultaneous=True)
    obs = te.copy(); obs[mask] = np.nan
    mu, sd = imp.impute(obs)
    assert not np.isnan(mu).any() and (sd > 0).all()


def test_observed_points_untouched_and_get_imputer_registry():
    tr, te = _split(_corr_data())
    cfg = {"period": 24, "conditions": [{"name": "r30", "kind": "random", "ratio": 0.3}],
           "block_missing_ratio": 0.2, "ridge_mv": {"n_aug": 1, "min_rows": 50, "lags": [24]}}
    for name in ("ridge_uv", "ridge_mv", "ridge_mv_lag"):
        f = B.get_imputer(name, cfg, train=tr)
        obs = te.copy(); m = np.random.default_rng(0).random(te.shape) < 0.3; obs[m] = np.nan
        mu, sd = f(obs)
        assert np.allclose(mu[~m], te[~m])
