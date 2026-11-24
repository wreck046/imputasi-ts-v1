"""Baseline imputasi. Semua fungsi menerima x_obs (T, K) dengan NaN di titik hilang.

Kontrak: output hanya boleh bergantung pada nilai TEROBSERVASI (tidak ada kebocoran
ground truth). Tes tests/test_baselines.py memeriksa ini.

Setiap imputer mengembalikan (mu, sd) dengan sd=None untuk metode deterministik.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- sederhana
def linear(x_obs: np.ndarray, **_) -> tuple[np.ndarray, None]:
    df = pd.DataFrame(x_obs)
    out = df.interpolate(method="linear", limit_direction="both", axis=0)
    return out.to_numpy(), None


def ffill(x_obs: np.ndarray, **_) -> tuple[np.ndarray, None]:
    df = pd.DataFrame(x_obs)
    out = df.ffill().bfill()  # bfill hanya untuk NaN di awal deret
    return out.to_numpy(), None


def _nearest_seasonal(col: np.ndarray, period: int, direction: int, max_lags: int) -> np.ndarray:
    """Nilai terobservasi terdekat pada jam yang sama (t -/+ j*period), j=1..max_lags."""
    t = len(col)
    found = np.full(t, np.nan)
    for j in range(1, max_lags + 1):
        s = j * period
        if s >= t:
            break
        shifted = np.full(t, np.nan)
        if direction < 0:      # nilai di t - s
            shifted[s:] = col[: t - s]
        else:                  # nilai di t + s
            shifted[: t - s] = col[s:]
        take = np.isnan(found) & ~np.isnan(shifted)
        found[take] = shifted[take]
    return found


def seasonal_interp(x_obs: np.ndarray, period: int = 24, max_lags: int = 14, **_):
    """Isi dengan rata-rata nilai pada jam yang sama dari hari terobservasi terdekat
    (sebelum dan sesudah). Jika hanya satu sisi ada, pakai sisi itu; jika tidak ada,
    jatuh ke interpolasi linear."""
    fallback, _ = linear(x_obs)
    out = x_obs.copy()
    for k in range(x_obs.shape[1]):
        col = x_obs[:, k]
        back = _nearest_seasonal(col, period, -1, max_lags)
        fwd = _nearest_seasonal(col, period, +1, max_lags)
        both = ~np.isnan(back) & ~np.isnan(fwd)
        est = np.where(both, (back + fwd) / 2, np.where(np.isnan(back), fwd, back))
        miss = np.isnan(col)
        fill = np.where(np.isnan(est), fallback[:, k], est)
        out[miss, k] = fill[miss]
    return out, None


# ---------------------------------------------------------------- STL iteratif
def stl(x_obs: np.ndarray, period: int = 24, iterations: int = 5, robust: bool = True, **_):
    """Imputasi iteratif: isi linear -> STL -> ganti titik hilang dengan trend+musiman -> ulang."""
    from statsmodels.tsa.seasonal import STL

    filled, _ = linear(x_obs)
    miss = np.isnan(x_obs)
    for k in range(x_obs.shape[1]):
        if not miss[:, k].any():
            continue
        col = filled[:, k].copy()
        for _it in range(iterations):
            res = STL(col, period=period, robust=robust).fit()
            recon = res.trend + res.seasonal
            col[miss[:, k]] = recon[miss[:, k]]
        filled[:, k] = col
    return filled, None


# ---------------------------------------------------------------- Kalman (state-space)
class KalmanImputer:
    """Local linear trend + musiman harmonik (period) per variabel.

    Parameter di-fit SEKALI pada split train (lengkap, tanpa NaN), lalu dipakai tetap
    (smoothing saja) pada deret yang punya titik hilang. Ini mencegah fitting pada data
    test dan membuat biaya per-kondisi murah.
    Keluaran: rata-rata dan simpangan baku prediktif (Gaussian) untuk interval/CRPS.
    """

    def __init__(self, period: int = 24, harmonics: int = 3, maxiter: int = 200):
        self.period, self.harmonics, self.maxiter = period, harmonics, maxiter
        self.params_: list[np.ndarray] = []
        self.converged_: list[bool | None] = []   # per variabel; None = tidak diketahui

    def _model(self, endog: np.ndarray):
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        return UnobservedComponents(
            endog,
            level="local linear trend",
            freq_seasonal=[{"period": self.period, "harmonics": self.harmonics}],
        )

    def _cache_path(self, train: np.ndarray, cache_dir):
        import hashlib
        from pathlib import Path

        h = hashlib.sha1(np.ascontiguousarray(train).tobytes()).hexdigest()[:10]
        name = f"kalman_p{self.period}_h{self.harmonics}_it{self.maxiter}_{h}.npz"
        return Path(cache_dir) / name

    def fit(self, train: np.ndarray, cache_dir=None) -> "KalmanImputer":
        """Fit parameter per variabel pada train. Jika cache_dir diberikan, hasil fit disimpan
        dan dipakai ulang (kunci: period, harmonics, maxiter, hash data train)."""
        path = self._cache_path(train, cache_dir) if cache_dir else None
        if path is not None and path.exists():
            z = np.load(path, allow_pickle=False)
            self.params_ = [row for row in z["params"]]
            self.converged_ = [None if c < 0 else bool(c) for c in z["converged"]]
            return self

        self.params_, self.converged_ = [], []
        for k in range(train.shape[1]):
            res = self._model(train[:, k]).fit(disp=False, maxiter=self.maxiter, method="lbfgs")
            self.params_.append(np.asarray(res.params))
            conv = getattr(res, "mle_retvals", None) or {}
            self.converged_.append(conv.get("converged", None))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, params=np.stack(self.params_),
                     converged=np.array([-1 if c is None else int(c) for c in self.converged_]))
        return self

    def convergence_summary(self) -> str:
        n = len(self.converged_)
        ok = sum(1 for c in self.converged_ if c is True)
        unk = sum(1 for c in self.converged_ if c is None)
        return f"konvergen {ok}/{n} variabel" + (f" ({unk} tidak diketahui)" if unk else "")

    def impute(self, x_obs: np.ndarray, **_):
        if not self.params_:
            raise RuntimeError("Panggil fit(train) dulu.")
        t, k = x_obs.shape
        mu, sd = np.empty((t, k)), np.empty((t, k))
        for j in range(k):
            res = self._model(x_obs[:, j]).smooth(self.params_[j])
            # statsmodels mengembalikan matriks time-invariant sebagai 2D, time-varying sebagai 3D
            z = np.asarray(res.model.ssm["design"])
            z = (z[..., 0] if z.ndim == 3 else z)[0]                     # (n_states,)
            ov = np.asarray(res.model.ssm["obs_cov"])
            obs_var = float((ov[..., 0] if ov.ndim == 3 else ov)[0, 0])
            state = np.asarray(res.smoothed_state)                       # (n_states, T)
            cov = np.asarray(res.smoothed_state_cov)                     # (n_states, n_states, T)
            mu[:, j] = z @ state
            var = np.einsum("i,ijt,j->t", z, cov, z) + obs_var
            sd[:, j] = np.sqrt(np.maximum(var, 1e-12))
        # titik terobservasi dikembalikan apa adanya
        obs = ~np.isnan(x_obs)
        mu = np.where(obs, x_obs, mu)
        return mu, sd


# ---------------------------------------------------------------- registry
def get_imputer(name: str, cfg: dict, train: np.ndarray | None = None):
    """Kembalikan fungsi f(x_obs) -> (mu, sd)."""
    period = cfg.get("period", 24)
    if name == "linear":
        return linear
    if name == "ffill":
        return ffill
    if name == "seasonal_interp":
        return lambda x: seasonal_interp(x, period=period)
    if name == "stl":
        s = cfg.get("stl", {})
        return lambda x: stl(x, period=period, iterations=s.get("iterations", 5), robust=s.get("robust", True))
    if name == "kalman":
        kc = cfg.get("kalman", {})
        imp = KalmanImputer(period, kc.get("harmonics", 3), kc.get("maxiter", 200))
        imp.fit(train, cache_dir=cfg.get("_cache_dir"))
        return imp.impute
    if name in ("ridge_uv", "ridge_mv", "ridge_mv_lag"):
        from .mv_baselines import RidgeMV

        rc = cfg.get("ridge_mv", {})
        lags = tuple(rc.get("lags", [24])) if name == "ridge_mv_lag" else ()
        imp = RidgeMV(period, lags, rc.get("alpha", 1.0), rc.get("min_rows", 100), rc.get("n_aug", 2),
                      cfg.get("conditions"), cfg.get("block_missing_ratio", 0.2),
                      use_others=(name != "ridge_uv")).fit(train)
        return imp.impute
    raise ValueError(f"metode tidak dikenal: {name}")
