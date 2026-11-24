"""Satu eksperimen = (metode, kondisi, seed) -> baris hasil."""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import norm

from . import metrics as M
from .masks import make_mask

VAL_SEED_OFFSET = 10_000  # mask validasi memakai seed berbeda dari mask test


def _apply(imputer, series: np.ndarray, mask: np.ndarray):
    x_obs = series.copy()
    x_obs[mask] = np.nan
    t0 = time.perf_counter()
    mu, sd = imputer(x_obs)
    return mu, sd, time.perf_counter() - t0


def run_one(imputer, name: str, cond: dict, seed: int, val: np.ndarray, test: np.ndarray,
            block_ratio: float, level: float = 0.95, columns=None) -> dict:
    mask = make_mask(cond, test.shape, seed, block_ratio)
    mu, sd, dt = _apply(imputer, test, mask)

    if sd is not None:  # prediktif Gaussian bawaan (Kalman)
        zq = norm.ppf(0.5 + level / 2)
        lo, hi = mu - zq * sd, mu + zq * sd
        crps = M.crps_gaussian(test, mu, sd, mask)
        interval_src = "gaussian_model"
    else:               # titik: interval dari kuantil |residual| pada VALIDASI (mask berbeda)
        vmask = make_mask(cond, val.shape, seed + VAL_SEED_OFFSET, block_ratio)
        vmu, _, _ = _apply(imputer, val, vmask)
        q = float(np.quantile(np.abs(val[vmask] - vmu[vmask]), level))
        lo, hi = mu - q, mu + q
        crps = M.crps_point(test, mu, mask)
        interval_src = "val_residual_quantile"

    cov, width = M.interval_stats(test, lo, hi, mask)
    names = list(columns) if columns is not None else [str(i) for i in range(test.shape[1])]
    per_var = {f"mae_var_{n}": v for n, v in zip(names, M.mae_per_variable(test, mu, mask))}
    row = {
        "method": name, "condition": cond["name"], "seed": seed,
        "n_missing": int(mask.sum()), "missing_frac": float(mask.mean()),
        "mae": M.mae(test, mu, mask), "rmse": M.rmse(test, mu, mask),
        "crps": crps, "coverage": cov, "width": width, "interval_src": interval_src,
        "impute_time_s": dt,
    }
    row.update(per_var)
    return row
