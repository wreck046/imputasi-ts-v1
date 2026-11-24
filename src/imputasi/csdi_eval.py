"""Evaluasi imputer berbasis sampel (CSDI) pada mask & titik evaluasi yang sama dengan baseline.

`sampler` adalah callable (tanpa ketergantungan PyTorch di sini):
    sampler(obs (W,L,K) float32 [nilai di titik hilang = 0],
            cond (W,L,K) float32 [1 = terobservasi, 0 = hilang],
            tp   (W,L)   float32,
            nsample int) -> samples (W, nsample, L, K) numpy
"""
from __future__ import annotations

import time

import numpy as np

from . import metrics as M
from .masks import make_mask
from .windows import stitch, window_starts, window_timepoints


def evaluate_condition(sampler, test: np.ndarray, cond: dict, seed: int, block_ratio: float, *,
                       window_len: int, stride: int, nsample: int, level: float = 0.95,
                       columns=None, tp_mode: str = "relative", tp_offset: int = 0) -> dict:
    t_len, k = test.shape
    mask = make_mask(cond, test.shape, seed, block_ratio)      # True = hilang (sama dgn baseline)

    obs = test.copy()
    obs[mask] = 0.0                                            # tidak ada kebocoran ground truth
    starts = window_starts(t_len, window_len, stride)
    idx = starts[:, None] + np.arange(window_len)[None, :]     # (W, L)
    w_obs = obs[idx].astype(np.float32)
    w_cond = (~mask)[idx].astype(np.float32)
    w_tp = window_timepoints(starts, window_len, tp_offset, tp_mode).astype(np.float32)

    t0 = time.perf_counter()
    samples = np.asarray(sampler(w_obs, w_cond, w_tp, nsample))
    dt = time.perf_counter() - t0
    if samples.shape != (len(starts), nsample, window_len, k):
        raise ValueError(f"Bentuk sampel {samples.shape} tidak sesuai harapan.")

    full = stitch(samples, starts, t_len)                      # (S, T, K)
    med = np.median(full, axis=0)
    mean = full.mean(axis=0)
    a = (1 - level) / 2
    lo, hi = np.quantile(full, a, axis=0), np.quantile(full, 1 - a, axis=0)
    cov, width = M.interval_stats(test, lo, hi, mask)

    names = list(columns) if columns is not None else [str(i) for i in range(k)]
    row = {
        "method": "csdi", "condition": cond["name"], "seed": seed,
        "n_missing": int(mask.sum()), "missing_frac": float(mask.mean()),
        "mae": M.mae(test, med, mask),          # titik = median sampel (protokol CSDI)
        "rmse": M.rmse(test, mean, mask),       # titik = rata-rata sampel (penduga RMSE-optimal)
        "crps": M.crps_samples(test, full, mask),
        "coverage": cov, "width": width, "interval_src": "csdi_samples_quantile",
        "impute_time_s": dt,
        "nsample": nsample, "window_len": window_len, "stride": stride, "tp_mode": tp_mode,
    }
    row.update({f"mae_var_{n}": v for n, v in zip(names, M.mae_per_variable(test, med, mask))})
    return row
