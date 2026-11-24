"""Metrik pada titik hilang saja. Skala: data terstandardisasi (z-score dari train)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def mae(y: np.ndarray, p: np.ndarray, m: np.ndarray) -> float:
    return float(np.abs(y[m] - p[m]).mean())


def rmse(y: np.ndarray, p: np.ndarray, m: np.ndarray) -> float:
    return float(np.sqrt(((y[m] - p[m]) ** 2).mean()))


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sd: np.ndarray, m: np.ndarray) -> float:
    """CRPS bentuk tertutup untuk prediktif Gaussian (Gneiting & Raftery, 2007)."""
    sd = np.maximum(sd[m], 1e-8)
    z = (y[m] - mu[m]) / sd
    c = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(c.mean())


def crps_point(y: np.ndarray, p: np.ndarray, m: np.ndarray) -> float:
    """CRPS prediksi deterministik = MAE."""
    return mae(y, p, m)


def interval_stats(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, m: np.ndarray) -> tuple[float, float]:
    """(cakupan empiris, lebar rata-rata) pada titik hilang."""
    inside = (y[m] >= lo[m]) & (y[m] <= hi[m])
    return float(inside.mean()), float((hi[m] - lo[m]).mean())


def mae_per_variable(y: np.ndarray, p: np.ndarray, m: np.ndarray) -> np.ndarray:
    """MAE per kolom pada titik hilang; NaN bila kolom itu tidak punya titik hilang."""
    out = np.full(y.shape[1], np.nan)
    for k in range(y.shape[1]):
        mk = m[:, k]
        if mk.any():
            out[k] = np.abs(y[mk, k] - p[mk, k]).mean()
    return out


def crps_samples(y: np.ndarray, samples: np.ndarray, m: np.ndarray) -> float:
    """CRPS dari sampel prediktif (estimator 'fair', tak bias untuk S terbatas).

    y (T,K), samples (S,T,K), m (T,K) mask titik evaluasi.
    CRPS = rata-rata |x_i - y| - sum_{i!=j}|x_i - x_j| / (2 S (S-1)).
    Jumlah selisih berpasangan dihitung lewat pengurutan (O(S log S)).
    """
    x = samples[:, m]                                   # (S, N)
    s = x.shape[0]
    if s < 2:
        raise ValueError("Butuh minimal 2 sampel.")
    term1 = np.abs(x - y[m][None, :]).mean(axis=0)      # (N,)
    xs = np.sort(x, axis=0)
    w = (2 * np.arange(1, s + 1) - s - 1)[:, None]
    pair_sum = 2.0 * (w * xs).sum(axis=0)               # sum atas pasangan terurut
    term2 = pair_sum / (2.0 * s * (s - 1))
    return float((term1 - term2).mean())
