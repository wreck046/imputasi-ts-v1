"""Pemuatan data, split kronologis, dan standardisasi (statistik dari TRAIN saja)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Splits:
    train: np.ndarray  # (T_train, K) terstandardisasi
    val: np.ndarray
    test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    columns: list[str]


def load_csv(path: str | Path, date_col: str = "date", target_cols=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col in df.columns:
        df = df.drop(columns=[date_col])
    if target_cols:
        df = df[list(target_cols)]
    if df.isna().any().any():
        raise ValueError("Data asli sudah mengandung NaN; ground truth harus lengkap.")
    return df


def chronological_split(df: pd.DataFrame, fractions=(0.7, 0.1, 0.2)) -> Splits:
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions harus berjumlah 1")
    x = df.to_numpy(dtype=np.float64)
    n = len(x)
    n_tr = int(n * fractions[0])
    n_va = int(n * fractions[1])
    tr, va, te = x[:n_tr], x[n_tr : n_tr + n_va], x[n_tr + n_va :]
    mean, std = tr.mean(0), tr.std(0)
    std = np.where(std < 1e-12, 1.0, std)
    z = lambda a: (a - mean) / std
    return Splits(z(tr), z(va), z(te), mean, std, list(df.columns))


def make_synthetic_ett(n: int = 17420, k: int = 7, seed: int = 123) -> pd.DataFrame:
    """Data sintetis mirip ETT (musiman 24 & 168 jam + tren + noise).

    HANYA untuk uji kode saat ETTh1 asli tidak tersedia. Jangan dipakai untuk hasil paper.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cols = {}
    for j in range(k):
        daily = rng.uniform(0.5, 2.0) * np.sin(2 * np.pi * t / 24 + rng.uniform(0, 6.28))
        weekly = rng.uniform(0.2, 1.0) * np.sin(2 * np.pi * t / 168 + rng.uniform(0, 6.28))
        trend = np.cumsum(rng.normal(0, 0.01, n))
        noise = rng.normal(0, rng.uniform(0.1, 0.4), n)
        cols[f"V{j}"] = daily + weekly + trend + noise
    return pd.DataFrame(cols)
