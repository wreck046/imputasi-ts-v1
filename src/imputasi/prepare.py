"""Menyiapkan dataset kedua ke format yang sama dengan ETTh1 (date + K kolom numerik, per jam, tanpa NaN).

Murni pandas/numpy; tidak butuh PyTorch. Aturan pemilihan kolom bersifat DETERMINISTIK dan dicatat
(tulis aturannya di makalah; jangan memilih kolom berdasarkan hasil model).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_raw(path: str | Path, fmt: str = "generic", date_col: str = "date") -> pd.DataFrame:
    """fmt='uci_electricity': LD2011_2014.txt (sep ';', kolom pertama = timestamp, desimal ',' atau '.').
    fmt='generic': CSV dengan kolom tanggal `date_col`."""
    if fmt == "uci_electricity":
        df = pd.read_csv(path, sep=";", index_col=0, parse_dates=True, decimal=",")
        if (df.dtypes == object).any():           # berkas dengan desimal titik
            df = pd.read_csv(path, sep=";", index_col=0, parse_dates=True, decimal=".")
        df.index.name = "date"
        return df.astype(float)
    if fmt == "generic":
        df = pd.read_csv(path)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df.index.name = "date"
        return df.select_dtypes("number").astype(float)
    raise ValueError(f"format tidak dikenal: {fmt}")


def to_hourly(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """Agregasi ke per jam bila data lebih rapat dari 1 jam; bila sudah per jam, kembalikan apa adanya."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("indeks harus DatetimeIndex")
    step = pd.Series(df.index).diff().median()
    if step < pd.Timedelta(hours=1):
        return getattr(df.resample("1h"), how)()
    if step > pd.Timedelta(hours=1):
        raise ValueError(f"resolusi {step} lebih kasar dari 1 jam; protokol ini mengasumsikan data per jam")
    return df


def fix_all_zero_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Baris di mana SEMUA kolom tepat 0 (artefak pergantian jam musim panas pada data listrik UCI)
    dijadikan NaN lalu diinterpolasi linear. Mengembalikan (df, jumlah_baris)."""
    bad = (df == 0).all(axis=1)
    n = int(bad.sum())
    if n:
        df = df.copy()
        df.loc[bad] = np.nan
        df = df.interpolate(limit=3, limit_direction="both")
    return df, n


def eligible_columns(df: pd.DataFrame, max_zero_frac: float = 0.01, max_nan_frac: float = 0.0) -> list[str]:
    """Kolom tanpa NaN (di atas batas), hampir tanpa nol (klien yang belum ada tercatat 0), dan tidak konstan."""
    out = []
    for c in df.columns:
        s = df[c]
        if s.isna().mean() > max_nan_frac:
            continue
        if (s == 0).mean() > max_zero_frac:
            continue
        if s.nunique() < 10:
            continue
        out.append(c)
    return out


def select_columns(cols: list[str], k: int, method: str = "even") -> list[str]:
    """even: k kolom berjarak merata dari daftar terurut; first: k pertama."""
    cols = sorted(map(str, cols))
    if k > len(cols):
        raise ValueError(f"hanya {len(cols)} kolom memenuhi syarat, diminta {k}")
    if method == "first":
        return cols[:k]
    if method == "even":
        idx = np.round(np.linspace(0, len(cols) - 1, k)).astype(int)
        return [cols[i] for i in idx]
    raise ValueError(f"method tidak dikenal: {method}")


def prepare(df: pd.DataFrame, k: int, select: str = "even", start: str | None = None, end: str | None = None,
            last: int | None = None, drop=(), max_zero_frac: float = 0.01) -> tuple[pd.DataFrame, dict]:
    """Pipeline lengkap -> (DataFrame hasil, info untuk dicatat di makalah)."""
    info: dict = {}
    df = to_hourly(df)
    if start or end:
        df = df.loc[start:end]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    elig = eligible_columns(df, max_zero_frac=max_zero_frac)
    cols = select_columns(elig, k, select)
    out = df[cols].copy()
    out, n_fix = fix_all_zero_rows(out)
    if last:
        out = out.iloc[-last:]
    if out.isna().any().any():
        raise ValueError("Masih ada NaN setelah perbaikan; persempit rentang waktu (--start/--end).")
    info.update({"n_rows": len(out), "n_eligible": len(elig), "columns": cols, "select": select,
                 "n_allzero_rows_fixed": n_fix, "first": str(out.index[0]), "last": str(out.index[-1])})
    return out, info


def split_shift_report(df: pd.DataFrame, fractions=(0.7, 0.1, 0.2)) -> pd.DataFrame:
    """Rata-rata dan sd z-score (statistik dari TRAIN) pada val dan test, per kolom. Nilai jauh dari 0/1 = pergeseran
    distribusi, yang dapat membuat ambang validasi tidak berpindah antar dataset."""
    x = df.to_numpy(float)
    n = len(x)
    a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    mu, sd = x[:a].mean(0), x[:a].std(0)
    rows = []
    for j, c in enumerate(df.columns):
        zj = lambda s, j=j: (s - mu[j]) / sd[j]
        rows.append({"kolom": c, "val_mean": zj(x[a:b, j]).mean(), "val_std": zj(x[a:b, j]).std(),
                     "test_mean": zj(x[b:, j]).mean(), "test_std": zj(x[b:, j]).std()})
    return pd.DataFrame(rows)


def make_configs(name: str, csv_path: str, columns: list[str], base_baseline: dict, base_csdi: dict,
                 conditions=("random20", "block24", "block72", "block24_sim"), seeds=(0, 1),
                 methods=("linear", "seasonal_interp", "kalman", "ridge_mv"), nsample: int = 20) -> tuple[dict, dict]:
    """Konfigurasi baseline dan CSDI untuk dataset baru dengan protokol DIPERKECIL (dicatat di makalah)."""
    b = {k: v for k, v in base_baseline.items()}
    b["data"] = dict(base_baseline["data"], path=str(csv_path), name=name, target_cols=list(columns))
    b["seeds"] = list(seeds)
    b["conditions"] = [c for c in base_baseline["conditions"] if c["name"] in conditions]
    missing = set(conditions) - {c["name"] for c in b["conditions"]}
    if missing:
        raise ValueError(f"kondisi tidak ada di baseline.yaml: {missing}")
    b["methods"] = list(methods)
    c = {k: v for k, v in base_csdi.items()}
    c["base_config"] = f"configs/{name}_baseline.yaml"
    c["eval"] = dict(base_csdi["eval"], nsample=nsample)
    return b, c
