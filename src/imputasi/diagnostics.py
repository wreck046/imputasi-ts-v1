"""Diagnostik: bagaimana sebuah model mengisi SATU blok hilang tertentu (plot imputasi vs kebenaran).

Murni numpy/matplotlib; sampler dari run_csdi (atau sampler palsu di tes). Jendela yang dihitung hanya yang
menyentuh rentang yang diplot, sehingga cepat (puluhan jendela, bukan seluruh deret).
"""
from __future__ import annotations

import numpy as np

from .windows import owner_windows, stitch, window_starts, window_timepoints


def find_blocks(mask: np.ndarray, min_len: int = 1, simultaneous: bool = True) -> list[tuple[int, int]]:
    """Daftar (awal, akhir_eksklusif) run hilang. simultaneous=True: baris dengan SEMUA variabel hilang."""
    row = mask.all(axis=1) if simultaneous else mask.any(axis=1)
    out, t, n = [], 0, len(row)
    while t < n:
        if row[t]:
            s = t
            while t < n and row[t]:
                t += 1
            if t - s >= min_len:
                out.append((s, t))
        else:
            t += 1
    return out


def impute_range(sampler, test: np.ndarray, mask: np.ndarray, t_lo: int, t_hi: int, *, window_len: int,
                 stride: int, nsample: int, tp_mode: str = "relative", tp_offset: int = 0) -> np.ndarray:
    """Sampel imputasi (S, t_hi-t_lo, K) untuk rentang [t_lo, t_hi), dengan perlakuan identik dengan
    evaluate_condition (nilai hilang dinolkan, jendela sama, titik pemilik sama)."""
    T, K = test.shape
    obs = test.copy()
    obs[mask] = 0.0
    starts = window_starts(T, window_len, stride)
    owner = owner_windows(T, window_len, starts)
    need = np.unique(owner[t_lo:t_hi])                      # hanya jendela yang menjadi pemilik titik di rentang
    idx = starts[need][:, None] + np.arange(window_len)[None, :]
    w_obs = obs[idx].astype(np.float32)
    w_cond = (~mask)[idx].astype(np.float32)
    w_tp = window_timepoints(starts[need], window_len, tp_offset, tp_mode).astype(np.float32)
    s = np.asarray(sampler(w_obs, w_cond, w_tp, nsample))
    if s.shape != (len(need), nsample, window_len, K):
        raise ValueError(f"Bentuk sampel {s.shape} tidak sesuai harapan.")
    full_samples = np.zeros((len(starts), nsample, window_len, K), dtype=s.dtype)
    full_samples[need] = s
    return stitch(full_samples, starts, T)[:, t_lo:t_hi, :]


def block_summary(truth: np.ndarray, mask: np.ndarray, samples: np.ndarray, t_lo: int, t_hi: int,
                  columns, level: float = 0.95) -> list[dict]:
    """MAE (median sampel), bias rata-rata (imputasi - kebenaran), lebar dan cakupan interval per variabel
    pada titik hilang dalam [t_lo, t_hi)."""
    a = (1 - level) / 2
    med = np.median(samples, 0)
    lo, hi = np.quantile(samples, a, 0), np.quantile(samples, 1 - a, 0)
    y, m = truth[t_lo:t_hi], mask[t_lo:t_hi]
    rows = []
    for j, c in enumerate(columns):
        mj = m[:, j]
        if not mj.any():
            continue
        e = med[mj, j] - y[mj, j]
        rows.append({"variabel": c, "mae": float(np.abs(e).mean()), "bias": float(e.mean()),
                     "lebar": float((hi - lo)[mj, j].mean()),
                     "cakupan": float(((y[mj, j] >= lo[mj, j]) & (y[mj, j] <= hi[mj, j])).mean())})
    return rows


def render_figure(truth, mask, samples_by_label: dict, columns, t_lo, t_hi, variables, out_path, *,
                  level: float = 0.95, baselines: dict | None = None, title: str = "", context: int = 0):
    """Satu panel per (variabel x model): kebenaran (hitam), median imputasi + interval (berwarna), titik hilang
    (latar abu), garis putus-putus = baseline (opsional)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = (1 - level) / 2
    labels = list(samples_by_label)
    ncol, nrow = len(labels), len(variables)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.6 * nrow), sharex=True, squeeze=False)
    cols = list(columns)
    x = np.arange(t_lo, t_hi)
    colors = ["#2a6f97", "#c1121f", "#6a994e", "#9d4edd"]
    for r, v in enumerate(variables):
        j = cols.index(v)
        for c, lab in enumerate(labels):
            ax = axes[r][c]
            smp = samples_by_label[lab]
            med = np.median(smp[:, :, j], 0)
            lo, hi = np.quantile(smp[:, :, j], a, 0), np.quantile(smp[:, :, j], 1 - a, 0)
            miss = mask[t_lo:t_hi, j]
            ax.fill_between(x, 0, 1, where=miss, transform=ax.get_xaxis_transform(), color="0.9", lw=0)
            ax.fill_between(x, lo, hi, where=miss, color=colors[c % 4], alpha=0.25, lw=0)
            ax.plot(x, truth[t_lo:t_hi, j], color="k", lw=1.0, label="kebenaran")
            ax.plot(x[miss], med[miss], ".", color=colors[c % 4], ms=3, label="median imputasi")
            if baselines:
                for bi, (bn, bm) in enumerate(baselines.items()):
                    ax.plot(x[miss], bm[t_lo:t_hi, j][miss], "--", lw=0.9, color=["#e9a000", "#555555"][bi % 2],
                            label=bn)
            if r == 0:
                ax.set_title(lab, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{v} (z-score)")
            if r == nrow - 1:
                ax.set_xlabel("indeks waktu pada split test")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", fontsize=8, ncol=len(l))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
