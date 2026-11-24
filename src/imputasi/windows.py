"""Windowing dan penyambungan (stitching) untuk model berbasis jendela seperti CSDI.

Baseline mengimputasi seluruh deret test sekaligus; CSDI bekerja pada jendela tetap.
Modul ini memetakan keduanya ke titik evaluasi yang SAMA (mask global dari masks.make_mask).
Murni numpy: tidak butuh PyTorch.
"""
from __future__ import annotations

import numpy as np


def window_starts(T: int, L: int, stride: int) -> np.ndarray:
    """Awal jendela pada grid `stride`; jendela terakhir selalu menempel di ujung deret."""
    if L > T:
        raise ValueError(f"window_len {L} > panjang deret {T}")
    starts = list(range(0, T - L + 1, stride))
    if starts[-1] != T - L:
        starts.append(T - L)
    return np.array(starts, dtype=int)


def owner_windows(T: int, L: int, starts: np.ndarray) -> np.ndarray:
    """Untuk tiap t: indeks jendela yang paling 'sentral' di t, yaitu memaksimalkan
    min(jarak ke tepi kiri, jarak ke tepi kanan). Jendela seperti itu punya konteks
    terbanyak di kedua sisi. Seri -> jendela paling awal."""
    owner = np.full(T, -1, dtype=int)
    score = np.full(T, -1, dtype=int)
    for w, s in enumerate(starts):
        t = np.arange(s, s + L)
        sc = np.minimum(t - s, s + L - 1 - t)
        better = sc > score[t]
        owner[t[better]] = w
        score[t[better]] = sc[better]
    if (owner < 0).any():
        raise ValueError("Ada titik waktu yang tidak tercakup jendela mana pun.")
    return owner


def stitch(samples: np.ndarray, starts: np.ndarray, T: int) -> np.ndarray:
    """samples (W, S, L, K) -> (S, T, K), tiap t diambil dari jendela pemiliknya."""
    W, S, L, K = samples.shape
    owner = owner_windows(T, L, starts)
    out = np.empty((S, T, K), dtype=samples.dtype)
    for w, s in enumerate(starts):
        t = np.arange(s, s + L)
        sel = owner[t] == w
        out[:, t[sel], :] = samples[w][:, sel, :]
    return out


def window_timepoints(starts: np.ndarray, L: int, offset: int = 0, mode: str = "relative") -> np.ndarray:
    """Nilai 'timepoints' yang dikirim ke embedding waktu CSDI, bentuk (W, L).

    relative     : 0..L-1 (default CSDI; model TIDAK tahu jam-dalam-hari, harus menyimpulkannya).
    hour_of_week : (indeks global) mod 168. Memberi fase harian/mingguan secara eksplisit,
                   setara dengan pengetahuan period=24 yang dipakai baseline musiman.
    `offset` = indeks global awal split (0 untuk train, len(train) untuk val, len(train)+len(val) untuk test).
    """
    starts = np.asarray(starts)
    if mode == "relative":
        return np.tile(np.arange(L, dtype=np.float64), (len(starts), 1))
    if mode == "hour_of_week":
        return ((offset + starts[:, None] + np.arange(L)[None, :]) % 168).astype(np.float64)
    raise ValueError(f"mode tidak dikenal: {mode}")
