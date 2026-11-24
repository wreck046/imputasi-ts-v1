"""Generator mask. Konvensi: mask=True berarti titik HILANG (disembunyikan)."""
from __future__ import annotations

import numpy as np


def random_mask(shape: tuple[int, int], ratio: float, seed: int) -> np.ndarray:
    """Hilang acak per entri (T, K), tepat round(ratio*T*K) titik."""
    rng = np.random.default_rng(seed)
    t, k = shape
    n_miss = int(round(ratio * t * k))
    flat = np.zeros(t * k, dtype=bool)
    flat[rng.choice(t * k, size=n_miss, replace=False)] = True
    return flat.reshape(t, k)


def _place_blocks(t: int, length: int, n_blocks: int, rng: np.random.Generator) -> np.ndarray:
    """Tempatkan n_blocks blok tak tumpang-tindih dengan panjang `length` pada deret panjang t."""
    if n_blocks * length > t:
        raise ValueError("Total blok melebihi panjang deret.")
    # Teknik 'stars and bars': sebar sisa ruang sebagai celah acak antar blok.
    free = t - n_blocks * length
    cuts = np.sort(rng.integers(0, free + 1, size=n_blocks))
    starts = cuts + np.arange(n_blocks) * length
    m = np.zeros(t, dtype=bool)
    for s in starts:
        m[s : s + length] = True
    return m


def block_mask(
    shape: tuple[int, int],
    length: int,
    ratio: float,
    seed: int,
    simultaneous: bool = False,
) -> np.ndarray:
    """Blok hilang berurutan sepanjang `length` jam.

    ratio        : proporsi total titik hilang (per variabel).
    simultaneous : True -> blok sama di semua variabel; False -> posisi blok independen per variabel.
    """
    rng = np.random.default_rng(seed)
    t, k = shape
    n_blocks = max(1, int(round(ratio * t / length)))
    if simultaneous:
        col = _place_blocks(t, length, n_blocks, rng)
        return np.tile(col[:, None], (1, k))
    return np.stack([_place_blocks(t, length, n_blocks, rng) for _ in range(k)], axis=1)


def make_mask(cond: dict, shape: tuple[int, int], seed: int, block_ratio: float) -> np.ndarray:
    if cond["kind"] == "random":
        return random_mask(shape, cond["ratio"], seed)
    if cond["kind"] == "block":
        return block_mask(shape, cond["length"], block_ratio, seed, cond.get("simultaneous", False))
    raise ValueError(f"kind tidak dikenal: {cond['kind']}")
