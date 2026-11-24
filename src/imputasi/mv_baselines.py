"""Baseline REGRESI (non-diffusion) untuk memisahkan efek 'memakai variabel lain' dari efek 'diffusion'.

Tiga varian, semuanya ridge pada ANOMALI terhadap referensi musiman (tidak bergantung level, sehingga tahan
terhadap pergeseran level train->test):
  ridge_uv     : hanya fitur dari deret target sendiri                         (kontrol univariat terlatih)
  ridge_mv     : + variabel lain yang terobservasi pada waktu t                (multivariat)
  ridge_mv_lag : + anomali variabel lain pada t -/+ lag (diisi linear bila hilang)

Definisi, untuk target (t, k):
  ref_j(t)  = rata-rata nilai jam yang sama pada hari terobservasi terdekat sebelum/sesudah (fallback: linear)
  target    = x_k(t) - ref_k(t)
  fitur     = [x_j(t) - ref_j(t) untuk j lain yang terobservasi di t]  (hanya mv; model per POLA observasi)
              [anomali lin_j(t -/+ lag) - ref_j(...) untuk semua j lain] (hanya mv_lag)
              [lin_k(t) - ref_k(t)]
              [anomali terobservasi terdekat sebelum/sesudah, dibobot exp(-jarak/tau), tau = 6 dan 24 jam]
  model     = ridge per (k, pola); pola jarang jatuh ke model "tanpa variabel lain" (pola 0).
Latih pada TRAIN yang dimasking dengan JENIS kondisi yang sama dengan evaluasi (seed berbeda dari seed test).
Itu memberi baseline ini pengetahuan jenis pola missing, setara CSDI 'mix_block'; catat di paper.
Keluaran (mu, sd): sd = simpangan baku residual latih per model (Gaussian).
"""
from __future__ import annotations

import numpy as np

from .baselines import _nearest_seasonal, linear
from .masks import make_mask

AUG_SEED0 = 100_000   # jauh dari seed test (0..4), validasi (10_000+) dan tuning (20_000+)
TAUS = (6.0, 24.0)


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float):
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ yc)
    b = ym - xm @ w
    res = y - (X @ w + b)
    n, p = X.shape
    sd = float(np.sqrt((res ** 2).sum() / max(n - p - 1, 1)))
    return w, float(b), max(sd, 1e-6)


class RidgeMV:
    def __init__(self, period: int = 24, lags: tuple = (), alpha: float = 1.0, min_rows: int = 100,
                 n_aug: int = 2, conditions: list | None = None, block_ratio: float = 0.2,
                 use_others: bool = True, max_lags: int = 14):
        self.period, self.lags, self.alpha = period, tuple(lags), alpha
        self.min_rows, self.n_aug = min_rows, n_aug
        self.conditions, self.block_ratio = conditions, block_ratio
        self.use_others, self.max_lags = use_others, max_lags
        if not use_others:
            self.lags = ()
        self.models: dict[tuple[int, int], tuple] = {}

    # ---- komponen fitur
    def _ref(self, x_obs, lin):
        ref = np.empty_like(lin)
        for k in range(x_obs.shape[1]):
            col = x_obs[:, k]
            back = _nearest_seasonal(col, self.period, -1, self.max_lags)
            fwd = _nearest_seasonal(col, self.period, +1, self.max_lags)
            both = ~np.isnan(back) & ~np.isnan(fwd)
            est = np.where(both, (back + fwd) / 2, np.where(np.isnan(back), fwd, back))
            ref[:, k] = np.where(np.isnan(est), lin[:, k], est)
        return ref

    @staticmethod
    def _persist(x_obs, ref, k, t):
        col = x_obs[:, k]; obs = ~np.isnan(col); T = len(col); ar = np.arange(T)
        prev = np.maximum.accumulate(np.where(obs, ar, -1))
        nxt = np.minimum.accumulate(np.where(obs, ar, T)[::-1])[::-1]
        anom = np.where(obs, col - ref[:, k], 0.0)
        out = []
        for tau in TAUS:
            pv, nx = prev[t], nxt[t]
            out.append(anom[np.clip(pv, 0, T - 1)] * np.where(pv >= 0, np.exp(-(t - pv) / tau), 0.0))
            out.append(anom[np.clip(nx, 0, T - 1)] * np.where(nx < T, np.exp(-(nx - t) / tau), 0.0))
        return np.stack(out, 1)

    @staticmethod
    def _others(k, K):
        return [j for j in range(K) if j != k]

    @staticmethod
    def _cols(p, others):
        return [others[i] for i in range(len(others)) if (p >> i) & 1]

    def _X(self, x_obs, lin, ref, k, t, cols):
        T, K = x_obs.shape
        parts = []
        if cols:
            parts.append(x_obs[np.ix_(t, cols)] - ref[np.ix_(t, cols)])   # terobservasi menurut pola
        oth = self._others(k, K)
        for lag in self.lags:
            for sgn in (-1, 1):
                tt = np.clip(t + sgn * lag, 0, T - 1)
                parts.append(lin[np.ix_(tt, oth)] - ref[np.ix_(tt, oth)])
        parts.append((lin[t, k] - ref[t, k])[:, None])
        parts.append(self._persist(x_obs, ref, k, t))
        return np.hstack(parts)

    def _patterns(self, x_obs, k, t):
        oth = self._others(k, x_obs.shape[1])
        if not self.use_others:
            return np.zeros(len(t), dtype=int), oth
        flags = ~np.isnan(x_obs[np.ix_(t, oth)])
        return (flags * (1 << np.arange(len(oth)))).sum(1), oth

    # ---- latih
    def fit(self, train: np.ndarray) -> "RidgeMV":
        T, K = train.shape
        conds = self.conditions or [
            {"name": "r10", "kind": "random", "ratio": 0.1}, {"name": "r30", "kind": "random", "ratio": 0.3},
            {"name": "r50", "kind": "random", "ratio": 0.5},
            {"name": "b24", "kind": "block", "length": 24, "simultaneous": False},
            {"name": "b72", "kind": "block", "length": 72, "simultaneous": False},
            {"name": "b24s", "kind": "block", "length": 24, "simultaneous": True},
            {"name": "b72s", "kind": "block", "length": 72, "simultaneous": True}]
        buckets: dict = {}
        d = 0
        for _ in range(self.n_aug):
            for cond in conds:
                mask = make_mask(cond, train.shape, AUG_SEED0 + d, self.block_ratio)
                d += 1
                obs = train.copy(); obs[mask] = np.nan
                lin, _ = linear(obs)
                ref = self._ref(obs, lin)
                for k in range(K):
                    t = np.where(mask[:, k])[0]
                    if len(t) == 0:
                        continue
                    pats, oth = self._patterns(obs, k, t)
                    for p in np.unique(pats):
                        sel = t[pats == p]
                        X = self._X(obs, lin, ref, k, sel, self._cols(int(p), oth))
                        buckets.setdefault((k, int(p)), []).append((X, train[sel, k] - ref[sel, k]))
        self.models = {}
        for (k, p), items in buckets.items():
            X = np.vstack([a for a, _ in items]); y = np.concatenate([b for _, b in items])
            if len(y) >= self.min_rows or p == 0:
                self.models[(k, p)] = _ridge(X, y, self.alpha)
        for k in range(K):
            if (k, 0) not in self.models:
                raise RuntimeError(f"Tidak ada data latih untuk pola 'tanpa variabel lain' pada target {k}.")
        return self

    # ---- imputasi
    def impute(self, x_obs: np.ndarray, **_):
        if not self.models:
            raise RuntimeError("Panggil fit(train) dulu.")
        T, K = x_obs.shape
        mask = np.isnan(x_obs)
        lin, _ = linear(x_obs)
        ref = self._ref(x_obs, lin)
        mu = np.where(mask, lin, x_obs)
        sd = np.ones((T, K))
        for k in range(K):
            t = np.where(mask[:, k])[0]
            if len(t) == 0:
                continue
            pats, oth = self._patterns(x_obs, k, t)
            for p in np.unique(pats):
                sel = t[pats == p]
                used = int(p) if (k, int(p)) in self.models else 0
                X = self._X(x_obs, lin, ref, k, sel, self._cols(used, oth))
                w, b, s = self.models[(k, used)]
                mu[sel, k] = ref[sel, k] + X @ w + b
                sd[sel, k] = s
        return mu, sd
