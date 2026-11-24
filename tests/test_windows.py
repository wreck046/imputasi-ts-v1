import numpy as np
import pytest
from imputasi import metrics as M
from imputasi.csdi_eval import evaluate_condition
from imputasi.data import chronological_split, make_synthetic_ett
from imputasi.masks import make_mask
from imputasi.windows import owner_windows, stitch, window_starts, window_timepoints


def test_starts_cover_and_end_aligned():
    for T, L, s in [(3484, 168, 24), (500, 100, 37), (168, 168, 24)]:
        st = window_starts(T, L, s)
        assert st[0] == 0 and st[-1] == T - L
        cover = np.zeros(T, bool)
        for a in st:
            cover[a:a + L] = True
        assert cover.all()


def test_owner_is_most_central_and_stitch_assigns_one_window_per_t():
    T, L = 400, 100
    st = window_starts(T, L, 25)
    own = owner_windows(T, L, st)
    for t in [0, 50, 99, 150, 399]:
        cands = [(min(t - s, s + L - 1 - t), -w) for w, s in enumerate(st) if s <= t < s + L]
        best = max(cands)
        assert own[t] == -best[1]
    # stitch: tiap jendela berisi nilai = id jendela -> hasil harus sama dengan owner
    samples = np.zeros((len(st), 2, L, 3))
    for w in range(len(st)):
        samples[w] = w
    out = stitch(samples, st, T)
    assert (out[0, :, 0] == own).all()


def test_timepoints_modes():
    st = np.array([0, 24])
    rel = window_timepoints(st, 10, mode="relative")
    assert (rel == np.arange(10)).all()
    how = window_timepoints(st, 10, offset=160, mode="hour_of_week")
    assert how[0, 0] == 160 and how[0, 8] == 0 and how[1, 0] == (160 + 24) % 168


def test_crps_samples_matches_bruteforce_and_gaussian():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(30, 2)); m = rng.random((30, 2)) < 0.5
    x = rng.normal(size=(40, 30, 2))
    xs = x[:, m]; yy = y[m]; s = xs.shape[0]
    brute = np.mean([np.abs(xs[:, n] - yy[n]).mean()
                     - sum(abs(xs[i, n] - xs[j, n]) for i in range(s) for j in range(s) if i != j) / (2 * s * (s - 1))
                     for n in range(xs.shape[1])])
    assert np.isclose(M.crps_samples(y, x, m), brute)
    # sampel besar dari N(0,1), y=0 -> mendekati nilai tertutup (sqrt2-1)/sqrt(pi)
    big = rng.normal(size=(400, 1, 1)); z = np.zeros((1, 1)); mm = np.ones((1, 1), bool)
    assert abs(M.crps_samples(z, big, mm) - (np.sqrt(2) - 1) / np.sqrt(np.pi)) < 0.03


def _oracle_setup():
    sp = chronological_split(make_synthetic_ett(n=2000), (0.7, 0.1, 0.2))
    return sp.test                      # (400, 7)


def test_eval_oracle_and_no_leakage():
    test = _oracle_setup()
    L, stride = 96, 24
    seen = {}

    def oracle(w_obs, w_cond, w_tp, n):
        # tidak boleh melihat kebenaran di titik hilang
        assert np.all(w_obs[w_cond == 0] == 0.0)
        st = window_starts(len(test), L, stride)
        truth = np.stack([test[a:a + L] for a in st]).astype(np.float32)
        seen["n"] = len(st)
        return np.repeat(truth[:, None], n, axis=1)

    cond = {"name": "block24", "kind": "block", "length": 24, "simultaneous": False}
    row = evaluate_condition(oracle, test, cond, 0, 0.2, window_len=L, stride=stride, nsample=5)
    assert row["mae"] < 1e-6 and row["width"] < 1e-6   # interval selebar nol (oracle tanpa variansi)
    # mask identik dengan yang dipakai baseline
    assert row["n_missing"] == int(make_mask(cond, test.shape, 0, 0.2).sum())


def test_eval_uses_context_zero_predictor_has_expected_mae():
    test = _oracle_setup()
    L = 96

    def zeros(w_obs, w_cond, w_tp, n):
        return np.zeros((len(w_obs), n, L, test.shape[1]), dtype=np.float32)

    cond = {"name": "random30", "kind": "random", "ratio": 0.3}
    row = evaluate_condition(zeros, test, cond, 1, 0.2, window_len=L, stride=24, nsample=3)
    mask = make_mask(cond, test.shape, 1, 0.2)
    assert np.isclose(row["mae"], np.abs(test[mask]).mean())
    assert {"mae_var_0", "mae_var_6"} <= set(row)
