import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from imputasi import diagnostics as D
from imputasi import prepare as P
from imputasi.masks import make_mask


def _raw(n_days=60, k=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-01", periods=n_days * 96, freq="15min")
    t = np.arange(len(idx))
    data = {}
    for j in range(k):
        data[f"MT_{j:03d}"] = 5 + np.sin(2 * np.pi * t / 96 + j) + rng.normal(0, .1, len(t))
    df = pd.DataFrame(data, index=idx)
    df.iloc[:2000, 3] = 0.0            # klien yang 'belum ada' -> banyak nol -> tidak memenuhi syarat
    df.iloc[1000:1004, :] = 0.0        # artefak semua-nol (DST)
    df.index.name = "date"
    return df


def test_hourly_and_eligibility_and_selection_are_deterministic():
    df = _raw()
    h = P.to_hourly(df)
    assert len(h) == len(df) // 4 and pd.Series(h.index).diff().median() == pd.Timedelta(hours=1)
    elig = P.eligible_columns(h)
    assert "MT_003" not in elig
    a = P.select_columns(elig, 7); b = P.select_columns(list(reversed(elig)), 7)
    assert a == b and len(a) == 7 and a[0] == sorted(elig)[0] and a[-1] == sorted(elig)[-1]


def test_prepare_pipeline_no_nan_and_dst_fix_and_last():
    out, info = P.prepare(_raw(), k=7, last=800)
    assert len(out) == 800 and not out.isna().any().any() and out.shape[1] == 7
    assert info["n_allzero_rows_fixed"] >= 1 and info["n_eligible"] == 19


def test_split_shift_report_flags_shift():
    x = np.concatenate([np.random.default_rng(0).normal(0, 1, 700), np.random.default_rng(1).normal(0, 1, 100),
                        np.random.default_rng(2).normal(3, 1, 200)])
    r = P.split_shift_report(pd.DataFrame({"a": x, "b": np.random.default_rng(3).normal(size=1000)}))
    assert r.loc[0, "test_mean"] > 2.5 and abs(r.loc[0, "val_mean"]) < 0.5 and abs(r.loc[1, "test_mean"]) < 0.5


def test_make_configs_reduced_protocol():
    root = Path(__file__).resolve().parents[1]
    b0 = yaml.safe_load(open(root / "configs" / "baseline.yaml"))
    c0 = yaml.safe_load(open(root / "configs" / "csdi_ett_mixblock.yaml"))
    b, c = P.make_configs("elec", "data/processed/elec.csv", ["a", "b"], b0, c0)
    assert b["data"]["name"] == "elec" and b["data"]["target_cols"] == ["a", "b"]
    assert [x["name"] for x in b["conditions"]] == ["block24", "block72", "block24_sim", "random20"] or \
        {x["name"] for x in b["conditions"]} == {"random20", "block24", "block72", "block24_sim"}
    assert c["base_config"] == "configs/elec_baseline.yaml" and c["eval"]["nsample"] == 20
    assert c["diffusion"] == c0["diffusion"]          # hyperparameter CSDI tidak berubah


def test_find_blocks_and_impute_range_matches_full_pipeline():
    cond = {"name": "b", "kind": "block", "length": 24, "simultaneous": True}
    T, K = 600, 3
    mask = make_mask(cond, (T, K), 0, 0.2)
    blocks = D.find_blocks(mask)
    assert all(e - s >= 24 for s, e in blocks) and all(mask[s:e].all() for s, e in blocks)
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(T, K))

    def sampler(obs, cond_, tp, ns):           # sampler deterministik: mengembalikan 7.0 + indeks jendela-agnostik
        W, L, K_ = obs.shape
        return np.full((W, ns, L, K_), 7.0, dtype=np.float32)

    s0, e0 = blocks[0]
    out = D.impute_range(sampler, truth, mask, s0 - 10, e0 + 10, window_len=100, stride=25, nsample=4)
    assert out.shape == (4, e0 - s0 + 20, K) and np.allclose(out, 7.0)
    rows = D.block_summary(truth, mask, out, s0 - 10, e0 + 10, ["a", "b", "c"])
    assert len(rows) == 3 and abs(rows[0]["bias"] - (7.0 - truth[s0:e0, 0].mean())) < 1e-6


def test_render_figure_writes_png(tmp_path):
    T, K = 120, 2
    truth = np.random.default_rng(0).normal(size=(T, K))
    mask = np.zeros((T, K), bool); mask[40:70] = True
    smp = np.random.default_rng(1).normal(size=(10, 60, K))
    D.render_figure(truth, mask, {"m1": smp, "m2": smp + 1}, ["a", "b"], 30, 90, ["a", "b"], tmp_path / "x.png",
                    baselines={"bl": truth})
    assert (tmp_path / "x.png").stat().st_size > 1000
