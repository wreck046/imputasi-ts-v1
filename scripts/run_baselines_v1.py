"""Jalankan semua baseline pada semua kondisi missing dan seed.

Contoh:
    python scripts/run_baselines.py                          # penuh, ETTh1
    python scripts/run_baselines.py --methods linear ffill   # subset cepat
    python scripts/run_baselines.py --synthetic              # uji kode tanpa data asli
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imputasi.baselines import get_imputer  # noqa: E402
from imputasi.data import chronological_split, load_csv, make_synthetic_ett  # noqa: E402
from imputasi.evaluate import run_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--conditions", nargs="*", help="nama kondisi saja (default: semua)")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--synthetic", action="store_true", help="data sintetis (HANYA uji kode)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    methods = args.methods or cfg["methods"]
    seeds = args.seeds or cfg["seeds"]
    conds = [c for c in cfg["conditions"] if not args.conditions or c["name"] in args.conditions]

    if args.synthetic:
        df, tag = make_synthetic_ett(), "synthetic"
        print("!! DATA SINTETIS: hasil tidak boleh dipakai untuk paper !!")
    else:
        path = ROOT / cfg["data"]["path"]
        if not path.exists():
            sys.exit(f"Data tidak ada: {path}\nJalankan: python scripts/download_data.py")
        df, tag = load_csv(path, cfg["data"]["date_col"], cfg["data"]["target_cols"]), "etth1"
    sp = chronological_split(df, tuple(cfg["data"]["split"]))
    out = Path(args.out)
    cfg["_cache_dir"] = str(out / "cache")   # cache hasil fit Kalman
    print(f"[{tag}] train={sp.train.shape} val={sp.val.shape} test={sp.test.shape}")

    rows = []
    for m in methods:
        t0 = time.perf_counter()
        imputer = get_imputer(m, cfg, train=sp.train)
        fit_t = time.perf_counter() - t0  # bermakna untuk kalman (0 bila dari cache)
        owner = getattr(imputer, "__self__", None)
        if owner is not None and hasattr(owner, "convergence_summary"):
            print(f"  [kalman] harmonics={owner.harmonics} maxiter={owner.maxiter}: "
                  f"{owner.convergence_summary()}")
        for cond in conds:
            for seed in seeds:
                r = run_one(imputer, m, cond, seed, sp.val, sp.test,
                            cfg["block_missing_ratio"], cfg["interval_level"], columns=sp.columns)
                r["fit_time_s"] = fit_t
                rows.append(r)
            sub = [r for r in rows if r["method"] == m and r["condition"] == cond["name"]]
            mae = sum(r["mae"] for r in sub) / len(sub)
            print(f"  {m:16s} {cond['name']:12s} MAE={mae:.4f}")

    out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / f"baselines_raw_{tag}.csv", index=False)
    agg = (raw.groupby(["condition", "method"])[["mae", "rmse", "crps", "coverage", "width", "impute_time_s"]]
              .agg(["mean", "std"]))
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    order = {c["name"]: i for i, c in enumerate(cfg["conditions"])}
    agg = agg.sort_values(["condition", "mae_mean"], key=lambda s: s.map(order) if s.name == "condition" else s)
    agg.to_csv(out / f"baselines_summary_{tag}.csv", index=False)
    # MAE per variabel (format panjang): kondisi x metode x variabel
    vcols = [c for c in raw.columns if c.startswith("mae_var_")]
    long = raw.melt(id_vars=["condition", "method", "seed"], value_vars=vcols,
                    var_name="variable", value_name="mae_v")
    long["variable"] = long["variable"].str.replace("mae_var_", "", regex=False)
    pv = (long.groupby(["condition", "method", "variable"])["mae_v"].agg(["mean", "std"])
            .rename(columns={"mean": "mae_mean", "std": "mae_std"}).reset_index())
    pv.to_csv(out / f"baselines_per_variable_{tag}.csv", index=False)

    print("\n" + agg.round(4).to_string(index=False))
    print(f"\nTersimpan di {out}")


if __name__ == "__main__":
    main()
