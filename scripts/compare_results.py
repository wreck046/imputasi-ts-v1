"""Gabungkan hasil baseline dan CSDI; cetak tabel rata-rata dan selisih BERPASANGAN per seed mask.

    python scripts/compare_results.py --baseline results/v2/baselines_raw_etth1.csv \
        --csdi results/csdi_eval/csdi_raw_etth1_model*.csv
Selisih = CSDI - baseline (negatif = CSDI lebih baik). Seed model dirata-ratakan dulu per (kondisi, seed mask).
"""
from __future__ import annotations

import argparse
import glob

import pandas as pd


def load(args):
    base = pd.concat([pd.read_csv(f) for f in args.baseline], ignore_index=True)
    files = sorted(sum((glob.glob(p) for p in args.csdi), []))
    if not files:
        raise SystemExit("Tidak ada file CSDI yang cocok.")
    cs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    combos = cs.groupby(["nsample", "stride"]).size()
    if len(combos) > 1 or args.nsample or args.stride:
        ns = args.nsample or int(combos.sort_values().index[-1][0])
        st = args.stride or int(combos.sort_values().index[-1][1])
        print(f"PERINGATAN: ada beberapa kombinasi (nsample, stride) di CSV: {dict(combos)}. Dipakai nsample={ns}, stride={st}.")
        cs = cs[(cs["nsample"] == ns) & (cs["stride"] == st)]
    n_models = cs["model_seed"].nunique()
    cs = cs.groupby(["method", "condition", "seed"], as_index=False)[["mae", "rmse", "crps", "coverage", "width"]].mean()
    return base, cs, n_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True, help="satu atau lebih CSV raw baseline")
    ap.add_argument("--csdi", nargs="+", required=True)
    ap.add_argument("--nsample", type=int, help="filter; default = kombinasi paling banyak")
    ap.add_argument("--stride", type=int)
    args = ap.parse_args()
    base, cs, n_models = load(args)
    cols = ["method", "condition", "seed", "mae", "rmse", "crps", "coverage", "width"]
    allr = pd.concat([base[cols], cs[cols]], ignore_index=True)
    allr = allr[allr["condition"].isin(cs["condition"].unique())]

    for met in ("mae", "crps"):
        t = allr.groupby(["condition", "method"])[met].agg(["mean", "std"]).round(4)
        wide = t["mean"].unstack("method")
        print(f"\n== {met.upper()} rata-rata (CSDI: rata-rata {n_models} model) ==")
        print(wide.to_string())

    print("\n== Selisih berpasangan CSDI - baseline (negatif = CSDI lebih baik) ==")
    rows = []
    piv = {m: allr.pivot_table(index=["condition", "seed"], columns="method", values=m) for m in ("mae", "crps")}
    for m, p in piv.items():
        for b in [c for c in p.columns if c != "csdi"]:
            d = (p["csdi"] - p[b]).dropna()
            for cond, g in d.groupby(level=0):
                rows.append({"metrik": m, "kondisi": cond, "vs": b, "selisih": g.mean(),
                             "sd": g.std(ddof=1), "n": len(g), "csdi_lebih_baik": int((g < 0).sum())})
    r = pd.DataFrame(rows).round(4)
    print(r[r["vs"].isin(["linear", "seasonal_interp", "kalman", "ridge_uv", "ridge_mv", "ridge_mv_lag"])].to_string(index=False))


if __name__ == "__main__":
    main()
