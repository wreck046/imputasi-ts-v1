"""Bandingkan varian CSDI (default / ablasi / kontrol deterministik) dengan baseline, berpasangan per (kondisi, seed mask).

    python scripts/compare_variants.py --baseline results/v2/baselines_raw_etth1.csv ^
        --variant default="results/csdi_eval/csdi_raw_etth1_model0.csv" ^
        --variant mixblock="results/csdi_eval_mixblock/csdi_raw_etth1_model0.csv" ^
        --variant det_mixblock="results/csdi_eval_det_mixblock/csdi_raw_etth1_model0.csv" ^
        --conditions block24 block72 block24_sim block72_sim --ref mixblock

Hanya (kondisi, seed) yang ada di SEMUA varian yang dipakai. Duplikat: dipakai nsample terbesar; jika ada beberapa model seed (file csdi_raw_*_model{N}.csv
dalam satu glob), metrik dirata-rata lintas model seed per (kondisi, seed mask).
Catatan: untuk varian deterministik hanya MAE/RMSE yang bermakna (CRPS = MAE, interval lebar 0).
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd


def load_variant(spec: str) -> tuple[str, pd.DataFrame]:
    label, path = spec.split("=", 1)
    files = sorted(glob.glob(path.strip('"')))
    if not files:
        raise SystemExit(f"tidak ada file: {path}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "model_seed" not in d:
        d["model_seed"] = 0
    # satu baris per (kondisi, seed mask, model seed): pakai nsample terbesar
    d = d.sort_values("nsample").drop_duplicates(["condition", "seed", "model_seed"], keep="last")
    n_models = d.model_seed.nunique()
    num = d.select_dtypes("number").columns.difference(["seed", "model_seed"])
    d = d.groupby(["condition", "seed"])[list(num)].mean()          # rata-rata lintas model seed
    d.attrs["n_models"] = n_models
    return label, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, nargs="+")
    ap.add_argument("--variant", required=True, action="append")
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--ref", help="label varian acuan untuk selisih berpasangan (default: varian pertama)")
    ap.add_argument("--metric", default="mae", choices=["mae", "rmse"])
    args = ap.parse_args()

    V = dict(load_variant(s) for s in args.variant)
    print("model seed per varian:", {k: v.attrs.get("n_models") for k, v in V.items()})
    ref = args.ref or next(iter(V))
    base = pd.concat([pd.read_csv(f) for f in args.baseline], ignore_index=True)
    B = {m: g.set_index(["condition", "seed"]) for m, g in base.groupby("method")
         if m in ("seasonal_interp", "ridge_mv", "kalman")}
    common = set.intersection(*[set(v.index) for v in V.values()], *[set(b.index) for b in B.values()])
    if args.conditions:
        common = {k for k in common if k[0] in args.conditions}
    keys = sorted(common)
    conds = sorted({k[0] for k in keys})
    m = args.metric
    print(f"metrik={m}; pasangan (kondisi,seed) bersama: {len(keys)}; seed per kondisi: "
          f"{ {c: sum(1 for k in keys if k[0] == c) for c in conds} }\n")

    cols = {**{k: v for k, v in V.items()}, **B}
    table = pd.DataFrame({name: pd.Series({k: d.loc[k, m] for k in keys}) for name, d in cols.items()})
    table.index = pd.MultiIndex.from_tuples(table.index, names=["condition", "seed"])
    print(f"== {m.upper()} rata-rata ==")
    print(table.groupby("condition").mean().loc[conds].round(4).to_string(), "\n")

    print(f"== selisih berpasangan (kolom - {ref}); negatif = kolom lebih baik; menang = jumlah seed kolom lebih baik ==")
    rows = []
    for name in table.columns:
        if name == ref:
            continue
        for c in conds:
            d = (table.loc[c, name] - table.loc[c, ref]).to_numpy()
            rows.append((c, name, d.mean(), d.std(ddof=1) if len(d) > 1 else np.nan, int((d < 0).sum()), len(d)))
    out = pd.DataFrame(rows, columns=["kondisi", "vs", "selisih", "sd", "menang", "n"])
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
