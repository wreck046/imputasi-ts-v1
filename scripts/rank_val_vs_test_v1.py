"""Apakah MAE VALIDASI memprediksi MAE TEST antar-model? (menguji aturan pemilihan model)

    python scripts/rank_val_vs_test.py \
        --val  "results/csdi_val_mixblock/csdi_raw_etth1_val_model*.csv" \
        --test "results/csdi_eval_mixblock/csdi_raw_etth1_model*.csv" \
        --conditions block24_sim block72_sim --test-seeds 0 1 2

Per (kondisi, model) dihitung rata-rata MAE atas seed mask. Lalu dilaporkan:
  * tabel val vs test per model dan peringkatnya,
  * korelasi Spearman (n kecil: jangan dipercaya sendirian),
  * hasil aturan "pilih model dengan MAE val terendah": MAE test model terpilih vs rata-rata/terbaik/terburuk.
Aturan dipilih dari VALIDASI saja; test hanya dipakai untuk menilai aturan itu.
"""
from __future__ import annotations

import argparse
import glob

import pandas as pd


def load(pattern: str, conditions, seeds=None, nsample=None) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"Tidak ada berkas untuk pola: {pattern}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["condition"].isin(conditions)]
    if seeds is not None:
        df = df[df["seed"].isin(seeds)]
    if nsample is not None:
        df = df[df["nsample"] == nsample]
    else:  # bila ada beberapa nsample, pakai yang terbesar per (model, kondisi, seed)
        df = df.sort_values("nsample").groupby(["model_seed", "condition", "seed"], as_index=False).last()
    return df.groupby(["condition", "model_seed"], as_index=False)["mae"].mean()


def analyse(val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    m = val.merge(test, on=["condition", "model_seed"], suffixes=("_val", "_test"))
    rows = []
    for cond, g in m.groupby("condition"):
        g = g.copy()
        g["rank_val"] = g["mae_val"].rank()
        g["rank_test"] = g["mae_test"].rank()
        sel = g.loc[g["mae_val"].idxmin()]
        rows.append({
            "condition": cond, "n_model": len(g),
            "spearman": g["mae_val"].corr(g["mae_test"], method="spearman"),
            "selected_model": int(sel["model_seed"]), "test_selected": sel["mae_test"],
            "test_mean": g["mae_test"].mean(), "test_best": g["mae_test"].min(), "test_worst": g["mae_test"].max(),
        })
        print(f"\n== {cond} ==")
        print(g.sort_values("mae_val")[["model_seed", "mae_val", "rank_val", "mae_test", "rank_test"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--conditions", nargs="+", default=["block24_sim", "block72_sim"])
    ap.add_argument("--val-seeds", nargs="*", type=int)
    ap.add_argument("--test-seeds", nargs="*", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    val = load(a.val, a.conditions, a.val_seeds)
    test = load(a.test, a.conditions, a.test_seeds)
    res = analyse(val, test)
    print("\n== Aturan 'pilih MAE val terendah' ==")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nCatatan: n model kecil; konfirmasi pada model BARU sebelum mengklaim aturan ini bekerja.")


if __name__ == "__main__":
    main()
