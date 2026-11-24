"""Pilih konfigurasi Kalman (jumlah harmonik) memakai HANYA split validasi.

Prosedur:
  1. Untuk tiap harmonics: fit parameter pada TRAIN (maxiter dari argumen).
  2. Sembunyikan titik pada deret VALIDASI (seed terpisah dari seed test) pada kondisi tuning.
  3. Ukur MAE (dan CRPS) pada titik tersembunyi; pilih harmonics dengan MAE rata-rata terendah.
Test tidak disentuh. Hasil disalin MANUAL ke configs/baseline.yaml (bagian kalman).

Contoh:
    python scripts/tune_kalman.py                       # harmonics 3 dan 6, maxiter 200
    python scripts/tune_kalman.py --harmonics 3 6 12    # 12 bisa sangat lambat (jalankan semalam)
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

from imputasi import metrics as M  # noqa: E402
from imputasi.baselines import KalmanImputer  # noqa: E402
from imputasi.data import chronological_split, load_csv, make_synthetic_ett  # noqa: E402
from imputasi.masks import make_mask  # noqa: E402

TUNE_SEED_OFFSET = 20_000  # berbeda dari seed test (0..) dan mask kalibrasi validasi (10_000+)
TUNE_CONDS = [
    {"name": "random30", "kind": "random", "ratio": 0.30},
    {"name": "block24", "kind": "block", "length": 24, "simultaneous": False},
    {"name": "block72", "kind": "block", "length": 72, "simultaneous": False},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    ap.add_argument("--harmonics", nargs="*", type=int, default=[3, 6])
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.synthetic:
        df = make_synthetic_ett()
        print("!! DATA SINTETIS: hanya uji kode !!")
    else:
        path = ROOT / cfg["data"]["path"]
        if not path.exists():
            sys.exit(f"Data tidak ada: {path}")
        df = load_csv(path, cfg["data"]["date_col"], cfg["data"]["target_cols"])
    sp = chronological_split(df, tuple(cfg["data"]["split"]))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    period, ratio = cfg.get("period", 24), cfg["block_missing_ratio"]

    rows = []
    for h in args.harmonics:
        t0 = time.perf_counter()
        imp = KalmanImputer(period, h, args.maxiter).fit(sp.train, cache_dir=cache)
        fit_t = time.perf_counter() - t0
        print(f"[harmonics={h}] fit {fit_t:.0f}s, {imp.convergence_summary()}")
        for cond in TUNE_CONDS:
            for seed in args.seeds:
                mask = make_mask(cond, sp.val.shape, seed + TUNE_SEED_OFFSET, ratio)
                obs = sp.val.copy(); obs[mask] = float("nan")
                mu, sd = imp.impute(obs)
                rows.append({
                    "harmonics": h, "maxiter": args.maxiter, "condition": cond["name"], "seed": seed,
                    "val_mae": M.mae(sp.val, mu, mask),
                    "val_crps": M.crps_gaussian(sp.val, mu, sd, mask),
                    "fit_time_s": fit_t,
                    "n_converged": sum(1 for c in imp.converged_ if c is True),
                })
    df_res = pd.DataFrame(rows)
    df_res.to_csv(out / "kalman_tuning_val.csv", index=False)

    by_cond = df_res.groupby(["harmonics", "condition"])["val_mae"].mean().unstack()
    by_cond["MEAN"] = by_cond.mean(axis=1)
    print("\nMAE VALIDASI (rata-rata seed):\n" + by_cond.round(4).to_string())
    best = by_cond["MEAN"].idxmin()   # seri -> idxmin memilih yang pertama (harmonics terkecil)
    print(f"\nTerpilih (MAE validasi rata-rata terendah): harmonics={best}")
    print("Salin ke configs/baseline.yaml:\n  kalman:\n"
          f"    harmonics: {best}\n    maxiter: {args.maxiter}")
    if best == max(args.harmonics) and len(args.harmonics) > 1:
        print("CATATAN: nilai terbesar yang dicoba menang; kemungkinan masih ada yang lebih baik di atasnya.")


if __name__ == "__main__":
    main()
