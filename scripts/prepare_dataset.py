"""Siapkan dataset kedua (format sama dengan ETTh1) dan buat config baseline + CSDI-nya.

Contoh (Electricity UCI, LD2011_2014.txt dari https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014):
    python scripts/prepare_dataset.py --name electricity --raw data/raw/LD2011_2014.txt --format uci_electricity ^
        --k 7 --start 2012-01-01 --last 17420
(tulis dalam SATU baris di PowerShell; `^` hanya contoh). CSV lain (mis. Traffic/Weather bergaya Autoformer):
    python scripts/prepare_dataset.py --name traffic --raw data/raw/traffic.csv --format generic --drop OT --k 7

Keluaran: data/processed/<name>.csv, configs/<name>_baseline.yaml, configs/csdi_<name>_mixblock.yaml,
          results/<name>_dataset_info.json. Tidak ada data yang diunduh oleh skrip ini.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imputasi.prepare import make_configs, prepare, read_raw, split_shift_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="tag dataset; dipakai di nama berkas hasil")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--format", default="generic", choices=["generic", "uci_electricity"])
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--k", type=int, default=7, help="jumlah variabel (7 = sama dengan ETTh1, beban GPU sama)")
    ap.add_argument("--select", default="even", choices=["even", "first"])
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--last", type=int, help="ambil N jam TERAKHIR (17420 = panjang ETTh1)")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--max-zero-frac", type=float, default=0.01)
    ap.add_argument("--nsample", type=int, default=20)
    args = ap.parse_args()

    raw = read_raw(args.raw, args.format, args.date_col)
    df, info = prepare(raw, args.k, args.select, args.start, args.end, args.last, args.drop, args.max_zero_frac)

    out_csv = ROOT / "data" / "processed" / f"{args.name}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(out_csv, index=False)

    b0 = yaml.safe_load(open(ROOT / "configs" / "baseline.yaml"))
    c0 = yaml.safe_load(open(ROOT / "configs" / "csdi_ett_mixblock.yaml"))
    b, c = make_configs(args.name, out_csv.relative_to(ROOT).as_posix(), info["columns"], b0, c0, nsample=args.nsample)
    (ROOT / "configs" / f"{args.name}_baseline.yaml").write_text(
        "# DIHASILKAN oleh scripts/prepare_dataset.py (protokol diperkecil). Periksa kalman.harmonics/maxiter dengan tune_kalman.py.\n"
        + yaml.safe_dump(b, sort_keys=False, allow_unicode=True))
    (ROOT / "configs" / f"csdi_{args.name}_mixblock.yaml").write_text(
        "# DIHASILKAN oleh scripts/prepare_dataset.py. Blok train/diffusion/model identik dengan ETTh1 (hyperparameter dikunci).\n"
        + yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
    shift = split_shift_report(df)
    info["split_shift"] = shift.round(3).to_dict("records")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{args.name}_dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    print(f"[{args.name}] {info['n_rows']} baris per jam, {info['first']} .. {info['last']}")
    print(f"  kolom ({len(info['columns'])}, dipilih '{args.select}' dari {info['n_eligible']} yang memenuhi syarat): {info['columns']}")
    print(f"  baris semua-nol diperbaiki (artefak DST): {info['n_allzero_rows_fixed']}")
    print("  pergeseran z-score (statistik train) pada val/test:\n" + shift.round(2).to_string(index=False))
    print(f"Tersimpan: {out_csv}\n  configs/{args.name}_baseline.yaml\n  configs/csdi_{args.name}_mixblock.yaml")


if __name__ == "__main__":
    main()
