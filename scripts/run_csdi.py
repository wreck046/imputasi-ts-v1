"""Evaluasi satu model CSDI terlatih pada kondisi missing yang SAMA dengan baseline.

    python scripts/run_csdi.py --model-dir results/csdi/model_seed0
    python scripts/run_csdi.py --model-dir results/csdi/model_seed0 --conditions block24 --seeds 0 --nsample 20
    python scripts/run_csdi.py --model-dir ... --split val --conditions block24_sim block72_sim --seeds 0 1 --nsample 10 --out results/csdi_val_mixblock
Bisa dilanjutkan (resume): baris (kondisi, seed mask) yang sudah ada di CSV dilewati.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imputasi.csdi_eval import evaluate_condition  # noqa: E402
from imputasi.csdi_torch import build_model, make_sampler  # noqa: E402
from imputasi.data import chronological_split, load_csv, make_synthetic_ett  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "csdi_ett.yaml"))
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int, help="seed MASK (default: sama dengan baseline)")
    ap.add_argument("--nsample", type=int)
    ap.add_argument("--stride", type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ROOT / "results" / "csdi_eval"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="val = evaluasi pada split VALIDASI (untuk menguji aturan pemilihan model tanpa menyentuh test)")
    args = ap.parse_args()

    mdir = Path(args.model_dir)
    used = json.loads((mdir / "config_used.json").read_text())      # konfigurasi SAAT latihan
    meta = json.loads((mdir / "meta.json").read_text())
    cfg = yaml.safe_load(open(args.config))
    base = yaml.safe_load(open(ROOT / used["base_config"]))
    ad, ev = used["adapter"], cfg["eval"]
    nsample = args.nsample or ev["nsample"]
    stride = args.stride or ev["stride"]
    seeds = args.seeds if args.seeds is not None else base["seeds"]
    conds = [c for c in base["conditions"] if not args.conditions or c["name"] in args.conditions]

    if args.synthetic != meta["synthetic"]:
        sys.exit("Model dilatih dengan data sintetis/asli yang berbeda dari --synthetic saat ini.")
    if args.synthetic:
        df, tag = make_synthetic_ett(), "synthetic"
    else:
        df, tag = load_csv(ROOT / base["data"]["path"], base["data"]["date_col"], base["data"]["target_cols"]), base["data"].get("name", "etth1")
    sp = chronological_split(df, tuple(base["data"]["split"]))
    if not (np.allclose(sp.mean, meta["mean"]) and np.allclose(sp.std, meta["std"])):
        sys.exit("Statistik z-score berbeda dari saat latihan: data/split tidak sama. Hentikan.")

    data = sp.val if args.split == "val" else sp.test
    model = build_model(used, args.device, sp.test.shape[1], ROOT / used["csdi_path"], tuple(ad["block_lengths"]))
    model.load_state_dict(torch.load(mdir / "model.pth", map_location=args.device))
    sampler = make_sampler(model, args.device, ev["batch_size"])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.split == "val":
        tag += "_val"                            # berkas terpisah; tidak bercampur dengan hasil test
    csv = out / f"csdi_raw_{tag}_model{meta['seed']}.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["condition"], int(r["seed"]), int(r["nsample"]), int(r["stride"])) for r in rows}
    tp_offset = len(sp.train) if args.split == "val" else len(sp.train) + len(sp.val)

    for cond in conds:
        for seed in seeds:
            if (cond["name"], seed, nsample, stride) in done:
                continue
            torch.manual_seed(meta["seed"] * 1000 + seed)
            row = evaluate_condition(
                sampler, data, cond, seed, base["block_missing_ratio"],
                window_len=ad["window_len"], stride=stride, nsample=nsample,
                level=base["interval_level"], columns=sp.columns,
                tp_mode=ad["timepoints"], tp_offset=tp_offset)
            if meta.get("deterministic"):
                row["method"] = "csdi_det"      # kontrol non-diffusion: CRPS/coverage/width tidak bermakna
            row.update({"model_seed": meta["seed"], "train_strategy": meta["target_strategy"],
                        "epochs": meta["epochs"], "csdi_commit": meta["csdi_commit"], "split": args.split})
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv, index=False)
            print(f"  {cond['name']:12s} mask-seed {seed}  MAE={row['mae']:.4f}  CRPS={row['crps']:.4f}  "
                  f"cov={row['coverage']:.3f}  {row['impute_time_s']:.0f}s")
    print(f"Tersimpan: {csv}")


if __name__ == "__main__":
    main()
