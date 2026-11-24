"""Gambar imputasi satu blok hilang serentak untuk beberapa model CSDI (mis. model baik vs gagal).

    python scripts/plot_imputation.py --config configs/csdi_ett_mixblock.yaml --condition block72_sim --mask-seed 0 ^
        --models "model7 (baik)=results/csdi_mixblock/model_seed7" "model8 (gagal)=results/csdi_mixblock/model_seed8" ^
        --variables HUFL MUFL OT --baselines seasonal_interp ridge_mv --out results/diagnostik_hufl.png
(tulis dalam SATU baris di PowerShell). Hasil: PNG + CSV ringkasan MAE/bias/cakupan per variabel di dalam blok.
Pilih blok dengan --block-index (urutan blok serentak pada mask) atau --t-start.
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

from imputasi.baselines import get_imputer  # noqa: E402
from imputasi.csdi_torch import build_model, make_sampler  # noqa: E402
from imputasi.data import chronological_split, load_csv  # noqa: E402
from imputasi.diagnostics import block_summary, find_blocks, impute_range, render_figure  # noqa: E402
from imputasi.masks import make_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "csdi_ett_mixblock.yaml"))
    ap.add_argument("--models", nargs="+", required=True, help='"label=folder_model" (boleh lebih dari satu)')
    ap.add_argument("--condition", default="block72_sim")
    ap.add_argument("--mask-seed", type=int, default=0)
    ap.add_argument("--block-index", type=int, default=0)
    ap.add_argument("--t-start", type=int, help="awal rentang plot pada split test (menggantikan --block-index)")
    ap.add_argument("--context", type=int, default=48, help="jam konteks di kiri-kanan blok")
    ap.add_argument("--variables", nargs="+", default=["HUFL", "MUFL", "OT"])
    ap.add_argument("--nsample", type=int, default=50)
    ap.add_argument("--baselines", nargs="*", default=[])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ROOT / "results" / "diagnostik.png"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base = yaml.safe_load(open(ROOT / cfg["base_config"]))
    df = load_csv(ROOT / base["data"]["path"], base["data"]["date_col"], base["data"]["target_cols"])
    sp = chronological_split(df, tuple(base["data"]["split"]))
    cond = next(c for c in base["conditions"] if c["name"] == args.condition)
    mask = make_mask(cond, sp.test.shape, args.mask_seed, base["block_missing_ratio"])
    blocks = find_blocks(mask, simultaneous=bool(cond.get("simultaneous", False)))
    if args.t_start is not None:
        t0 = args.t_start; t1 = t0 + cond["length"]
    else:
        t0, t1 = blocks[args.block_index]
    t_lo, t_hi = max(0, t0 - args.context), min(len(sp.test), t1 + args.context)
    print(f"{args.condition}, seed mask {args.mask_seed}: blok [{t0}, {t1}) dari {len(blocks)} blok; plot [{t_lo}, {t_hi})")

    base_mu = {}
    for bn in args.baselines:
        imp = get_imputer(bn, dict(base, _cache_dir=str(ROOT / "results" / "cache")), train=sp.train)
        x = sp.test.copy(); x[mask] = np.nan
        base_mu[bn] = imp(x)[0]

    samples, rows = {}, []
    tp_offset = len(sp.train) + len(sp.val)
    for item in args.models:
        label, mdir = item.split("=", 1)
        mdir = Path(mdir)
        used = json.loads((mdir / "config_used.json").read_text())
        meta = json.loads((mdir / "meta.json").read_text())
        ad = used["adapter"]
        model = build_model(used, args.device, sp.test.shape[1], ROOT / used["csdi_path"], tuple(ad["block_lengths"]))
        model.load_state_dict(torch.load(mdir / "model.pth", map_location=args.device))
        sampler = make_sampler(model, args.device, cfg["eval"]["batch_size"])
        torch.manual_seed(meta["seed"] * 1000 + args.mask_seed)
        s = impute_range(sampler, sp.test, mask, t_lo, t_hi, window_len=ad["window_len"], stride=cfg["eval"]["stride"],
                         nsample=args.nsample, tp_mode=ad["timepoints"], tp_offset=tp_offset)
        samples[label] = s
        for r in block_summary(sp.test, mask, s, t_lo, t_hi, sp.columns):
            rows.append(dict(model=label, **r))
    for bn, mu in base_mu.items():
        y, m = sp.test[t_lo:t_hi], mask[t_lo:t_hi]
        for j, c in enumerate(sp.columns):
            if m[:, j].any():
                rows.append({"model": bn, "variabel": c, "mae": float(np.abs(mu[t_lo:t_hi, j][m[:, j]] - y[m[:, j], j]).mean()),
                             "bias": float((mu[t_lo:t_hi, j][m[:, j]] - y[m[:, j], j]).mean())})

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    render_figure(sp.test, mask, samples, sp.columns, t_lo, t_hi, args.variables, out, baselines=base_mu or None,
                  title=f"{args.condition}, seed mask {args.mask_seed}, blok [{t0}, {t1})")
    csv = out.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"Tersimpan: {out}\n           {csv}")


if __name__ == "__main__":
    main()
