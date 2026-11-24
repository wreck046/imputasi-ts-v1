"""Latih satu model CSDI (satu seed) pada split train ETTh1.

    python scripts/train_csdi.py --seed 0 --epochs 1            # uji cepat
    python scripts/train_csdi.py --seed 0                       # epochs dari configs/csdi_ett.yaml
Keluaran: results/csdi/model_seed{seed}/{model.pth, config_used.json, meta.json}
Catatan: utils.train milik CSDI hanya menyimpan model AKHIR (bukan yang terbaik di validasi).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imputasi.csdi_torch import TrainWindows, ValWindows, build_model, ensure_csdi_on_path  # noqa: E402
from imputasi.data import chronological_split, load_csv, make_synthetic_ett  # noqa: E402


def git_head(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "csdi_ett.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ROOT / "results" / "csdi"))
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base = yaml.safe_load(open(ROOT / cfg["base_config"]))
    ad = cfg["adapter"]
    csdi_path = ROOT / cfg["csdi_path"]
    ensure_csdi_on_path(csdi_path)
    from utils import train as csdi_train  # dari repo CSDI

    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    if args.synthetic:
        df = make_synthetic_ett(); print("!! DATA SINTETIS (hanya uji alat) !!")
    else:
        df = load_csv(ROOT / base["data"]["path"], base["data"]["date_col"], base["data"]["target_cols"])
    sp = chronological_split(df, tuple(base["data"]["split"]))
    K, L = sp.train.shape[1], ad["window_len"]

    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(TrainWindows(sp.train, L, ad["windows_per_epoch"], ad["timepoints"], args.seed),
                              batch_size=bs, shuffle=False, drop_last=True)
    valid_loader = DataLoader(ValWindows(sp.val, L, ad["val_windows_stride"], ad["val_missing_ratio"],
                                         ad["timepoints"], tp_offset=len(sp.train)),
                              batch_size=bs, shuffle=False)
    model = build_model(cfg, args.device, K, csdi_path, tuple(ad["block_lengths"]))

    folder = Path(args.out) / f"model_seed{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config_used.json").write_text(json.dumps(cfg, indent=2))
    meta = {
        "seed": args.seed, "csdi_commit": git_head(csdi_path), "torch": torch.__version__,
        "device": args.device, "synthetic": bool(args.synthetic), "epochs": cfg["train"]["epochs"],
        "n_train": len(sp.train), "n_val": len(sp.val), "n_test": len(sp.test),
        "mean": sp.mean.tolist(), "std": sp.std.tolist(), "columns": sp.columns,
        "target_strategy": cfg["model"]["target_strategy"], "timepoints": ad["timepoints"],
        "window_len": L,
        "deterministic": bool(cfg["model"].get("deterministic", False)),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Latih seed {args.seed}: {cfg['train']['epochs']} epoch x {len(train_loader)} iterasi -> {folder}")

    csdi_train(model, cfg["train"], train_loader, valid_loader=valid_loader,
               valid_epoch_interval=20, foldername=str(folder))
    print(f"Selesai. Model: {folder / 'model.pth'}")


if __name__ == "__main__":
    main()
