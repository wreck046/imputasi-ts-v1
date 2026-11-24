"""Ukur: GPU/VRAM, waktu 1 epoch latih, waktu imputasi per jendela-sampel, lalu proyeksikan
total waktu eksperimen. Jalankan SEBELUM menetapkan epochs/nsample/jumlah seed.

    python scripts/csdi_benchmark.py                # data ETTh1 asli
    python scripts/csdi_benchmark.py --synthetic    # uji alat tanpa data asli
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imputasi.csdi_torch import TrainWindows, build_model, ensure_csdi_on_path, make_sampler  # noqa: E402
from imputasi.data import chronological_split, load_csv, make_synthetic_ett  # noqa: E402
from imputasi.windows import window_starts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "csdi_ett.yaml"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--impute-batches", type=int, default=2, help="jumlah batch jendela untuk ukur imputasi")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base = yaml.safe_load(open(ROOT / cfg["base_config"]))
    ad, ev = cfg["adapter"], cfg["eval"]
    csdi_path = ROOT / cfg["csdi_path"]
    ensure_csdi_on_path(csdi_path)

    print(f"torch {torch.__version__} | CUDA tersedia: {torch.cuda.is_available()}")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            sys.exit("CUDA tidak tersedia. Pasang PyTorch build CUDA (lihat README) dan cek `nvidia-smi`.")
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} | VRAM {p.total_memory / 2**30:.1f} GiB")
        torch.cuda.reset_peak_memory_stats()

    if args.synthetic:
        df = make_synthetic_ett(); print("!! DATA SINTETIS (hanya uji alat) !!")
    else:
        df = load_csv(ROOT / base["data"]["path"], base["data"]["date_col"], base["data"]["target_cols"])
    sp = chronological_split(df, tuple(base["data"]["split"]))
    K, L = sp.train.shape[1], ad["window_len"]
    model = build_model(cfg, args.device, K, csdi_path, tuple(ad["block_lengths"]))
    print(f"parameter model: {sum(p.numel() for p in model.parameters()):,}")

    # --- 1 epoch latih (memakai utils.train milik CSDI, sama dengan latihan sebenarnya)
    from torch.utils.data import DataLoader
    from utils import train as csdi_train  # dari repo CSDI

    bs = cfg["train"]["batch_size"]
    loader = DataLoader(TrainWindows(sp.train, L, ad["windows_per_epoch"], ad["timepoints"], 0),
                        batch_size=bs, shuffle=False, drop_last=True)
    n_it = len(loader)
    one = dict(cfg["train"]); one["epochs"] = 1
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    csdi_train(model, one, loader, valid_loader=None, foldername="")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t_epoch = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if args.device.startswith("cuda") else float("nan")
    print(f"\n[latih] 1 epoch = {n_it} iterasi (batch {bs}) -> {t_epoch:.1f} s | VRAM puncak {peak:.2f} GiB")
    print("        (epoch pertama sudah termasuk 'pemanasan' GPU; angka sebenarnya bisa sedikit lebih kecil)")

    # --- waktu imputasi: satu batch jendela, nsample kecil
    T = len(sp.test)
    ebs = ev["batch_size"]
    sampler = make_sampler(model, args.device, ebs)
    starts = window_starts(T, L, ev["stride"])
    idx = starts[:ebs, None] + np.arange(L)[None, :]
    w_obs = sp.test[idx].astype(np.float32)
    w_cond = np.ones_like(w_obs); w_cond[:, L // 3: L // 3 + 24, :] = 0.0
    w_obs = w_obs * w_cond
    w_tp = np.tile(np.arange(L, dtype=np.float32), (len(w_obs), 1))
    sampler(w_obs, w_cond, w_tp, 1)  # pemanasan
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.impute_batches):
        sampler(w_obs, w_cond, w_tp, 2)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t_pass = (time.perf_counter() - t0) / (args.impute_batches * 2)       # detik per (batch jendela x 1 sampel)
    print(f"[imputasi] {t_pass:.2f} s per (batch {len(w_obs)} jendela x 1 sampel, 50 langkah diffusion)")

    # --- proyeksi
    n_cond = len(base["conditions"]); n_mask_seeds = len(base["seeds"])
    print("\n[proyeksi] jam GPU untuk SATU model, semua kondisi x seed mask")
    print(f"           ({n_cond} kondisi x {n_mask_seeds} seed mask = {n_cond * n_mask_seeds} evaluasi)")
    print(f"{'stride':>7} {'nsample':>8} {'jendela':>8} {'jam/model':>10}")
    for stride in (24, 48, 84):
        W = len(window_starts(T, L, stride))
        for ns in (10, 20, 50):
            hrs = math.ceil(W / ebs) * ns * t_pass * n_cond * n_mask_seeds / 3600
            print(f"{stride:>7} {ns:>8} {W:>8} {hrs:>10.1f}")
    print("\nKalikan dengan jumlah seed model (3) untuk total. Latih: "
          f"{t_epoch:.0f} s/epoch x epochs x 3 model.")


if __name__ == "__main__":
    main()
