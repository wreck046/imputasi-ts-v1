"""Adaptor PyTorch untuk repo CSDI (https://github.com/ermongroup/CSDI, MIT).

Tidak mengubah arsitektur maupun hyperparameter diffusion. Yang ditambahkan hanya:
  - subclass CSDI_base untuk data ETT (process_data sama dengan CSDI_Physio),
  - dataset jendela, dan
  - sampler yang menyambungkan model.evaluate ke imputasi/csdi_eval.py.
Modul ini butuh `torch` dan repo CSDI di `csdi_path` (lihat scripts/setup_csdi.py).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .masks import block_mask, random_mask
from .windows import window_starts, window_timepoints


def ensure_csdi_on_path(csdi_path) -> str:
    p = Path(csdi_path).resolve()
    if not (p / "main_model.py").exists():
        raise FileNotFoundError(f"Repo CSDI tidak ditemukan di {p}. Jalankan: python scripts/setup_csdi.py")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return str(p)


def build_model(config: dict, device, target_dim: int, csdi_path, block_lengths=(24, 72)):
    """Bangun CSDI untuk imputasi deret waktu multivariat dengan `target_dim` variabel."""
    ensure_csdi_on_path(csdi_path)
    from main_model import CSDI_base  # dari repo CSDI

    deterministic = bool(config["model"].get("deterministic", False))
    det_loss = str(config["model"].get("deterministic_loss", "l2")).lower()   # "l2" (MSE) atau "l1" (MAE)
    if det_loss not in ("l1", "l2"):
        raise ValueError("model.deterministic_loss harus 'l1' atau 'l2'")

    class CSDI_ETT(CSDI_base):
        def __init__(self, cfg, dev, k, blens, det, dloss="l2"):
            super().__init__(k, cfg, dev)
            self.block_lengths = tuple(blens)
            self.deterministic = det
            self.det_loss = dloss

        # ---- KONTROL DETERMINISTIK (model.deterministic: true) -----------------------------
        # Jaringan SAMA (diff_CSDI, jumlah parameter sama), tetapi tanpa diffusion: satu forward
        # pass, kanal 'target ber-noise' = 0, langkah diffusion tetap 0, dan loss = MSE langsung
        # antara keluaran dan nilai asli di titik target. Mask latihan & jendela sama dgn CSDI.
        def _det_forward(self, observed_data, cond_mask, side_info):
            t = torch.zeros(len(observed_data), dtype=torch.long, device=observed_data.device)
            x = torch.stack([cond_mask * observed_data, torch.zeros_like(observed_data)], dim=1)
            return self.diffmodel(x, side_info, t)                      # (B,K,L)

        def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
            if not self.deterministic:
                return super().calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train, set_t)
            pred = self._det_forward(observed_data, cond_mask, side_info)
            target_mask = observed_mask - cond_mask
            n = target_mask.sum()
            err = (pred - observed_data) * target_mask
            err = err.abs() if self.det_loss == "l1" else err ** 2
            return err.sum() / (n if n > 0 else 1)

        def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train):
            if not self.deterministic:
                return super().calc_loss_valid(observed_data, cond_mask, observed_mask, side_info, is_train)
            return self.calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train)

        def impute(self, observed_data, cond_mask, side_info, n_samples):
            if not self.deterministic:
                return super().impute(observed_data, cond_mask, side_info, n_samples)
            pred = self._det_forward(observed_data, cond_mask, side_info).detach()
            return pred.unsqueeze(1).repeat(1, n_samples, 1, 1)         # sampel identik (B,S,K,L)

        def process_data(self, batch):  # identik dengan CSDI_Physio
            observed_data = batch["observed_data"].to(self.device).float()
            observed_mask = batch["observed_mask"].to(self.device).float()
            observed_tp = batch["timepoints"].to(self.device).float()
            gt_mask = batch["gt_mask"].to(self.device).float()
            observed_data = observed_data.permute(0, 2, 1)       # (B,K,L)
            observed_mask = observed_mask.permute(0, 2, 1)
            gt_mask = gt_mask.permute(0, 2, 1)
            cut_length = torch.zeros(len(observed_data)).long().to(self.device)
            return observed_data, observed_mask, observed_tp, gt_mask, observed_mask, cut_length

        def get_hist_mask(self, observed_mask, for_pattern_mask=None):
            """Strategi 'mix_block' (ABLASI): 50% mask acak CSDI, 50% blok seperti kondisi uji.
            Catatan: ini melatih CSDI pada pola mirip test, jadi harus dilaporkan sebagai keuntungan."""
            B, K, L = observed_mask.shape
            rand_cond = self.get_randmask(observed_mask)
            cond = rand_cond.clone()
            for i in range(B):
                if np.random.rand() < 0.5:
                    continue
                length = int(np.random.choice(self.block_lengths))
                length = min(length, L)
                ratio = float(np.random.uniform(0.1, 0.4))
                miss = block_mask((L, K), length, ratio, int(np.random.randint(0, 2**31 - 1)),
                                  simultaneous=bool(np.random.rand() < 0.5))      # (L,K) True=hilang
                cond[i] = torch.from_numpy(~miss.T).float().to(self.device)
            return cond

    cfg = copy.deepcopy({k: config[k] for k in ("train", "diffusion", "model")})  # CSDI_base memodifikasi cfg
    return CSDI_ETT(cfg, device, target_dim, block_lengths, deterministic, det_loss).to(device)


# ----------------------------------------------------------------------------- dataset
def _item(x: np.ndarray, gt_mask: np.ndarray, tp: np.ndarray) -> dict:
    return {
        "observed_data": torch.from_numpy(x.astype(np.float32)),
        "observed_mask": torch.ones(x.shape, dtype=torch.float32),   # data lengkap
        "timepoints": torch.from_numpy(tp.astype(np.float32)),
        "gt_mask": torch.from_numpy(gt_mask.astype(np.float32)),
    }


class TrainWindows(Dataset):
    """Jendela acak (awal seragam, stride 1) dari split train. Panjang dataset = jendela per epoch."""

    def __init__(self, series: np.ndarray, window_len: int, n_items: int, tp_mode: str, seed: int):
        self.x, self.L, self.n, self.mode = series, window_len, n_items, tp_mode
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = int(self.rng.integers(0, len(self.x) - self.L + 1))
        tp = window_timepoints(np.array([s]), self.L, 0, self.mode)[0]
        return _item(self.x[s:s + self.L], np.ones((self.L, self.x.shape[1])), tp)


class ValWindows(Dataset):
    """Jendela validasi dengan mask acak TETAP (gt_mask = bagian terobservasi) untuk loss validasi."""

    def __init__(self, series: np.ndarray, window_len: int, stride: int, ratio: float,
                 tp_mode: str, tp_offset: int, seed: int = 12345):
        self.x, self.L, self.mode, self.off = series, window_len, tp_mode, tp_offset
        self.starts = window_starts(len(series), window_len, stride)
        self.keep = ~random_mask(series.shape, ratio, seed)           # True = terobservasi

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        tp = window_timepoints(np.array([s]), self.L, self.off, self.mode)[0]
        return _item(self.x[s:s + self.L], self.keep[s:s + self.L], tp)


# ----------------------------------------------------------------------------- sampler
def make_sampler(model, device, batch_size: int = 16):
    """Bungkus model.evaluate menjadi sampler untuk imputasi.csdi_eval.evaluate_condition."""

    @torch.no_grad()
    def sampler(obs: np.ndarray, cond: np.ndarray, tp: np.ndarray, nsample: int) -> np.ndarray:
        model.eval()
        out = []
        for i in range(0, len(obs), batch_size):
            o = torch.from_numpy(obs[i:i + batch_size])
            batch = {
                "observed_data": o,
                "observed_mask": torch.ones_like(o),
                "timepoints": torch.from_numpy(tp[i:i + batch_size]),
                "gt_mask": torch.from_numpy(cond[i:i + batch_size]),
            }
            samples, *_ = model.evaluate(batch, nsample)              # (B, S, K, L)
            out.append(samples.permute(0, 1, 3, 2).float().cpu().numpy())
        return np.concatenate(out, axis=0)                            # (W, S, L, K)

    return sampler
