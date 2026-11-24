"""Unduh ETTh1 dari repo resmi ETDataset ke data/raw/. Jalankan di mesin dengan internet."""
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
dest = Path(__file__).resolve().parents[1] / "data" / "raw" / "ETTh1.csv"
dest.parent.mkdir(parents=True, exist_ok=True)
if dest.exists():
    print(f"Sudah ada: {dest}")
    sys.exit(0)
urllib.request.urlretrieve(URL, dest)
n = sum(1 for _ in open(dest)) - 1
print(f"Tersimpan: {dest} ({n} baris)")
if n != 17420:
    print(f"PERINGATAN: ETTh1 standar berisi 17420 baris, ditemukan {n}.")
