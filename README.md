# Imputasi deret waktu: baseline vs diffusion (CSDI) — Tahap 1

Baseline untuk studi "kapan imputer diffusion mengalahkan baseline sederhana" pada ETTh1.

## Struktur folder

```
imputasi-ts/
├── configs/baseline.yaml        # kondisi missing, metode, seed, hyperparameter
├── data/raw/                    # ETTh1.csv (diunduh, tidak di-commit)
├── data/processed/
├── results/                     # CSV hasil (mentah + ringkasan)
├── scripts/
│   ├── download_data.py         # unduh ETTh1
│   ├── tune_kalman.py           # pilih harmonics Kalman (HANYA data validasi)
│   ├── run_baselines.py         # eksperimen utama baseline
│   ├── setup_csdi.py            # clone CSDI ke third_party/ pada commit terkunci
│   ├── csdi_benchmark.py        # ukur GPU/VRAM/waktu 1 epoch/waktu imputasi + proyeksi
│   ├── train_csdi.py            # latih 1 model CSDI (1 seed)
│   ├── run_csdi.py              # evaluasi CSDI pada mask yang sama dengan baseline
│   └── compare_results.py       # tabel + selisih berpasangan CSDI vs baseline
├── src/imputasi/
│   ├── data.py                  # load, split kronologis, standardisasi (stat dari train)
│   ├── masks.py                 # mask acak & blok (independen / serentak)
│   ├── baselines.py             # linear, ffill, seasonal_interp, STL, Kalman
│   ├── metrics.py               # MAE, RMSE, CRPS, cakupan & lebar interval
│   └── evaluate.py              # satu eksperimen -> satu baris hasil
├── third_party/CSDI/            # repo CSDI (MIT), di-clone oleh setup_csdi.py
├── tests/                       # pytest (mask, kebocoran data, perilaku baseline)
├── requirements.txt             # untuk baseline
├── requirements-csdi.txt        # nanti, minggu 11 (PyTorch dipasang terpisah)
└── environment.yml              # alternatif conda
```

## Setup

Python 3.11 disarankan (3.10–3.12 aman).

```bash
# opsi A: venv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# opsi B: conda
conda env create -f environment.yml && conda activate imputasi-ts
```

Cek: `python -m pytest -q` (semua tes harus lulus, termasuk STL dan Kalman).

## Menjalankan (urutan)

```bash
python scripts/download_data.py                        # sekali
python scripts/tune_kalman.py                          # 1) pilih harmonics di VALIDASI (lambat)
#   -> salin hasil ke configs/baseline.yaml (bagian kalman)
python scripts/run_baselines.py --out results/v2       # 2) run penuh (5 seed, 8 kondisi)
python scripts/run_baselines.py --methods linear ffill seasonal_interp --out results/cepat
python scripts/run_baselines.py --synthetic            # uji kode tanpa data asli
```

Keluaran di folder `--out`: `baselines_raw_etth1.csv`, `baselines_summary_etth1.csv`,
`baselines_per_variable_etth1.csv`, dan `cache/` (parameter Kalman hasil fit, dipakai ulang
selama data, period, harmonics, dan maxiter sama). Gunakan `--out` berbeda untuk setiap versi
agar hasil lama tidak tertimpa.

## Keputusan desain (perlu kamu setujui)

- Split 70/10/20 kronologis; standardisasi dari train. Ini mengikuti dokumen rencana, bukan split ETT 12/4/4 bulan.
- Evaluasi pada split test sebagai satu deret kontinu (baseline tidak butuh jendela). CSDI nanti memakai jendela; pastikan mask dan titik evaluasi yang sama (gunakan `masks.make_mask` dengan seed yang sama) agar perbandingan adil.
- Kondisi blok: total 20% titik hilang (`block_missing_ratio`), blok tak tumpang-tindih. Dengan 20% dan blok 72 jam, hanya ~10 blok per variabel pada test, jadi variansi antar seed besar.
- Interval 95%: Kalman memakai prediktif Gaussian bawaan. Metode titik memakai kuantil |residual| pada validasi (mask seed berbeda), sehingga cakupannya dekat 95% secara konstruksi; yang diperbandingkan adalah LEBAR interval. CRPS metode titik = MAE.
- Kalman: parameter di-fit sekali pada train (lengkap), lalu smoothing pada test. Waktu fit dicatat di `fit_time_s`.
- Seasonal interp: rata-rata nilai jam yang sama dari hari terobservasi terdekat sebelum dan sesudah (maks 14 hari), fallback linear.
- Tidak ada tuning pada test. Semua hyperparameter ada di `configs/baseline.yaml`.
- Pemilihan harmonics Kalman memakai MAE validasi (random30, block24, block72; seed 20000+), bukan test.
  Jika nilai terbesar yang dicoba menang, perluas grid (`--harmonics 3 6 12`).
- `fit` Kalman memperlihatkan jumlah variabel yang konvergen; jika <7/7, naikkan `maxiter`.

## Tahap CSDI (minggu 11)

CSDI: https://github.com/ermongroup/CSDI, lisensi MIT, commit terkunci `7f24a436f08d98853a6b43d4f7f04e5a65ecdf27`.

```bash
# 1) PyTorch CUDA: perintah dari https://pytorch.org/get-started/locally/ (di venv yang sama), lalu:
pip install -r requirements-csdi.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # harus True
python scripts/setup_csdi.py                       # clone + kunci commit + cek lisensi
# 2) ukur dulu, baru putuskan epochs / nsample / jumlah seed
python scripts/csdi_benchmark.py
# 3) latih lalu evaluasi
python scripts/train_csdi.py --seed 0 --epochs 1       # uji alur (beberapa menit)
python scripts/train_csdi.py --seed 0
python scripts/run_csdi.py --model-dir results/csdi/model_seed0 --conditions block24 --seeds 0 --nsample 20
python scripts/run_csdi.py --model-dir results/csdi/model_seed0
python scripts/compare_results.py --baseline results/v2/baselines_raw_etth1.csv --csdi "results/csdi_eval/csdi_raw_etth1_model*.csv"
```

### Keputusan desain adaptor (BUKAN dari paper; wajib dicatat di paper)

- **Hyperparameter model/diffusion/optimizer** = `config/base.yaml` milik CSDI (imputasi). Tidak diubah.
  `epochs` boleh diturunkan setelah benchmark; catat angkanya.
- **Strategi mask latihan**: default `random` (default CSDI). Strategi `hist` CSDI tidak berguna di sini karena
  ETTh1 lengkap (pola histori = semua terobservasi). Konsekuensi: CSDI TIDAK pernah melihat blok saat latihan,
  sedangkan baseline tidak dilatih sama sekali. `mix_block` (ablasi) melatih pada blok 24/72 jam; itu keuntungan
  yang harus dilaporkan terpisah, jangan dicampur dengan hasil utama.
- **Timepoints**: default `relative` (0..167), model tidak tahu jam-dalam-hari. Baseline musiman justru memakai
  period=24. `hour_of_week` memberi fase eksplisit (ablasi yang adil terhadap baseline musiman).
- **Jendela**: 168 jam (> blok 72 jam). Latihan: awal jendela acak (stride 1), `windows_per_epoch` per epoch.
  Evaluasi: stride 24, tiap titik waktu diambil dari jendela yang paling sentral (konteks terbanyak di kedua sisi).
- **Metrik CSDI**: MAE dari median sampel (protokol CSDI), RMSE dari rata-rata sampel, CRPS estimator "fair" dari
  sampel (bukan aproksimasi kuantil CSDI), interval dari kuantil sampel 2.5%/97.5%. Semua pada z-score yang sama.
- `utils.train` CSDI hanya menyimpan model AKHIR, bukan yang terbaik di validasi.
- Mask evaluasi memakai `masks.make_mask` dengan seed yang sama dengan baseline, jadi selisih berpasangan valid.
- Kode PyTorch (`csdi_torch.py`, skrip latih/eval/benchmark) belum pernah dijalankan oleh penulis; bagian numpy
  (windowing, stitching, CRPS sampel, evaluasi) teruji lewat `pytest`.

### Baseline regresi non-diffusion (memisahkan "multivariat" dari "diffusion")

`ridge_uv` (hanya deret sendiri), `ridge_mv` (+ variabel lain pada waktu t), `ridge_mv_lag` (+ variabel lain pada t -/+ 24 jam).
Ridge pada anomali terhadap referensi musiman (tahan pergeseran level train->test), satu model per (variabel, pola observasi).

```bash
python scripts/run_baselines.py --methods ridge_uv ridge_mv ridge_mv_lag --out results/v2_mv
python scripts/compare_results.py --baseline results/v2/baselines_raw_etth1.csv results/v2_mv/baselines_raw_etth1.csv \
       --csdi "results/csdi_eval/csdi_raw_etth1_model*.csv"
```

Cara membaca: `ridge_uv` vs `ridge_mv` mengisolasi efek variabel lain; `ridge_mv` vs CSDI mengisolasi efek model
nonlinear/diffusion. Catatan keadilan: ridge dilatih pada masking dengan JENIS kondisi yang sama dengan evaluasi
(seed berbeda), sedangkan CSDI default hanya melihat mask acak. Bandingkan juga dengan CSDI `mix_block`.

```bash
python scripts/train_csdi.py --seed 1 --config configs/csdi_ett_det_l1_mixblock.yaml --out results/csdi_det_l1_mixblock
python scripts/run_csdi.py --model-dir results/csdi_det_l1_mixblock/model_seed1 --config configs/csdi_ett_det_l1_mixblock.yaml --conditions random10 block24 block72 block24_sim block72_sim --seeds 0 1 2 --nsample 2 --out results/csdi_eval_det_l1_mixblock

#evaluasi menggunakan seed mask 3 4
python scripts/run_csdi.py --model-dir results/csdi_det_l1_mixblock/model_seed1 --config configs/csdi_ett_det_l1_mixblock.yaml --conditions random10 block24 block72 block24_sim block72_sim --seeds 3 4 --nsample 2 --out results/csdi_eval_det_l1_mixblock

python scripts/run_csdi.py --model-dir results/csdi_det_l1_mixblock/model_seed2 --config configs/csdi_ett_det_l1_mixblock.yaml --conditions random10 block24 block72 block24_sim block72_sim --seeds 3 4 --nsample 2 --out results/csdi_eval_det_l1_mixblock

#train seed 3 dan 4
python scripts/train_csdi.py --seed 3 --config configs/csdi_ett_mixblock.yaml --out results/csdi_mixblock

#evaluasi seed 3 dan 4
python scripts/run_csdi.py --model-dir results/csdi_mixblock/model_seed3 --config configs/csdi_ett_mixblock.yaml --conditions block24_sim block72_sim --seeds 0 1 2 --nsample 50 --out results/csdi_eval_mixblock
```

train seed 1 kemudian jalankan skrip untuk uji coba pada lag-1. ulangi langkah train untuk seed 2 dan jalankan sekali lagi skrip uji coba dengan mengubah nomor seed.
