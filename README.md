# DDECC reproduction

Reproduction of *Denoising Diffusion Error Correction Codes* (Choukroun & Wolf, ICLR 2023)
with the code of https://github.com/yoniLc/DDECC, for 14 codes at Eb/N0 = 4, 5, 6 dB.

- Model: N = 6, d = 128, h = 8; training: 2000 epochs × 1000 batches, batch 128, lr 1e-4 (cosine to 5e-6), β = 0.01
- Test: per Eb/N0, until > 500 frame errors and > 1e5 frames, or 1e7 frames;
  regular and line-search reverse diffusion; metric −ln(BER), higher is better

### Regular reverse diffusion: −ln(BER)

| Code | 4 dB | 5 dB | 6 dB |
|---|---|---|---|
| BCH(63,36) | 4.71 | 6.46 | 8.94 |
| BCH(63,45) | 5.69 | 7.83 | 10.65 |
| BCH(63,51) | 5.25 | 7.09 | 9.24 |
| POLAR(64,32) | 7.04 | 9.50 | 12.16 |
| POLAR(64,48) | 5.86 | 7.87 | 10.33 |
| POLAR(128,64) | 8.81 | 12.85 | 15.85 |
| POLAR(128,86) | 7.44 | 10.63 | 14.56 |
| POLAR(128,96) | 7.18 | 10.01 | 13.19 |
| LDPC(49,24) | 5.99 | 8.46 | 11.77 |
| LDPC(121,60) | 5.54 | 8.97 | 14.21 |
| LDPC(121,70) | 6.88 | 10.90 | 16.67 |
| LDPC(121,80) | 7.86 | 12.52 | 16.96 |
| MACKAY(96,48) | 8.21 | 11.85 | 15.46 |
| CCSDS(128,64) | 7.88 | 12.42 | 17.41 |

### Line-search reverse diffusion: −ln(BER)

| Code | 4 dB | 5 dB | 6 dB |
|---|---|---|---|
| BCH(63,36) | 4.70 | 6.39 | 8.86 |
| BCH(63,45) | 5.60 | 8.04 | 11.24 |
| BCH(63,51) | 5.21 | 7.29 | 10.34 |
| POLAR(64,32) | 6.90 | 9.55 | 12.78 |
| POLAR(64,48) | 5.84 | 8.01 | 10.71 |
| POLAR(128,64) | 8.62 | 12.55 | 15.76 |
| POLAR(128,86) | 7.19 | 10.28 | 14.15 |
| POLAR(128,96) | 7.00 | 10.12 | 13.24 |
| LDPC(49,24) | 5.84 | 8.31 | 11.65 |
| LDPC(121,60) | 5.31 | 8.56 | 13.88 |
| LDPC(121,70) | 6.68 | 10.70 | 17.62 |
| LDPC(121,80) | 7.63 | 12.14 | 18.97 |
| MACKAY(96,48) | 7.86 | 11.51 | 15.37 |
| CCSDS(128,64) | 7.66 | 12.05 | 17.44 |

Some test points stopped at about 100 frame errors (Main.py's default) instead of the paper's 500; `figures/results.csv` lists the frames and frame errors of every point. CCSDS(128,64) regular 6 dB: 9.9e6 of 1e7 frames.

## Comparison with the DDECC paper (N = 6, d = 128)

![regular](figures/vs_paper_reg.png)

![line search](figures/vs_paper_LS.png)

## Contents

| folder | content |
|---|---|
| `training_logs/` | `logging.txt` (and small status files) of every training run |
| `runner_logs/` | console output of the training runner |
| `test_results/` | resumable test logs and states (`eval_official`, `eval_resume`, `eval_v3`) |
| `figures/` | `results.csv`, BER / −ln(BER) figures, comparison with the paper |
| `DDECC_reproduction.ipynb` | Colab notebook: clone → environment → check → train → test → figures → comparison with the paper |
| `scripts/ddecc_all_in_one.py` | one Colab cell (no GPU): check + tables + figures + comparison + this package |

Trained models (`best_model`) are not included (size).
