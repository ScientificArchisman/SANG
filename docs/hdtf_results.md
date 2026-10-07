# HDTF results: SANG-M and published methods

Published rows are copied from the named table (sang/sota_hdtf.py) and only compare within the same source:
test splits, crops, resolutions and SyncNet/FID code differ between papers (FLOAT's HDTF FID is 21.10 in its
own Tab. 1 and 9.164 in IMTalker's Tab. 2). LSE-C = Sync-C (higher is better), LSE-D = Sync-D (lower is better).
Params: trainable generator; +LP = also runs the frozen LivePortrait renderer (~130 M); n/r = not reported.
E-FID needs a 3DMM fitter and is not computed by our script.

Our rows are filled by: `python scripts/eval_hdtf.py --table results/hdtf/*/summary.json --write-doc docs/hdtf_results.md`

| Source | Method | Family | Params | FID ↓ | FVD ↓ | CSIM ↑ | LSE-C ↑ | LSE-D ↓ | PSNR ↑ | SSIM ↑ | E-FID ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours (pending) | SANG-M — run scripts/eval_hdtf.py | flow-matching DiT, 42-d | 53.4 M +LP | – | – | – | – | – | – | – | – |
| FLOAT Tab. 1 | SadTalker | 3DMM + warping | n/r | 71.95 | 339.06 | 0.644 | 7.305 | 7.947 | – | – | 1.914 |
| FLOAT Tab. 1 | EDTalk | motion bases | n/r | 50.08 | 211.28 | 0.626 | 7.623 | 8.123 | – | – | 1.579 |
| FLOAT Tab. 1 | AniTalker | motion | n/r | 39.51 | 184.45 | 0.643 | 7.288 | 7.907 | – | – | 1.83 |
| FLOAT Tab. 1 | Hallo | pixel (SD-1.5) | ~1.7 B est. | 25.36 | 197.20 | 0.869 | 7.582 | 7.792 | – | – | 1.039 |
| FLOAT Tab. 1 | EchoMimic | pixel (SD-1.5) | ~1.7 B est. | 33.55 | 296.76 | 0.823 | 6.242 | 8.903 | – | – | 1.234 |
| FLOAT Tab. 1 | FLOAT | motion latent | n/r | 21.10 | 162.05 | 0.843 | 8.222 | 7.29 | – | – | 1.229 |
| KDTalker Tab. 1 | Real video | - | n/r | – | – | – | 8.243 | 6.929 | – | – | – |
| KDTalker Tab. 1 | KDTalker | LivePortrait keypoints | 42.9 M +LP | 9.756 | – | 0.949 | 7.326 | 7.548 | – | – | – |
| Teller Tab. 1 | Real video | - | n/r | – | – | – | 8.094 | 6.976 | – | – | – |
| Teller Tab. 1 | Teller | motion tokens (AR) | n/r | 21.352 | 173.463 | – | 7.696 | 7.536 | – | – | – |
| IMTalker Tab. 2 | FLOAT | motion latent | n/r | 9.164 | 198.964 | 0.843 | – | – | – | – | – |
| IMTalker Tab. 2 | Ditto | LivePortrait motion | n/r +LP | 11.746 | – | 0.886 | – | – | – | – | – |
| IMTalker Tab. 2 | IMTalker | implicit motion | 39 M + 124 M | 9.084 | 143.623 | 0.869 | 7.711 | – | – | – | – |
| SoulX Tab. 3 | EchoMimic | pixel (SD-1.5) | ~1.7 B est. | 9.00 | 155.71 | – | 3.56 | 10.22 | – | – | – |
| SoulX Tab. 3 | SoulX-FlashHead Pro | pixel (Wan2.1) | 1.3 B | 9.97 | 111.38 | – | 5.73 | 8.77 | – | – | – |
| SoulX Tab. 3 | SoulX-FlashHead Lite | pixel (Wan2.1) | 1.3 B | 11.37 | 126.52 | – | 4.21 | 9.49 | – | – | – |
| SoulX Tab. 3 | Ditto | LivePortrait motion | n/r +LP | 12.35 | 199.13 | – | 3.57 | 10.49 | – | – | – |
| SoulX Tab. 3 | Sonic | pixel | n/r | 13.53 | 113.31 | – | 5.17 | 8.69 | – | – | – |
| SoulX Tab. 3 | Hallo3 | pixel | n/r | 15.95 | 160.94 | – | 3.18 | 10.72 | – | – | – |
| SoulX Tab. 3 | AniPortrait | pixel | n/r | 19.83 | 242.29 | – | 1.89 | 11.91 | – | – | – |
| SoulX Tab. 3 | SadTalker | 3DMM + warping | n/r | 21.58 | 207.67 | – | 4.60 | 9.21 | – | – | – |
| Playmate2 | Playmate2 (after DPO) | pixel | n/r | – | – | – | 8.15 | 7.32 | – | – | – |
