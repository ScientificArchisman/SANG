"""Published HDTF results for audio-driven one-shot talking heads, copied from each paper's table.

Every number is from the table named in `source`, as checked in research_notes/ and
docs/sota_review_2026.md. **Numbers only compare within one source**: test splits, crops,
resolutions, SyncNet checkpoints and FID code differ between papers (FLOAT's HDTF FID is 21.10 in
its own Tab. 1 and 9.164 in IMTalker's Tab. 2). LSE-C = Sync-C (SyncNet confidence, higher is
better); LSE-D = Sync-D (SyncNet distance, lower is better).

params: trainable generator as the paper states it; "+LP" = also runs the frozen LivePortrait
renderer (~130 M, counted from its released fp32 weights); None = not reported.
"""

# metric keys, in table order, with the direction that is better
METRICS = (("FID", "low"), ("FVD", "low"), ("CSIM", "high"), ("LSE-C", "high"), ("LSE-D", "low"),
           ("PSNR", "high"), ("SSIM", "high"), ("E-FID", "low"))

SOTA = [
    # ---- FLOAT, ICCV 2025, arXiv 2412.01064, Table 1 (HDTF)
    {"source": "FLOAT Tab. 1", "method": "SadTalker", "family": "3DMM + warping", "params": None,
     "m": {"FID": 71.95, "FVD": 339.06, "CSIM": 0.644, "E-FID": 1.914, "LSE-D": 7.947, "LSE-C": 7.305}},
    {"source": "FLOAT Tab. 1", "method": "EDTalk", "family": "motion bases", "params": None,
     "m": {"FID": 50.08, "FVD": 211.28, "CSIM": 0.626, "E-FID": 1.579, "LSE-D": 8.123, "LSE-C": 7.623}},
    {"source": "FLOAT Tab. 1", "method": "AniTalker", "family": "motion", "params": None,
     "m": {"FID": 39.51, "FVD": 184.45, "CSIM": 0.643, "E-FID": 1.830, "LSE-D": 7.907, "LSE-C": 7.288}},
    {"source": "FLOAT Tab. 1", "method": "Hallo", "family": "pixel (SD-1.5)", "params": "~1.7 B est.",
     "m": {"FID": 25.36, "FVD": 197.20, "CSIM": 0.869, "E-FID": 1.039, "LSE-D": 7.792, "LSE-C": 7.582}},
    {"source": "FLOAT Tab. 1", "method": "EchoMimic", "family": "pixel (SD-1.5)", "params": "~1.7 B est.",
     "m": {"FID": 33.55, "FVD": 296.76, "CSIM": 0.823, "E-FID": 1.234, "LSE-D": 8.903, "LSE-C": 6.242}},
    {"source": "FLOAT Tab. 1", "method": "FLOAT", "family": "motion latent", "params": None,
     "m": {"FID": 21.10, "FVD": 162.05, "CSIM": 0.843, "E-FID": 1.229, "LSE-D": 7.290, "LSE-C": 8.222}},
    # ---- KDTalker, IJCV 2025, arXiv 2503.12963, Table 1 (HDTF)
    {"source": "KDTalker Tab. 1", "method": "Real video", "family": "-", "params": None,
     "m": {"LSE-C": 8.243, "LSE-D": 6.929}},
    {"source": "KDTalker Tab. 1", "method": "KDTalker", "family": "LivePortrait keypoints", "params": "42.9 M +LP",
     "m": {"FID": 9.756, "CSIM": 0.949, "LSE-C": 7.326, "LSE-D": 7.548}},
    # ---- Teller, CVPR 2025, arXiv 2503.18429, Table 1 (HDTF)
    {"source": "Teller Tab. 1", "method": "Real video", "family": "-", "params": None,
     "m": {"LSE-C": 8.094, "LSE-D": 6.976}},
    {"source": "Teller Tab. 1", "method": "Teller", "family": "motion tokens (AR)", "params": None,
     "m": {"FID": 21.352, "FVD": 173.463, "LSE-C": 7.696, "LSE-D": 7.536}},
    # ---- IMTalker, arXiv 2511.22167, Table 2 (HDTF, audio-driven)
    {"source": "IMTalker Tab. 2", "method": "FLOAT", "family": "motion latent", "params": None,
     "m": {"FID": 9.164, "FVD": 198.964, "CSIM": 0.843}},
    {"source": "IMTalker Tab. 2", "method": "Ditto", "family": "LivePortrait motion", "params": "n/r +LP",
     "m": {"FID": 11.746, "CSIM": 0.886}},
    {"source": "IMTalker Tab. 2", "method": "IMTalker", "family": "implicit motion", "params": "39 M + 124 M",
     "m": {"FID": 9.084, "FVD": 143.623, "CSIM": 0.869, "LSE-C": 7.711}},
    # ---- SoulX-FlashHead, 2026, Table 3 (HDTF, 75 clips)
    {"source": "SoulX Tab. 3", "method": "EchoMimic", "family": "pixel (SD-1.5)", "params": "~1.7 B est.",
     "m": {"FID": 9.00, "FVD": 155.71, "LSE-C": 3.56, "LSE-D": 10.22}},
    {"source": "SoulX Tab. 3", "method": "SoulX-FlashHead Pro", "family": "pixel (Wan2.1)", "params": "1.3 B",
     "m": {"FID": 9.97, "FVD": 111.38, "LSE-C": 5.73, "LSE-D": 8.77}},
    {"source": "SoulX Tab. 3", "method": "SoulX-FlashHead Lite", "family": "pixel (Wan2.1)", "params": "1.3 B",
     "m": {"FID": 11.37, "FVD": 126.52, "LSE-C": 4.21, "LSE-D": 9.49}},
    {"source": "SoulX Tab. 3", "method": "Ditto", "family": "LivePortrait motion", "params": "n/r +LP",
     "m": {"FID": 12.35, "FVD": 199.13, "LSE-C": 3.57, "LSE-D": 10.49}},
    {"source": "SoulX Tab. 3", "method": "Sonic", "family": "pixel", "params": None,
     "m": {"FID": 13.53, "FVD": 113.31, "LSE-C": 5.17, "LSE-D": 8.69}},
    {"source": "SoulX Tab. 3", "method": "Hallo3", "family": "pixel", "params": None,
     "m": {"FID": 15.95, "FVD": 160.94, "LSE-C": 3.18, "LSE-D": 10.72}},
    {"source": "SoulX Tab. 3", "method": "AniPortrait", "family": "pixel", "params": None,
     "m": {"FID": 19.83, "FVD": 242.29, "LSE-C": 1.89, "LSE-D": 11.91}},
    {"source": "SoulX Tab. 3", "method": "SadTalker", "family": "3DMM + warping", "params": None,
     "m": {"FID": 21.58, "FVD": 207.67, "LSE-C": 4.60, "LSE-D": 9.21}},
    # ---- Playmate2, arXiv 2510.12089 (HDTF), DPO with a lip-sync reward
    {"source": "Playmate2", "method": "Playmate2 (after DPO)", "family": "pixel", "params": None,
     "m": {"LSE-C": 8.15, "LSE-D": 7.32}},
]


def fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, str):
        return v
    t = f"{v:.3f}".rstrip("0")                     # keep the paper's precision: 9.00 -> 9.00, 9.756 -> 9.756
    whole, _, dec = t.partition(".")
    return f"{whole}.{dec.ljust(2, '0')}"


def markdown_table(ours: list[dict] | None = None) -> str:
    """SOTA rows grouped by source table, then our rows (each a dict like the SOTA entries).
    Our rows come first, in their own group, because only they share one protocol."""
    cols = [k for k, _ in METRICS]
    arrows = {k: ("↓" if d == "low" else "↑") for k, d in METRICS}
    head = "| Source | Method | Family | Params | " + " | ".join(f"{c} {arrows[c]}" for c in cols) + " |"
    sep = "|" + "---|" * (4 + len(cols))
    rows = [head, sep]
    for r in (ours or []) + SOTA:
        rows.append(f"| {r['source']} | {r['method']} | {r['family']} | {r['params'] or 'n/r'} | "
                    + " | ".join(fmt(r["m"].get(c)) for c in cols) + " |")
    return "\n".join(rows)
