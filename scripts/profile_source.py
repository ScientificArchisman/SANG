#!/usr/bin/env python3
"""Why does preparing the source photo take ~3.7 s per clip (scripts/latency.py, job 180420)? Times each part of
MotionCodec.crop_image / source_motion on the first frame of a few HDTF clips, and prints which ONNX Runtime
execution providers and how many threads the face detector and the landmark model actually use (a silent
fall-back from CUDA to the CPU, or 128 CPU threads on an 8-CPU job, would explain it).

    sbatch --cpus-per-task=8 bash_scripts/job.sh scripts/profile_source.py --data /beegfs/work/$USER/HDTF
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--list", default=str(REPO / "configs/hdtf_test.csv"))
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    import onnxruntime as ort
    from sang.motion import MotionCodec, decode_clip
    import importlib.util
    spec = importlib.util.spec_from_file_location("ev", REPO / "scripts" / "eval_hdtf.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    print(f"onnxruntime {ort.__version__}, available providers {ort.get_available_providers()}, device {ort.get_device()}")
    print(f"CPUs: os.cpu_count {os.cpu_count()}, SLURM_CPUS_PER_TASK {os.environ.get('SLURM_CPUS_PER_TASK')}, "
          f"OMP_NUM_THREADS {os.environ.get('OMP_NUM_THREADS')}, torch threads {torch.get_num_threads()}", flush=True)
    t = time.perf_counter()
    codec = MotionCodec()
    print(f"MotionCodec loaded in {time.perf_counter() - t:.1f} s")
    fa, lm = codec.cropper.face_analysis_wrapper, codec.cropper.human_landmark_runner
    for name, model in getattr(fa, "models", {}).items():
        s = getattr(model, "session", None)
        print(f"  insightface {name}: providers {s.get_providers() if s else '?'}")
    s = getattr(lm, "session", None)
    print(f"  landmark.onnx: providers {s.get_providers() if s else '?'}", flush=True)

    clips = ev.read_list(args.list, Path(args.data), Path(args.data) / "crops512")[: args.n]
    rows = []
    for c in clips:
        src = np.ascontiguousarray(decode_clip(str(c), 1)[0])
        r = {}
        def tick(name, fn):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            r[name] = time.perf_counter() - t0
            return out
        faces = tick("detect (insightface)", lambda: fa.get(np.ascontiguousarray(src[..., ::-1]), flag_do_landmark_2d_106=True,
                                                            direction="large-small"))
        lmk = tick("landmarks (landmark.onnx)", lambda: lm.run(src, faces[0].landmark_2d_106))
        crop = tick("crop", lambda: codec.crop_with_box(src, codec._box(lmk)))
        tick("source motion (GPU)", lambda: codec.motion_from_crops(crop[0][None]))
        full = {"img_crop_256x256": crop[0], "M_c2o": crop[1], "lmk": lmk}
        tick("source_motion incl. lip normalisation", lambda: codec.source_motion(src, crop=full))
        tick("crop_image (detect + landmarks + crop)", lambda: codec.crop_image(src))
        rows.append(r)
        print(f"{c.stem:<28}" + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
    print("\nMEDIAN seconds (first clip included):")
    for k in rows[0]:
        print(f"  {k:<42}{np.median([r[k] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
