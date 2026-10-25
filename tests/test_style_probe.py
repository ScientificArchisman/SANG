"""scripts/style_probe.py: the ridge probe on synthetic speakers whose lip level shows in the photo but
not in the audio. CPU only; no model, data or decord."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sang.naturalness import lip_style

spec = importlib.util.spec_from_file_location("style_probe", REPO / "scripts" / "style_probe.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)


def clips(n_speakers: int, per: int, seed: int, shrink: float = 0.0) -> list[dict]:
    """shrink: how much of the speaker's level SANG's sample keeps (0 = none, all pulled to the mean)."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_speakers):
        level = rng.normal(0, 1.0)                                    # the speaker's habitual opening
        for _ in range(per):
            t = np.arange(200)
            talk = np.sin(t / rng.uniform(3, 6)) + 0.3 * rng.normal(size=200)
            real = level + talk
            gen = shrink * level - 0.5 + 0.9 * talk + 0.2 * rng.normal(size=200)   # pulled to the mean, biased
            g, r = lip_style(gen), lip_style(real)
            out.append({"speaker": f"s{s}", "gen": g, "real": r, "gen_lip": gen, "real_lip": real,
                        "self": np.array([g["lo"], g["mid"], g["hi"], gen.std()]),
                        "photo": np.concatenate([[level + 0.1 * rng.normal()], rng.normal(size=5)]),
                        "audio": rng.normal(size=8)})
    return out


def test_probe_finds_the_photo_and_orders_the_rows():
    rows = sp.probe(clips(60, 8, 0), clips(20, 5, 1))           # the level is only in the photo
    assert set(rows) == {"model", "const", "self", "photo", "audio", "all", "oracle"}
    lvl = {k: r["level"] for k, r in rows.items()}
    assert lvl["model"] < lvl["const"] < lvl["photo"]                           # bias, then the speaker
    assert abs(lvl["photo"] - lvl["oracle"]) < 0.02          # the median-matching oracle is not CCC-optimal
    assert lvl["photo"] > lvl["audio"] + 0.02 and lvl["photo"] > lvl["self"] + 0.02 and abs(lvl["all"] - lvl["photo"]) < 0.02
    assert rows["oracle"]["r2_mid"] == 1.0 and rows["photo"]["r2_mid"] > 0.8 and rows["audio"]["r2_mid"] < 0.2


def test_probe_recalibrates_a_shrunk_level_from_the_sample_itself():
    rows = sp.probe(clips(60, 8, 0, shrink=0.6), clips(20, 5, 1, shrink=0.6))
    assert rows["self"]["level"] > rows["const"]["level"] + 0.02 and abs(rows["self"]["level"] - rows["oracle"]["level"]) < 0.02


def test_ridge_and_lambda():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    Y = X @ rng.normal(size=(5, 3)) + 2.0
    pred = sp.ridge_fit(X, Y, 1e-6)(X)
    assert np.allclose(pred, Y, atol=1e-3)
    lam = sp.choose_lambda(X, Y + rng.normal(size=Y.shape) * 0.1, np.arange(400) // 10)
    assert lam <= 0.1                                                  # little noise: little shrinkage
