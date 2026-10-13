"""C0 error split (sang/error_split.py). CPU only, synthetic traces."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sang import error_split as es


def wave(n=250, seed=0):
    rng = np.random.default_rng(seed)
    return np.convolve(rng.standard_normal(n + 20), np.ones(5) / 5, mode="same")[10:-10]


def test_parts_add_up_to_the_error():
    y = wave()
    g = 0.7 * np.roll(y, 2) + 0.3 * wave(seed=1) + 0.2
    rec = es.split_clip(g, y, max_lag=4)
    assert abs(sum(rec["parts"].values()) - rec["mse_rel"]) < 1e-9
    assert rec["lag"] == 2 and rec["r_best"] > rec["r"]                  # generated lips move 2 frames later


def test_pure_delay_is_timing_and_pure_scaling_is_amplitude():
    y = wave()
    late = es.split_clip(np.roll(y, -3), y, max_lag=4)
    assert late["lag"] == -3 and late["parts"]["timing"] > 5 * late["parts"]["shape"]
    flat = es.split_clip(0.5 * y + y.mean() * 0.5, y, max_lag=4)
    assert abs(flat["r"] - 1) < 1e-9 and abs(flat["std_r"] - 0.5) < 1e-9
    assert flat["parts"]["timing"] < 1e-9 and flat["parts"]["shape"] < 1e-9 and flat["parts"]["amplitude"] > 0.2
    assert flat["speed_r_fast"] < 0.51                                     # half the movement, half the speed
    assert abs(flat["speed_q90_r"] - 0.5) < 1e-9 and abs(flat["speed_q50_r"] - 0.5) < 1e-9


def test_frames_take_the_nearest_phoneme_and_pauses_win():
    tokens = {0: "<pad>", 1: "p", 2: "a", 3: "|"}
    post = np.zeros((20, 4))
    post[:, 0] = 1
    post[2:4] = [0, 1, 0, 0]                                               # 'p' at ticks 2-3 -> frame 1
    post[12:14] = [0, 0, 1, 0]                                             # 'a' at ticks 12-13 -> frame 6
    post[16] = [0, 0, 0, 1]                                                # word boundary: not a phone
    spikes = es.phone_spikes(post, tokens, n=10)
    assert spikes == [(1, "p"), (6, "a")]
    pause = np.zeros(10, bool)
    pause[9] = True
    lab = es.frame_sounds(spikes, pause, 10, reach=2)
    assert list(lab) == ["closure p/b/m"] * 4 + ["open vowel"] * 5 + ["pause"]
    assert es.speech_rate(spikes, pause, 10) == 2 / (9 / 25)


def test_sound_classes():
    assert es.sound_of("b") == "closure p/b/m" and es.sound_of("v") == "labiodental f/v"
    assert es.sound_of("aɪ") == "open vowel" and es.sound_of("oʊ") == "rounded vowel/w" and es.sound_of("w") == "rounded vowel/w"
    assert es.sound_of("ə") == "other vowel" and es.sound_of("j") == "other consonant" and es.sound_of("tʃ") == "other consonant"


def test_pooling_shows_where_the_error_is():
    y = wave()
    g = y.copy()
    sounds = np.array(["open vowel"] * 125 + ["closure p/b/m"] * 125, dtype=object)
    g[125:] += 0.5 * y.std()                                               # lips stay open at closures
    snd = es.pool_sounds([es.frame_errors(g, y, sounds)], scale=y.std())
    assert snd["closure p/b/m"]["error_pct"] == 100 and snd["closure p/b/m"]["signed"] > 0.49
    assert snd["open vowel"]["ratio"] == 0
    s = es.summarise([dict(es.split_clip(g, y), rate=10.0), {}])
    assert s["clips"] == 1 and abs(sum(s["parts_pct"].values()) - 100) < 1e-9


def test_pooled_split_is_not_dominated_by_a_clip_whose_lips_barely_move():
    y = wave()
    good = es.split_clip(0.6 * y + 0.6 * wave(seed=2), y)                  # normal clip, mostly shape error
    still = 0.01 * wave(seed=3)                                            # real lips barely move
    bad = es.split_clip(still + 0.05, still)                               # generated sits 5 real stds too open
    s = es.summarise([dict(good, rate=10.0), dict(bad, rate=10.0)])
    assert s["parts_pct"]["bias"] > 90                                     # per-clip mean: the still clip wins
    assert s["pooled_pct"]["bias"] < 50 and s["pooled_pct"]["shape"] > 40  # pooled: in lip-opening units
    assert abs(s["gap"]["total"] - s["gap"]["timing"] - s["gap"]["shape"]) < 1e-12


def test_speed_quantiles_are_not_fooled_by_frame_selection():
    y = wave()
    rng = np.random.default_rng(5)
    g = y + 0.3 * y.std() * rng.standard_normal(len(y))                    # unbiased, just noisy
    rec = es.split_clip(g, y)
    assert rec["speed_r_fast"] < 1                                          # selected on the real trace: looks sluggish
    assert rec["speed_q90_r"] > 1                                           # quantile by quantile: it is not
