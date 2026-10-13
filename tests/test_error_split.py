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
    snd = es.pool_sounds([es.frame_errors(g, y, sounds)])
    assert snd["closure p/b/m"]["error_pct"] == 100 and snd["closure p/b/m"]["signed"] > 0.49
    assert snd["open vowel"]["ratio"] == 0
    s = es.summarise([dict(es.split_clip(g, y), rate=10.0), {}])
    assert s["clips"] == 1 and abs(sum(s["parts_pct"].values()) - 100) < 1e-9
