import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sang.codec import load_mimi
from sang.data import clip_tokens
from sang.video import load_vidtok


def test_clip_tokens_alignment():
    """Real TalkVid clip -> aligned video/audio tokens + ref with expected shapes (skips if no data)."""
    files = sorted(glob.glob("/beegfs/work_fast/shared/li_shared/archi_data/talkvid/clips/*/*.mp4"))
    if not files:
        print("no TalkVid data; skipping")
        return

    frames, res, k = 17, 128, 32768
    vidtok = load_vidtok(codebook=k, device="cpu")
    mimi = load_mimi(device="cpu")
    out = clip_tokens(files[0], vidtok, mimi, frames=frames, res=res)

    assert out["video"].shape == (1, (frames - 1) // 4 + 1, res // 8, res // 8)
    assert 0 <= int(out["video"].min()) and int(out["video"].max()) < k
    assert out["audio"].shape[:2] == (1, 32)
    assert out["ref"].shape == (1, 3, res, res)
    assert abs(out["audio"].shape[-1] - frames / 25 * mimi.frame_rate) <= 2


if __name__ == "__main__":
    test_clip_tokens_alignment()
    print("ok")
