"""Runnable checks for token_loss: FSQ partial credit, PaLM z-loss, label smoothing, gradient flow."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.model import token_loss


def test_ce_only_and_label_smoothing():
    """No weights -> plain CE (single part); label smoothing changes the value and still differentiates."""
    torch.manual_seed(0)
    B, L, V = 2, 5, 16
    logits, target = torch.randn(B, L, V, requires_grad=True), torch.randint(0, V, (B, L))
    ce, parts = token_loss(logits, target)
    assert "ce" in parts and torch.allclose(ce, parts["ce"])
    smoothed, _ = token_loss(logits, target, label_smoothing=0.1)
    assert not torch.allclose(ce, smoothed)
    smoothed.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_z_loss_grows_with_logit_magnitude():
    """PaLM z-loss penalises large logsumexp, so bigger logits -> bigger z term."""
    B, L, V = 2, 5, 16
    target = torch.randint(0, V, (B, L))
    _, small = token_loss(torch.zeros(B, L, V), target, z_weight=1.0)
    _, big = token_loss(torch.full((B, L, V), 5.0), target, z_weight=1.0)
    assert big["z"] > small["z"]


def test_fsq_loss_disabled():
    """FSQ partial-credit MSE is removed for streaming v2."""
    logits, target = torch.randn(2, 4, 16), torch.randint(0, 16, (2, 4))
    try:
        token_loss(logits, target, fsq_weight=0.5)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    test_ce_only_and_label_smoothing()
    test_z_loss_grows_with_logit_magnitude()
    test_fsq_loss_disabled()
    print("ok")
