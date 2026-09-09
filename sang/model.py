"""Model factory."""
from torch import nn


def build_talking_head(cfg: dict, fsq_codes=None) -> nn.Module:
    """Build the streaming block-causal talking head from a config dict.

    Slice layout: motion_ctx gives [identity ref, motion ctx, content x tv]; otherwise
    [ref x ref_slices, content...]. `continuous` swaps FSQ token prediction for a flow head
    over VAE latents (v4)."""
    from sang.streaming_transformer import StreamingTalkingHead

    tv = (cfg["frames"] - 1) // 4 + 1
    if cfg.get("motion_ctx"):
        ref_slices, tv = 2, tv + 2
    else:
        ref_slices = cfg.get("ref_slices", 1)
        tv = tv + ref_slices - 1
    continuous = cfg.get("continuous", False)
    return StreamingTalkingHead(
        video_card=cfg["codebook"], dim=cfg["dim"], num_heads=cfg["heads"],
        num_layers=cfg["layers"], dropout=cfg.get("dropout", 0.0),
        tv=tv, spatial=cfg["res"] // 8, ref_slices=ref_slices,
        audio_dim={"wavlm-base": 768, "wavlm-large": 1024}[cfg["audio_encoder"]],
        audio_dropout=cfg.get("audio_dropout", 0.0),
        factorized_head=cfg.get("factorized_head", True) and not continuous,
        coupled_fsq_head=cfg.get("coupled_fsq_head", False),
        learn_logit_scale=cfg.get("learn_logit_scale", True),
        fsq_codes=None if continuous else fsq_codes,
        mask_schedule=cfg.get("mask_schedule", "per_slice_cosine"),
        p_inference_shaped=cfg.get("p_inference_shaped", 0.0),
        bridge_init=cfg.get("bridge_init", False),
        audio_lookahead=cfg.get("audio_lookahead", 0),
        use_struct_emb=cfg.get("face_cond", False),
        continuous=continuous, z_ch=int(cfg.get("z_ch", 16)) if continuous else 0,
        diff_depth=cfg.get("diff_depth", 3),
    )
