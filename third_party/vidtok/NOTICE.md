# Vendored: VidTok (Microsoft)

Frozen video tokenizer. Source: https://github.com/microsoft/VidTok @ `6b18d25edbccbb5ca967750b489f69aa3616aa9d` (MIT, see `LICENSE`).

Trimmed to the inference subset needed for the FSQ (causal 4x8x8) autoencoder:

```
vidtok/models/autoencoder.py
vidtok/modules/{util,ema,regularizers,distributions,model_3dcausal}.py
```

Dropped upstream: training/data/logger/loss/discriminator/lpips modules and the v1.1 and non-causal model variants. Code is unmodified; the training loss is bypassed at load time by `sang/video.py` (`loss_config -> torch.nn.Identity`).
