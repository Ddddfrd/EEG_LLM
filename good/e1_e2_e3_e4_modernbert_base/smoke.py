"""Run a real-weight ModernBERT Scheme C CUDA forward/backward smoke test."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import torch

from ai.chbmit.eeg_continual_pretrain_model import ServerSTFTConfig

from .model import build_model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("ModernBERT smoke test requires CUDA")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(
        local_files_only=True,
        pretrained_visual_encoder=True,
        stft_config_override=ServerSTFTConfig(
            source_channels=20,
            eeg_channels=20,
            n_fft=64,
            win_length=64,
            hop_length=32,
            zscore_input=False,
        ),
    ).to(device)
    model.train()
    waveform = torch.randn(args.batch_size, 1, 20, 1024, device=device) * 0.02
    baseline = torch.zeros(args.batch_size, 20, 33, device=device)
    labels = torch.arange(args.batch_size, device=device) % 2
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(waveform, baseline_log_magnitude=baseline)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    finite_gradients = [
        bool(parameter.grad is not None and torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not finite_gradients or not all(finite_gradients):
        raise RuntimeError("Missing or non-finite trainable gradients")
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(device),
                "batch_size": args.batch_size,
                "logits_shape": list(logits.shape),
                "loss": float(loss.detach().cpu()),
                "hidden_size": model.hidden_size,
                "lora_modules": len(model.lora_module_names),
                "parameters": model.parameter_summary(),
                "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
