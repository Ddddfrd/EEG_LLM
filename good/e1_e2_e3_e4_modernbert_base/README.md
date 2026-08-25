# Scheme C with ModernBERT-base

This candidate keeps the Scheme C EEG pipeline fixed and replaces the causal
Qwen decoder with the bidirectional `answerdotai/ModernBERT-base` encoder.

- Train: chb01-chb19.
- Validation: chb20-chb21.
- Final test: chb22-chb23.
- Input: direct 20-channel, 4-second EEG windows at 256 Hz.
- E1: STFT-64 and EfficientNet-B0, 32 visual tokens.
- E2/E3/E4: unchanged direct residual branches.
- Sequence: 35 trainable prompt embeddings plus 32 EEG visual tokens.
- ModernBERT: 22 layers, hidden size 768, bidirectional SDPA attention.
- LoRA: attention-only fused `Wqkv` and `attn.Wo` projections, rank 8 (44 modules total).
- Pooling: mean over all 67 contextualized tokens.
- Epoch, optimizer, sampling, baseline, checkpoint and threshold protocols are
  identical to the retained Qwen2.5 Scheme C experiment.

The retained training result and B1-B4 base-model experiments are documented in
[BASE_MODEL_DEVELOPMENT_PLAN.md](BASE_MODEL_DEVELOPMENT_PLAN.md).

Download the model once and run a real-weight smoke test:

```powershell
C:\ProgramData\anaconda3\Scripts\conda.exe run --no-capture-output -n qwen35-eeg `
  python -m good.e1_e2_e3_e4_modernbert_base.smoke --batch-size 32
```

Run the five-epoch comparison:

```powershell
C:\ProgramData\anaconda3\Scripts\conda.exe run --no-capture-output -n qwen35-eeg `
  python -m good.e1_e2_e3_e4_modernbert_base.train `
  --allow-model-download
```
