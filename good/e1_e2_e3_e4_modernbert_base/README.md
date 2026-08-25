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
- Default pooling: mean over all 67 contextualized tokens.
- Epoch, optimizer, sampling, baseline, checkpoint and threshold protocols are
  identical to the retained Qwen2.5 Scheme C experiment.

The package also provides matched `visual_mean` entry points. These average
only the 32 contextualized EEG tokens and exclude the 35 prompt tokens.

## Same-protocol results

| Pooling | STFT n/w/h | Selected epoch | Validation AUPRC | Test AUROC | Test AUPRC | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| `visual_mean` | 64/64/32 | 2 | 0.1101 | 0.9879 | **0.4993** | **0.2474** |
| `mean` over all tokens | 64/64/32 | 2 | **0.1235** | **0.9882** | 0.4375 | 0.2722 |
| `visual_mean` | 128/128/32 | 4 | 0.1137 | 0.9867 | 0.3825 | 0.2154 |

The `128/128/32` S1 change is rejected for ModernBERT because it lowers test
AUPRC by 0.1168 relative to the matched `64/64/32` `visual_mean` run.

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

Run the matched `visual_mean` experiment:

```powershell
C:\ProgramData\anaconda3\Scripts\conda.exe run --no-capture-output -n qwen35-eeg `
  python -m good.e1_e2_e3_e4_modernbert_base.train_visual_mean `
  --allow-model-download
```

Run the S1 `128/128/32` experiment:

```powershell
C:\ProgramData\anaconda3\Scripts\conda.exe run --no-capture-output -n qwen35-eeg `
  python -m good.e1_e2_e3_e4_modernbert_base.train_visual_mean_stft_s1 `
  --allow-model-download
```
