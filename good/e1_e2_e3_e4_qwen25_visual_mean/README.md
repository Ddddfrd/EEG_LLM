# Qwen2.5 Visual-Mean Scheme C

This package is the promoted, directly runnable version of the Qwen2.5-0.5B
`visual_mean` pooling experiment. It reuses the validated E1-E4 implementation
and fixes pooling to the 32 contextualized EEG visual tokens. The 35 prompt
hidden states are excluded from the final average.

## Fixed architecture

- Input: 4-second, 18-channel CHB-MIT windows at 256 Hz.
- Channel adapter: deterministic 18-to-20 conversion.
- E1: STFT (`n_fft=64`, `win_length=64`, `hop_length=32`), EfficientNet-B0,
  and 32 visual tokens.
- E2: 120 patient-relative baseline spectrum features.
- E3: 40 high-frequency ratio and spike-density features.
- E4: 80 frequency-band ratio features.
- Language backbone: frozen Qwen2.5-0.5B with Q/V LoRA.
- Sequence: 35 trainable prompt embeddings followed by 32 visual embeddings.
- Pooling: `hidden[:, 35:67].mean(dim=1)`.
- Fusion: pooled language representation plus direct E2/E3/E4 residuals.

## Scheme C protocol

- Train: `chb01-chb19`.
- Validation and threshold selection: `chb20-chb21`.
- Final test: `chb22-chb23`.
- `chb24` is unused.
- Five epochs, seed 42, BF16, micro-batch 8, effective batch 32.
- Checkpoints are selected on validation metrics only.

The retained `64/64/32` validation-AUPRC-selected epoch 1 checkpoint achieved:

| Validation AUROC | Validation AUPRC | Test AUROC | Test AUPRC | Test F1 |
|---:|---:|---:|---:|---:|
| 0.8400 | 0.1051 | 0.9915 | **0.5183** | **0.2774** |

Authoritative report:
`artifacts/chbmit/scheme_c_qwen25_05b_pooling_ablation/visual_mean/SCHEME_C_EEGMAMBA_SPLIT_RESULTS.md`.

The S1 `128/128/32` validation-AUPRC-selected epoch 4 checkpoint achieved:

| Validation AUROC | Validation AUPRC | Test AUROC | Test AUPRC | Test F1 |
|---:|---:|---:|---:|---:|
| 0.9117 | **0.1344** | **0.9945** | **0.5285** | 0.2489 |

Authoritative S1 report:
`artifacts/chbmit/scheme_c_qwen25_05b_visual_mean_stft_s1_128_128_32/SCHEME_C_EEGMAMBA_SPLIT_RESULTS.md`.

Run the fixed five-epoch experiment:

```powershell
C:\ProgramData\anaconda3\Scripts\conda.exe run --no-capture-output -n qwen35-eeg `
  python -u -m good.e1_e2_e3_e4_qwen25_visual_mean.train
```

Run the S1 single-variable STFT experiment (`128/128/32`):

```powershell
C:\ProgramData\anaconda3\Scripts\conda.exe run --no-capture-output -n qwen35-eeg `
  python -u -m good.e1_e2_e3_e4_qwen25_visual_mean.train_stft_s1
```
