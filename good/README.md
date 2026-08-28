# Retained EEG-VL Baselines

This directory is the development hub for the retained EEG-VL baselines and
promoted same-protocol candidates. The executable factories and training entry
points live here. The validated core layers remain in `ai/chbmit` and are
imported instead of copied, so bug fixes do not create divergent
implementations.

## Alarm-Policy Research

The isolated temporal alarm-policy package is under
[good/RL](RL/README.md). It consumes immutable probability timelines from
the retained model and compares fixed rules, supervised controls, PPO,
record-grouped GRPO, and GiGPO-style step credit without modifying the EEG
backbone.

## Same-Protocol Scheme C Leaderboard

Protocol fixed across this board:

- Train: `chb01-chb19`.
- Validation and threshold selection: `chb20-chb21`.
- Final test: `chb22-chb23`; labels are never used for epoch or threshold
  selection.
- `chb24` is unused.
- Five training epochs, seed 42, natural unbalanced validation/test timelines,
  and pooled metrics across each patient partition.
- The primary checkpoint is selected only by validation AUPRC. The table is
  displayed in descending final-test AUPRC after selection.

| Rank | Backbone | Pooling | STFT n/w/h | Micro/effective batch | Selected epoch | Validation AUPRC | Test AUROC | Test AUPRC | Test F1 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Qwen2.5-0.5B | `visual_mean` | 128/128/32 | 8/32 | 4 | **0.1344** | **0.9945** | **0.5285** | 0.2489 |
| 2 | Qwen2.5-0.5B | `visual_mean` | 64/64/32 | 8/32 | 1 | 0.1051 | 0.9915 | 0.5183 | **0.2774** |
| 3 | ModernBERT-base | `visual_mean` | 64/64/32 | 8/32 | 2 | 0.1101 | 0.9879 | 0.4993 | 0.2474 |
| 4 | Qwen2.5-0.5B | `visual_attention` | 64/64/32 | 8/32 | 5 | 0.0816 | 0.9923 | 0.4892 | 0.2080 |
| 5 | Qwen2.5-0.5B | `summary_token` | 64/64/32 | 8/32 | 5 | 0.0801 | 0.9923 | 0.4652 | 0.2065 |
| 6 | ModernBERT-base | `mean` | 64/64/32 | 32/32 | 2 | 0.1235 | 0.9882 | 0.4375 | 0.2722 |
| 7 | Qwen2.5-0.5B | `mean` over prompt and visual tokens | 64/64/32 | 32/32 | 3 | 0.1208 | 0.9788 | 0.4176 | 0.2514 |
| 8 | ModernBERT-base | `visual_mean` | 128/128/32 | 8/32 | 4 | 0.1137 | 0.9867 | 0.3825 | 0.2154 |

Promoted Qwen2.5 `visual_mean` implementation:

- Model factory: `good/e1_e2_e3_e4_qwen25_visual_mean/model.py`.
- Fixed Scheme C training entry point:
  `good/e1_e2_e3_e4_qwen25_visual_mean/train.py`.
- Architecture, protocol, result, and reproduction command:
  `good/e1_e2_e3_e4_qwen25_visual_mean/README.md`.

Current interpretation:

- Qwen2.5 `visual_mean` with STFT `128/128/32` is the current main result. It
  improves final-test AUPRC from 0.5183 to 0.5285 and AUROC from 0.9915 to
  0.9945 over the matched `64/64/32` run.

- The `64/64/32` run retains the best test F1 (0.2774 versus 0.2489), so S1 is
  an AUPRC/AUROC improvement rather than a uniform improvement at the fixed
  validation-selected threshold.
- ModernBERT `visual_mean` does not benefit from S1: changing STFT from
  `64/64/32` to `128/128/32` reduces test AUPRC from 0.4993 to 0.3825.

- `visual_attention` and `summary_token` do not improve over the simpler
  visual-token mean.
- The mean-pooling baselines used micro-batch 32, while the pooling ablations
  used micro-batch 8 with accumulation to 32. EfficientNet BatchNorm therefore
  makes the pooling comparison close, but not yet a perfectly isolated
  single-variable ablation.
- P3/P4 reports and result JSON are mirrored locally; their checkpoints remain
  on the training server.

## 1. E1+E2 STFT-64

Files:

- `good/e1_e2_stft64/model.py`: exact model factory and checkpoint identity.
- `good/e1_e2_stft64/train.py`: historical training protocol.
- Core model: `ai/chbmit/eeg_continual_pretrain_model.py`.
- Core training/data pipeline: `ai/chbmit/eeg_continual_pretrain.py`.

Architecture:

- Input: `(B, 1, 18, 1024)`, 4 seconds at 256 Hz.
- Deterministic 18-to-20 channel adapter.
- E1: `n_fft=64`, `win=64`, `hop=32`, `log1p(abs(STFT))`, EfficientNet-B0,
  32 visual tokens, Qwen2.5-0.5B with Q/V LoRA.
- Visual residual bypass enabled.
- E2: 20 channels by 6 relative-to-baseline frequency bands, 120 features.
- Fusion: Qwen summary + E1 visual residual + E2 residual, then LayerNorm,
  `tanh`, and a zero-initialized binary head.
- No per-window channel z-score.

Historical strict Fold 0 B0 result:

- AUROC: `0.9478`
- AUPRC: `0.3188`
- F1: `0.0834`
- Recall: `0.6277`
- False alarms/hour: `16.75`

Reproduce the retained one-epoch checkpoint:

```powershell
C:\ProgramData\anaconda3\Scripts\conda.exe run --no-capture-output -n pytorch `
  python -m good.e1_e2_stft64.train
```

## 2. E1+E2+E3+E4 Full-Band Direct Residual

Files:

- `good/e1_e2_e3_e4_fullband/model.py`: exact ungated model factory.
- `good/e1_e2_e3_e4_fullband/train.py`: historical five-epoch experiment.
- Core model: `ai/chbmit/eegvl_multibranch_model.py`.
- Core training pipeline: `ai/chbmit/eegvl_multibranch_experiment.py`.

Architecture:

- Input and 18-to-20 channel adapter are the same as E1+E2.
- E1: `n_fft=256`, `win=128`, `hop=32`, full-band STFT and EfficientNet-B0.
- E2: relative spectrum, 120 features.
- E3: high-frequency power ratio and spike density, 40 features.
- E4: four band-power ratios per channel, 80 features.
- Fusion is direct and ungated:
  `LLM + Linear(E2) + Linear(E3) + Linear(E4)`.
- Qwen2.5-0.5B is frozen except Q/V LoRA.
- No per-window channel z-score.

Historical result:

- Fold 4 selected epoch 1: AUROC `0.9007`, AUPRC `0.3309`.
- Strict Fold 0 transfer: AUROC `0.9427`, AUPRC `0.0253`.

The Fold 4 score is a development result, not an unseen final-test result. This
model remains useful as the four-branch architecture baseline, but its transfer
failure must not be hidden.

```powershell
C:\ProgramData\anaconda3\Scripts\conda.exe run --no-capture-output -n pytorch `
  python -m good.e1_e2_e3_e4_fullband.train
```

## Development Rules

1. Promote active candidates only through the same-protocol leaderboard.
2. Use the same patient split, preprocessing cache, seed, epoch budget, and
   threshold protocol for every comparison.
3. Report both AUROC and AUPRC on natural timelines. Never describe AUPRC as
   AUROC.
4. Do not use the final test fold for epoch, threshold, or hyperparameter
   selection.
5. Add E3, E4, channel attention, gates, or normalization one change at a time.
6. Preserve raw probability arrays and per-patient metrics for every promoted
   experiment.

## Historical Artifacts

E1+E2:

- Result: `artifacts/chbmit/eeg_continual_pretrain_strict_e2_smoke/fold0_pretrain_c27817a49668.json`
- Checkpoint: `artifacts/chbmit/eeg_continual_pretrain_strict_e2_smoke/checkpoints/fold0_lora_stft_best.pt`
- Checkpoint SHA256: `c7c0683738d66a8476b17c642ff380078e78306665be4d54c180ee9cc1a48bde`

E1+E2+E3+E4:

- Result: `artifacts/chbmit/eegvl_multibranch_fullband/fold0_e1_e2_e3_e4_f1660457394b.json`
- Checkpoint: `artifacts/chbmit/eegvl_multibranch_fullband/checkpoints/fold0_e1_e2_e3_e4_best.pt`
- Checkpoint SHA256: `52d85560992237d661270e4ebd3a3db83391b8d288248f8971269e127c3a1873`
