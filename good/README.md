# Retained EEG-VL Baselines

This directory is the development hub for the two retained EEG-VL baselines.
The executable factories and training entry points live here. The validated
core layers remain in `ai/chbmit` and are imported instead of copied, so bug
fixes do not create two divergent implementations.

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

1. Keep these two model families as the only active architecture baselines.
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
