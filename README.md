# EEG_LLM

Research training pipelines for seizure detection with EEG encoders, Qwen2.5
LoRA, and patient-relative spectral features. This repository contains research
code only and is not a medical device.

## Available Model Families

1. **Mamba-C**: 72 direct EEGMamba tokens, Qwen2.5-0.5B Q/V LoRA, and an E2
   patient-relative spectral residual. This remains the original pipeline.
2. **E1+E2 STFT-64**: STFT-64 EfficientNet-B0 visual tokens, Qwen Q/V LoRA,
   a visual residual, and an E2 relative-baseline residual.
3. **E1+E2+E3+E4 full-band**: full-band STFT EfficientNet-Qwen with direct,
   ungated E2 relative-spectrum, E3 transient-spike, and E4 band-ratio residuals.

The retained EfficientNet-Qwen baselines and training entry points are under
`good/`. See `good/README.md` for architecture contracts, commands, historical
metrics, and checkpoint identities. Generated weights remain excluded.

## Architecture

```text
4 s EEG window (B, 1, 18, 1024), 256 Hz
  -> Fourier resample to 200 Hz
  -> EEGMamba patches (B, 18, 4, 200)
  -> official 12-layer alternating Mamba2 encoder
  -> 72 tokens (18 channels x 4 patches), no token pooling
  -> channel embedding + patch embedding
  -> Linear(200, 896)
  -> concatenate 35 trainable prompt tokens
  -> Qwen2.5-0.5B, Q/V LoRA only
  -> sequence pooling

Patient normal baseline
  -> log1p STFT relative spectrum, 20 channels x 6 bands = 120 E2 features
  -> MLP 120 -> 256 -> 896
  -> additive residual before the binary classification head
```

The Qwen sequence contains exactly `35 + 72 = 107` tokens. The visual path
does not use the 32-token adaptive pooling from the EfficientNet experiment.

## Evaluation Protocol

- Fold 1-3 patients: model fitting.
- Fold 4 patients: checkpoint and threshold selection.
- Fold 0 patients: one final evaluation with the frozen threshold.
- E2 baseline: earliest 128 known-normal windows from each patient.
- Training augmentation: disabled in the initial matched comparison.

## Environment

Linux, CUDA, and an NVIDIA GPU are required. The validated development
environment used Python 3.10, PyTorch 2.5.1+cu124, `mamba-ssm` 2.2.4,
`causal-conv1d` 1.5.0.post8, and Transformers 4.49.0.

Install a CUDA-enabled PyTorch build first, then install the remaining
dependencies. `mamba-ssm` may require `--no-build-isolation` so it can find the
installed PyTorch package.

```bash
conda create -n eeg-llm python=3.10 -y
conda activate eeg-llm
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Required Local Files

The repository intentionally excludes datasets, caches, pretrained weights,
Qwen files, and generated checkpoints. Before training, provide:

1. A strict Fold 0 reference artifact produced by the included CHB-MIT data
   pipeline. It records the sampled training cache, full natural Fold 4 cache,
   full natural Fold 0 cache, manifest, and patient-disjoint split.
2. The official EEGMamba checkpoint `pretrained_EEGMamba.pth` from
   `weighting666/EEGMamba`. The loader verifies SHA256
   `b452bb29ecf1d6131ba82a50c6e13823ec1d660d9009d013e691d19b2916f4fe`.
3. A local Hugging Face cache for `Qwen/Qwen2.5-0.5B`, or explicit permission
   to download it with `--allow-model-download`.

## Train Mamba-C

```bash
python -m ai.chbmit.eegmamba_c_experiment \
  --reference-artifact /path/to/fold0_pretrain_<hash>.json \
  --official-checkpoint /path/to/pretrained_EEGMamba.pth \
  --output-dir artifacts/chbmit/eegmamba_c_fold0 \
  --max-epochs 1 \
  --micro-batch-size 16 \
  --prediction-batch-size 64
```

The default effective batch size is 128 through gradient accumulation. The
optimizer uses separate learning rates:

- EEGMamba backbone: `2e-5`
- Qwen Q/V LoRA: `2e-5`
- projection, prompt, E2, and classification head: `1e-4`

The output checkpoint is portable: it includes EEGMamba and task weights plus
Qwen LoRA parameters, but excludes frozen Qwen base weights.

## Train Scheme C on the EEGMamba Patient Split

The retained Scheme C experiment uses the EEGMamba paper's patient IDs while
keeping the local E1+E2+E3+E4 architecture and training protocol:

- Train: `chb01`-`chb19`
- Validation and checkpoint/threshold selection: `chb20`-`chb21`
- Final untouched test: `chb22`-`chb23`
- Unused: `chb24`
- Input: direct 20-channel, 4-second windows at 256 Hz
- E1 STFT: `n_fft=64`, `win_length=64`, `hop_length=32`
- E2: per-patient earliest-normal relative spectral baseline
- E3/E4: transient-spike and band-ratio residuals
- Training: five epochs with independent AUROC-best and AUPRC-best checkpoints

```bash
python -m good.e1_e2_e3_e4_fullband.train_scheme_c_eegmamba_split \
  --reference-artifact /path/to/fold0_pretrain_<hash>.json \
  --data-root /path/to/chbmit/1.0.0 \
  --output-dir artifacts/chbmit/scheme_c_eegmamba_split \
  --shared-cache-dir artifacts/chbmit/good_multibranch_scheme_c_aligned/cache
```

The reported one-seed independent test result is AUROC `0.9788` and AUPRC
`0.4176` for the validation-AUPRC-selected checkpoint. See
`SCHEME_C_EEGMAMBA_SPLIT_RESULTS.md` for thresholds, per-patient metrics,
false-alarm rates, hashes, and comparison limits. This experiment aligns the
patient partition only; it is not an EEGMamba architecture reproduction.

## Main Files

- `ai/chbmit/eegmamba_c.py`: 72-token EEGMamba-Qwen-E2 model.
- `ai/chbmit/eegmamba_c_experiment.py`: CUDA training and strict Fold 0 evaluation.
- `ai/chbmit/eegmamba_b.py`: official-checkpoint-compatible EEGMamba backbone.
- `ai/chbmit/eeg_continual_pretrain_model.py`: STFT, E2, and LoRA components.
- `ai/chbmit/direct20.py`: deterministic direct 20-channel CHB-MIT mapping.
- `good/e1_e2_e3_e4_fullband/train_scheme_c_eegmamba_split.py`: five-epoch
  Scheme C training, validation selection, and independent test entry point.
- `tests/test_chbmit_eegmamba_c.py`: shape, freezing, and portable-state tests.

The implementation is a hybrid research extension inspired by EEGMamba and
EEG-VL; it is not a verbatim reproduction of either paper.
