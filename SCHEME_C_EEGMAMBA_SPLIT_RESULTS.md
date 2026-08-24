# Scheme C on the EEGMamba Patient Split

## Conclusion

Scheme C was trained on chb01-chb19, selected on chb20-chb21, and evaluated once on the untouched chb22-chb23 test set. The AUPRC-selected checkpoint is the preferred result:

- Test AUROC: **0.9787698932**
- Test AUPRC: **0.4176282214**
- Test F1 at the validation-selected threshold: **0.2513542795**
- Test sensitivity/specificity: **0.738853504 / 0.987412371**
- False alarms per negative hour: **11.3289**

The ranking result is strong on chb22-chb23, but the raw window-level operating point still produces too many false alarms. This experiment supports using the model as a base model for subsequent patient calibration and continual-learning research; it does not yet support deployment as an event alarm.

## Protocol

- Training patients: chb01-chb19.
- Validation patients: chb20-chb21.
- Final test patients: chb22-chb23.
- Unused patient: chb24.
- Validation and test patients are disjoint.
- Training windows: 32,739 total, including 9,819 ictal and 22,920 normal windows.
- Validation: 54,370 natural-distribution windows, including 119 ictal windows.
- Test: 51,795 natural-distribution windows, including 157 ictal windows.
- Model: Scheme C E1+E2+E3+E4 with Qwen residual fusion.
- Input: 20 channels, 4 seconds, 256 Hz.
- STFT: `n_fft=64`, `win_length=64`, `hop_length=32`.
- Training sampling: approximately 3:7 positive:negative per patient, capped at 20,000 windows per patient.
- E2 baseline: each patient's earliest known-normal windows, first 20%, capped at 4,000 windows.
- Training: five epochs, bf16, AdamW, RTX 4090 Laptop GPU.
- Checkpoint and threshold selection use only chb20-chb21. Test labels are not used for epoch or threshold selection.

## Epoch History

| Epoch | Loss | Validation AUROC | Validation AUPRC | Selection |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.322510 | 0.839253 | 0.110532 | Both improved |
| 2 | 0.209921 | 0.856340 | 0.110578 | Both improved |
| 3 | 0.144212 | 0.869086 | **0.120794** | AUPRC best |
| 4 | 0.094455 | **0.879917** | 0.109398 | AUROC best |
| 5 | 0.068877 | 0.863696 | 0.106157 | Neither improved |

Training loss continued to decrease while both validation ranking metrics fell in epoch 5. The checkpoint selection prevented the final overfit epoch from being used.

## Independent Test Results

| Validation selection | Epoch | Threshold | Test AUROC | Test AUPRC | F1 | Sensitivity | Specificity | False alarms/hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AUROC best | 4 | 0.399343 | **0.986930** | 0.288346 | 0.186567 | 0.796178 | 0.979511 | 18.4399 |
| AUPRC best | 3 | 0.467332 | 0.978770 | **0.417628** | **0.251354** | 0.738854 | **0.987412** | **11.3289** |

The AUPRC-selected epoch 3 checkpoint is preferable for this imbalanced seizure-detection task. It materially improves precision, F1, and false-alarm rate while retaining high AUROC.

## Per-Patient Test Results

### AUPRC-selected checkpoint

| Patient | Windows | Ictal windows | AUROC | AUPRC | F1 | Sensitivity | Specificity | False alarms/hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chb22 | 27,897 | 50 | 0.996453 | 0.689642 | 0.161616 | 0.960000 | 0.982188 | 16.0305 |
| chb23 | 23,898 | 107 | 0.973292 | 0.408070 | 0.413374 | 0.635514 | 0.993527 | 5.8257 |

### AUROC-selected checkpoint

| Patient | AUROC | AUPRC | F1 | Sensitivity | Specificity | False alarms/hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chb22 | 0.995090 | 0.335561 | 0.113821 | 0.980000 | 0.972636 | 24.6274 |
| chb23 | 0.989060 | 0.451447 | 0.317328 | 0.710280 | 0.987558 | 11.1975 |

Both test patients have high AUROC, so the pooled result is not caused by one successful patient hiding one failed patient. The large chb22 AUPRC difference between checkpoints also shows that AUROC-only checkpoint selection is unstable under severe class imbalance.

## Interpretation

The test prevalence is only 157/51,795, approximately **0.303%**. A random ranking would therefore have an expected AUPRC near 0.00303, while the preferred checkpoint reaches 0.4176. The high AUROC and much lower AUPRC are not contradictory: AUROC is relatively insensitive to the large number of normal windows, whereas AUPRC directly penalizes false positive predictions.

The F1 score is lower than the ranking metrics because the threshold was selected on chb20-chb21 and transferred unchanged to chb22-chb23. This is the correct held-out procedure, but patient-specific score calibration differs. The validation split itself is heterogeneous: at the AUPRC-best checkpoint chb20/chb21 AUROC is 0.7710/0.9626 and AUPRC is 0.0965/0.1876.

The remaining practical issue is false alarms. Even the preferred checkpoint produces 650 false-positive windows over 57.38 negative hours. Future evaluation should add temporal aggregation, refractory periods, and event-level sensitivity/false alarms per hour before interpreting it as a seizure alarm.

## Comparison Limits

- The earlier local Scheme C run used chb01-09 and chb15-24 for training and reused chb10-14 as its development validation/test set. Its best AUROC was 0.833074 and best AUPRC was 0.345867, but that protocol and patient difficulty are different, so the values are not a controlled model comparison.
- The EEGMamba paper reports AUROC 0.8938 +/- 0.0161 and AUPRC 0.3885 +/- 0.0418 for this patient partition. This run is numerically higher on one seed, but only the patient IDs are aligned. EEGMamba uses a different architecture, preprocessing, window length, pretraining setup, training duration, and five-seed reporting. Therefore this result must not be described as outperforming EEGMamba.
- A controlled architecture comparison requires running Scheme C and EEGMamba with the same cached windows, channel mapping, labels, seed set, checkpoint rule, and event-level evaluator.

## Artifacts

- Result JSON: `artifacts/chbmit/scheme_c_eegmamba_split/scheme_c_eegmamba_split_ca90a80a023b.json`
- AUROC checkpoint: `artifacts/chbmit/scheme_c_eegmamba_split/checkpoints/train_chb01_19_best_auroc.pt`
- AUPRC checkpoint: `artifacts/chbmit/scheme_c_eegmamba_split/checkpoints/train_chb01_19_best_auprc.pt`
- Artifact SHA256: `ca90a80a023bc494ca9332659846ba072d5a010a679b061b58a3472855efc104`
- Runtime: 1,774.12 seconds end-to-end; 1,287.20 seconds for training.

The result JSON and both checkpoint SHA256 hashes were verified after training.
