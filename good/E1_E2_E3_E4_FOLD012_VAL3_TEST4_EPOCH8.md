# E1+E2+E3+E4 Clean-Split Eight-Epoch Experiment

## Protocol

- Model: original full-band E1+E2+E3+E4 direct additive residual model.
- Train: sampled manifest windows from Fold 0-2 patients.
- Validation: all 130,976 natural-timeline windows from Fold 3.
- Final test: all 95,774 natural-timeline windows from Fold 4.
- Epoch budget: 8.
- Checkpoint selection: maximum pooled Fold 3 natural AUPRC.
- Threshold selection: Fold 3 only, exact maximum F1 subject to recall >= 0.6.
- Fold 4 was evaluated once after checkpoint and threshold lock.
- Preprocessing, patient baseline construction, seed, and optimizer match the
  retained full-band four-branch baseline.
- No training augmentation was used.

## Training Curve

| Epoch | Training loss | Fold 3 AUROC | Fold 3 AUPRC | Selected |
| ---: | ---: | ---: | ---: | :---: |
| 1 | 0.430673 | 0.852158 | 0.049343 | Yes |
| 2 | 0.295546 | 0.729587 | 0.017093 | No |
| 3 | 0.254845 | 0.747714 | 0.061298 | Yes |
| 4 | 0.219299 | 0.752024 | 0.057576 | No |
| 5 | 0.195280 | 0.699054 | 0.046190 | No |
| 6 | 0.167410 | 0.761150 | 0.065269 | Yes |
| 7 | 0.150696 | 0.745177 | 0.061075 | No |
| 8 | 0.140967 | 0.749754 | 0.062494 | No |

The best checkpoint is epoch 6. Training loss decreases monotonically, while
cross-patient natural AUPRC does not, so final-epoch checkpointing is not valid
for this model.

## Locked Evaluation

| Metric | Fold 3 validation | Fold 4 final test |
| --- | ---: | ---: |
| AUROC | 0.761150 | 0.907100 |
| AUPRC | 0.065269 | 0.407771 |
| F1 at locked threshold | 0.030883 | 0.013125 |
| Recall at locked threshold | 0.601671 | 0.908832 |
| Specificity at locked threshold | 0.794047 | 0.497595 |
| False alarms/hour | 185.36 | 452.16 |

The threshold selected on Fold 3 was 0.00690384. On Fold 4 it produced 319 TP,
32 FN, 47,482 TN, and 47,941 FP.

## Interpretation

The new split produces strong unseen Fold 4 ranking metrics, especially AUPRC,
so the full-band fusion model contains useful cross-patient seizure information.
The very low F1 is not a ranking failure. It is a score calibration and decision
threshold transfer failure: the pooled Fold 3 threshold is far too permissive
for Fold 4.

The next experiment should keep this checkpoint fixed and calibrate each new
patient using enrollment normal EEG only. Candidate methods are a patient-normal
score quantile, robust median/MAD shift, or a false-alarm-constrained threshold
selected on Fold 3 and applied without Fold 4 labels. Do not retrain another
architecture before separating this threshold problem from model ranking.

## Artifacts

- Training script:
  `good/e1_e2_e3_e4_fullband/train_fold012_val3_test4.py`
- Result:
  `artifacts/chbmit/good_multibranch_fold012_val3_test4_epoch8/fold012_val3_test4_epoch8_f2dcef462ded.json`
- Checkpoint:
  `artifacts/chbmit/good_multibranch_fold012_val3_test4_epoch8/checkpoints/fold012_val3_best_epoch8.pt`
- Checkpoint SHA256:
  `d5b40eb82ff9e7726ad35a9b329e414c268958ee23f0bc60b582567b42e27747`
- Runtime: 2,751.70 seconds (45.86 minutes), including Fold 3 cache creation.
