# ModernBERT Base Model Development Plan

## Current Baseline

The retained experiment uses the Scheme C patient split:

- Training: chb01-chb19.
- Validation: chb20-chb21.
- Final test: chb22-chb23.
- Unused: chb24.
- Input: 20 channels, 4 seconds, 256 Hz.
- Checkpoint selection: highest validation AUPRC.
- Training budget: 5 epochs.

### AUPRC-best result

| Model | Best epoch | Validation AUROC | Validation AUPRC | Test AUROC | Test AUPRC | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5 Scheme C | 3 | 0.8691 | 0.1208 | 0.9788 | 0.4176 | 0.2514 |
| ModernBERT Scheme C | 2 | 0.8499 | 0.1235 | 0.9882 | 0.4375 | 0.2722 |

The current ModernBERT run improves test AUPRC by 0.0199 and test F1 by
0.0208. This is a single-run result and is not yet evidence of a stable model
improvement.

The ModernBERT test-set patient metrics are:

| Patient | AUROC | AUPRC |
|---|---:|---:|
| chb22 | 0.9990 | 0.7402 |
| chb23 | 0.9913 | 0.5536 |
| Macro patient | 0.9952 | 0.6469 |
| Pooled chb22-chb23 | 0.9882 | 0.4375 |

The macro-to-pooled AUPRC gap indicates patient-dependent score calibration.
Within-patient ranking is strong, but score scales are not aligned between
patients.

## Experimental Rules

All B1-B4 experiments must retain the following controls unless the experiment
explicitly names the changed variable:

- Keep the chb01-19/chb20-21/chb22-23 patient split fixed.
- Never use chb22-chb23 labels for checkpoint, threshold, or hyperparameter
  selection.
- Select checkpoints using validation AUPRC.
- Report pooled and macro-patient AUROC/AUPRC together.
- Report sensitivity, precision, F1, specificity, false alarms per hour, and
  the confusion matrix at the validation-selected threshold.
- Record the random seed, model contract, dataset hashes, checkpoint SHA256,
  and per-epoch history.
- Change one primary factor per experiment.

## B1: Stability Verification

### Goal

Determine whether ModernBERT provides a repeatable improvement over Qwen2.5,
rather than a favorable single-seed result.

### Experiments

Run Qwen2.5 Scheme C and ModernBERT Scheme C with three matched seeds. Keep the
data manifest, sampling, preprocessing, learning rates, batch sizes, epoch
budget, and checkpoint rule identical.

### Outputs

- Per-seed validation and test metrics.
- Mean, standard deviation, median, and range for test AUPRC/AUROC/F1.
- Paired ModernBERT-minus-Qwen differences for each seed.
- Training curves and checkpoint epochs.

### Acceptance

Retain ModernBERT as the preferred language backbone when:

- Mean test AUPRC improves by at least 0.01.
- At least two of three paired seeds improve test AUPRC.
- Mean test AUROC does not decrease by more than 0.005.
- No run contains non-finite loss or a collapsed classifier.

If these conditions fail, treat ModernBERT and Qwen2.5 as equivalent and use
the cheaper or more stable model.

## B2: Patient Score Calibration

### Goal

Align output score scales between patients without using seizure labels from
validation or test patients.

### Candidate methods

Use only each patient's enrollment/rest normal windows to fit calibration:

1. Rest-logit z-score:

   `score = (raw_logit - rest_logit_mean) / max(rest_logit_std, epsilon)`

2. Rest percentile score based on the empirical distribution of normal
   enrollment logits.

3. Robust rest calibration using median and MAD when the standard deviation is
   unstable.

Fit any global calibration hyperparameter on chb20-chb21 only. Apply the frozen
calibration rule to chb22-chb23.

### Outputs

- Raw and calibrated patient score distributions.
- Raw and calibrated pooled/macro AUPRC.
- Per-patient AUPRC before and after calibration.
- Threshold metrics and false alarms per hour.

### Acceptance

- Improve pooled test AUPRC by at least 0.02.
- Do not reduce macro-patient test AUPRC by more than 0.01.
- Do not use ictal labels from chb22-chb23.
- Keep test recall at or above 0.70 at the validation-selected threshold.

## B3: False-Positive Reduction

### Goal

Reduce normal-window false alarms while retaining seizure recall. The current
ModernBERT AUPRC-best checkpoint has test recall 0.7325, 573 false positives,
and 9.99 false alarms per hour.

### Experiment order

1. Hard-negative mining: collect the highest-scoring training normal windows
   after a baseline run and include them in a short refit.
2. Compare cross-entropy against focal loss or asymmetric focal loss.
3. Use patient-balanced batches so high-volume patients cannot dominate the
   gradient.
4. Evaluate a temporal persistence rule as a separate post-processing result;
   do not mix it with window-level model metrics.

### Acceptance

- Keep test recall at or above 0.70.
- Reduce false alarms per hour by at least 20% from 9.99.
- Do not reduce pooled test AUPRC by more than 0.01.
- Verify that gains are not limited to only chb22 or only chb23.

## B4: Feature And Fusion Upgrade

### Goal

Improve representation quality only after B1 establishes a stable baseline and
B2 addresses score calibration.

### B4.1 Multi-scale STFT

Combine complementary resolutions:

- STFT-64 for transient and spike timing.
- STFT-256 for narrow-band frequency structure.

Compare the retained STFT-64 model against a dual-scale encoder while keeping
ModernBERT, E2/E3/E4, sampling, and loss unchanged.

### B4.2 Controlled branch fusion

Replace unconditional residual addition with monitored scalar branch weights:

`representation = main + alpha * E2 + beta * E3 + gamma * E4`

Initialize the weights conservatively and log each branch's feature norm and
effective contribution. Do not assign E2's high learning rate to gates or
normalization parameters.

### Acceptance

- Improve mean validation AUPRC across the B1 seeds by at least 0.01.
- Improve pooled test AUPRC without reducing macro-patient AUPRC.
- Show that the gain persists in at least two seeds.
- Keep peak GPU memory within the available 16 GB device.

## Execution Order

1. B1 matched-seed comparison.
2. B2 patient score calibration on the retained ModernBERT checkpoints.
3. B3 hard-negative mining, followed by one loss-function comparison only if
   hard-negative mining is insufficient.
4. B4 multi-scale STFT, then controlled branch fusion.
5. Freeze the final base-model contract and checkpoint before starting
   continual-learning adaptation.

Do not add larger language models, increase the epoch budget, or change several
training factors together during B1-B4. The next continual-learning phase must
start from the frozen base checkpoint and use new-patient rest data only for E2
baseline construction and score calibration before labeled adaptation begins.
