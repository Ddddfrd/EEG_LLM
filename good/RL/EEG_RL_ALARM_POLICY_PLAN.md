# EEG RL Alarm Decision Policy Development Plan

## 0. Implementation Status (2026-08-28)

R1-R4 are complete under the isolated `good/RL` package:

- R1 selected the robust fixed rule `threshold=0.90`, `2-of-5`, `300 s` on
  chb20-chb21 with 12/12 events and `0.415` false alarms/hour.
- R2 trained frozen logistic and 32x32 MLP causal temporal controls.
- R3 ran tabular Q-learning and five PPO seeds. Median-seed PPO failed the
  sensitivity gate and was excluded from final evaluation.
- R4 froze the complete protocol before exporting chb22-chb23 and ran one
  no-retuning final evaluation.

The final base probability metrics are AUROC `0.9945` and AUPRC `0.5285`. The
predeclared robust rule detected 9/10 events at `0.471` false alarms/hour but
failed the chb22 patient guardrail. The frozen MLP comparator detected 10/10 at
`0.837` false alarms/hour. Because that comparison was observed on the final
cohort, the MLP is a candidate for a new confirmation experiment rather than a
retrospectively promoted winner.

The L1 stop condition is met: PPO did not beat the tuned deterministic or
supervised controls. The next step is strict Tier B confirmation, not more PPO
tuning and not L2 model-output RL. See `EEG_RL_ALARM_POLICY_RESULTS.md` for the
complete evidence and figures.

### VERL-inspired G0-G3 status

G0 mathematical parity is complete. The independent binary-policy GRPO, RLOO,
and exact-anchor GiGPO advantage functions match the unmodified vendored
`verl-agent` reference implementation across 11 fixed cases. Maximum absolute
error is `1.192e-7` at a declared `1e-6` tolerance. The result is recorded in
`G0_VERL_ADVANTAGE_PARITY.md` and a content-addressed JSON artifact.

This gate validates formula translation, mask behavior, repeated-trajectory
handling, zero-variance groups, and singleton behavior.

G1 subsequently ran record-grouped GRPO for five seeds. Sparse initialization
fixed the refractory-saturation exploration dead zone, but all deterministic
chb21 policies converged to silence. G3 added GiGPO-style return-to-go step
advantages: it fixed credit dilution on chb20 but still failed to transfer to
chb21. Both stages stopped at the development gate and did not read
chb22-chb23. G2 RLOO remains skipped because it does not address episode-level
credit dilution.

## 1. Objective

This work evaluates whether reinforcement learning improves the temporal alarm
policy placed after the retained EEG seizure classifier. The EEG feature
extractor and classifier remain frozen in the first phase. RL consumes cached
probability timelines and decides whether to emit an alarm at each 4-second
step.

The primary question is:

> Can a learned sequential policy reduce false alarm episodes and detection
> latency without reducing event sensitivity, compared with a fully tuned
> threshold, voting, and refractory-period baseline?

This phase does not attempt to improve window-level AUROC or AUPRC. Those
metrics are properties of the frozen probability ranking and must remain
identical across all decision policies.

## 2. Decision and Scope

### 2.1 In scope

- Freeze the promoted Qwen2.5 `visual_mean` model with STFT `128/128/32`.
- Persist complete natural-timeline prediction artifacts for policy training.
- Add an evaluator that accepts explicit alarm actions rather than a threshold.
- Tune deterministic temporal alarm rules before introducing RL.
- Train and evaluate a small CPU policy on cached probability sequences.
- Compare fixed rules, supervised temporal models, and RL under one protocol.
- Preserve record boundaries, event identities, chronology, and refractory
  state in every method.

### 2.2 Out of scope for the first phase

- Updating EfficientNet, Qwen LoRA, E2/E3/E4 projections, or the classifier.
- Using `verl-agent` for binary EEG decisions.
- Claiming that RL improves AUROC or AUPRC.
- Adding a `wait N windows` action; `no_alarm` already means wait one step.
- Training a report-generation agent.
- Deep-RL sample acquisition for continual learning.
- Any deployment or clinical-safety claim.

### 2.3 Why `verl-agent-master` is retained but not imported

`verl-agent-master` is a token-generation RL framework. Its actor contract
expects an `AutoModelForCausalLM` or vision-language generator, response token
IDs, token log probabilities, and token-level advantages. The current EEG model
returns two class logits. Adapting the framework would require replacing its
model loader, rollout workers, response representation, reward manager, and
checkpoint format.

The first EEG alarm-policy implementation will therefore live beside the
vendored repository under `good/RL/eeg_alarm_policy/` and use a small,
purpose-built Gymnasium environment with a maintained RL implementation. The
vendored source is a reference for group-relative and multi-step RL research,
not a runtime dependency.

## 3. Frozen Base-Model Contract

The initial experiment uses the promoted same-protocol model:

- Architecture: E1+E2+E3+E4 with Qwen2.5-0.5B `visual_mean` pooling.
- STFT: `n_fft=128`, `win_length=128`, `hop_length=32`.
- Input step: one 4-second EEG window.
- Scheme C base-model split:
  - model training cases: `chb01-chb19`;
  - model validation cases: `chb20-chb21`;
  - existing final test cases: `chb22-chb23`;
  - `chb24` unused by the Scheme C leaderboard.
- Promoted result: test AUROC `0.9945`, AUPRC `0.5285`, and F1 `0.2489`.

The checkpoint path and SHA256 must be read from the authoritative S1 result
artifact rather than duplicated as an unchecked constant.

### 3.1 Identity and protocol warning

`chb01` and `chb21` are two cases from the same subject. They must be grouped
together in any new strict patient-level split. The historical Scheme C result
is retained for comparability, but the RL report must not describe `chb21` as an
independent unseen subject relative to a model trained on `chb01`.

### 3.2 Two evaluation tiers

#### Tier A: fast development benchmark

- Freeze the existing promoted checkpoint.
- Train the policy on natural timelines from model-training cases.
- Use `chb20` for policy/reward selection.
- Exclude `chb21` from policy selection because of its identity overlap with
  `chb01`.
- Run `chb22-chb23` only after the policy, reward coefficients, seed-selection
  rule, and all baselines are frozen.

This tier is inexpensive but its policy-training probabilities are in-sample
for the frozen base model. Results must be labeled as a development benchmark.

#### Tier B: strict confirmation

- Group `chb01` and `chb21` in the same partition.
- Generate out-of-fold base-model probabilities for every policy-training and
  policy-validation case.
- Fit enrollment statistics only from the permitted early normal windows of
  each case.
- Freeze the complete policy-development procedure before final evaluation.
- Keep the final held-out cases and their labels inaccessible to policy fitting,
  reward selection, early stopping, and debugging.

Tier B is required before making a cross-patient generalization claim.

## 4. Prediction Artifact Contract

The latest Paper Fold 4 run stores probability hashes but not the probability
arrays. Phase L0 must add a content-addressed prediction artifact. One artifact
contains these arrays in exact natural-timeline order:

| Field | Type | Meaning |
|---|---|---|
| `subject_id` | string array | CHB-MIT case ID |
| `probability` | float32 | Frozen model ictal probability |
| `label` | uint8 | Window label; inaccessible in deployment observations |
| `record_index` | int32 | Recording boundary identifier |
| `start_sample` | int64 | Window start in the source record |
| `event_index` | int32 | Ground-truth event ID for offline reward/evaluation |
| `sampling_frequency_hz` | scalar | Timeline sampling rate |
| `window_seconds` | scalar | Window duration |
| `stride_seconds` | scalar | Decision-step duration |

The sidecar JSON must include:

- schema version;
- checkpoint path, SHA256, model contract, and pooling mode;
- source manifest and cache hashes;
- subject order and row slices;
- per-array SHA256 values;
- probability count per subject;
- inference precision, device, and package versions;
- whether each case was in the base-model train, validation, or test partition.

The writer must use an atomic temporary file followed by `os.replace`. Loading
must reject reordered rows, duplicate keys, missing subjects, incompatible
window contracts, and checkpoint-hash mismatches.

## 5. Alarm Evaluation Contract

The existing `evaluate_target_timeline` converts probabilities to binary
decisions using a threshold and then applies k-of-n voting and a refractory
period. RL already emits decisions, so the evaluator must be separated into:

```python
evaluate_probability_policy(
    timeline,
    probabilities,
    threshold,
    alarm_config,
)

evaluate_alarm_actions(
    timeline,
    alarm_actions,
    *,
    action_scores=None,
)
```

Both paths must use exactly the same alarm-episode merging, record reset,
event-overlap matching, refractory semantics, normal-monitoring duration, and
latency definitions.

An equivalence test must show that explicit actions generated by the existing
threshold/vote/refractory path reproduce the current event metrics exactly.

### 5.1 Primary metrics

- Event sensitivity.
- False alarm episodes per normal monitoring hour.
- Mean and median detection latency.
- Detected, missed, and total event counts.
- Alarm episode count.

### 5.2 Secondary metrics

- Window precision, recall, specificity, and F1 for methods that expose a
  per-window action.
- Alarm duration and duplicate alarms per event.
- Per-patient metrics and pooled totals.
- Mean and worst-patient results.

AUROC and AUPRC are copied from the frozen base probabilities as invariance
checks. A changed AUROC/AUPRC indicates an artifact or evaluation bug.

## 6. Phase L0: Infrastructure

### L0.1 Package layout

Create:

```text
good/RL/eeg_alarm_policy/
  __init__.py
  artifacts.py
  evaluator.py
  features.py
  rules.py
  environment.py
  objectives.py
  splits.py
  train_supervised.py
  train_rl.py
  evaluate.py
  report.py
```

Tests remain under `good/RL/tests/` so this work can be developed, tested, and
moved without modifying the repository-level test suite.

### L0.2 Probability export

- Load the promoted S1 checkpoint and verify its SHA256.
- Run one deterministic inference pass over required natural timelines.
- Persist raw probabilities and timeline identity arrays.
- Reload the artifact and verify exact equality and hashes.
- Confirm that running policy experiments never imports or loads the EEG model.

### L0.3 Explicit-action evaluator

- Extract common alarm-episode logic without changing current behavior.
- Add explicit binary-action evaluation.
- Reset history and refractory state at every EDF record boundary.
- Treat the first alarm overlapping an event as a detection.
- Count alarms that overlap no event as false alarm episodes.
- Do not reward repeated alarms for an already detected event.

### L0 acceptance gate

- Existing evaluator tests remain green.
- Threshold-to-action equivalence is exact on synthetic and real timelines.
- Repeated artifact builds from identical inputs have identical hashes.
- No GPU is touched after prediction export.

## 7. Phase L1-A: Deterministic and Supervised Baselines

RL is not justified until it beats optimized non-RL policies.

### L1-A.1 Fixed-rule grid

Search only policy-training and validation timelines:

- threshold: quantiles derived from source/enrollment probabilities plus a
  fixed bounded grid;
- `vote_n`: `1, 2, 3, 4, 5, 8`;
- `vote_k`: every valid value from `1` to `vote_n`;
- refractory seconds: `0, 30, 60, 120, 300`;
- optional hysteresis control with separate on/off thresholds;
- optional exponential moving-average control.

The final test set must never be used to prune the grid.

### L1-A.2 Supervised temporal controls

Train two small controls on the same observation features:

1. Logistic regression.
2. Two-layer MLP with no more than 10,000 trainable parameters.

These controls determine whether recent probability history alone explains any
gain attributed to RL.

### L1-A.3 Selection objective

Report the full validation Pareto frontier. Select one operating point with a
predeclared objective:

```text
J = event_sensitivity
    - lambda_fa * false_alarms_per_hour
    - lambda_latency * normalized_mean_latency
```

The reward coefficients and normalization constants are chosen on development
data and frozen before test. In addition to scalar `J`, enforce a minimum event
sensitivity guardrail so a policy cannot improve by remaining silent.

### L1-A acceptance gate

- Best fixed rule and best supervised control are frozen and saved.
- Their decisions can be reproduced from the prediction artifact alone.
- Selection uses no final-test labels.

## 8. Phase L1-B: RL Alarm Policy

### 8.1 Environment

One environment episode is one EDF recording, not an entire concatenated
patient timeline. Patient-level metrics are accumulated across recordings.
Resetting per recording matches the current evaluator and avoids carrying
history across recording gaps.

At each 4-second step the environment reveals only information available at
that time. Labels and event IDs remain private to reward and evaluation.

### 8.2 Observation

Initial observation dimension is 14:

- latest eight frozen probabilities: 8 values;
- enrollment probability median, scaled MAD, and 95th percentile: 3 values;
- clipped seconds since the last accepted alarm: 1 value;
- clipped refractory time remaining: 1 value;
- record-start indicator: 1 value.

Enrollment statistics use only the permitted earliest normal calibration
windows for that patient. Zero-MAD handling must be explicit and tested. No
future probability, label, event boundary, patient ID, or file identity enters
the observation.

An ablation may add causal trajectory summaries such as slope and local
variance, but only after the 14-dimensional policy is complete.

### 8.3 Action

```text
0 = no_alarm
1 = alarm
```

The environment decides whether an emitted alarm is accepted under the chosen
refractory semantics. A separate multi-step wait action is deferred.

### 8.4 Reward

Use causal shaped rewards whose episode sum corresponds to the declared event
objective:

- first accepted alarm that overlaps an undetected event: positive hit reward;
- accepted alarm outside all events: false-alarm penalty;
- event ending without an accepted alarm: miss penalty;
- each seizure step before first detection: small latency penalty;
- repeated alarm for an already detected event: zero or a small duplicate
  penalty.

Reward components and raw counts must be logged separately. Reward clipping is
allowed only if the unclipped episode return is also retained. Always-alarm and
never-alarm policies are mandatory controls against reward-design failures.

### 8.5 Algorithms

Run in this order:

1. Random policy sanity control.
2. Discretized tabular Q-learning diagnostic.
3. Small PPO policy with two hidden layers of 32 units.
4. DQN only if PPO is unstable or materially worse than the tabular diagnostic.

Use a separate `eeg-rl` conda environment. Do not install `verl-agent`
dependencies into `qwen35-eeg`. CPU training is the default because all EEG
probabilities are cached.

### 8.6 Seeds and selection

- Use at least five fixed seeds for the RL policy.
- Select algorithm settings using validation aggregate `J` only.
- Freeze the policy-construction seed rule before final test.
- Report median, interquartile range, every seed, and every patient.
- Do not choose the best test seed.

### L1-B promotion gate

RL is retained only if it:

1. beats the best fixed-rule grid and supervised temporal control on validation
   `J`;
2. satisfies the event-sensitivity guardrail;
3. improves at least one of false alarms/hour or latency without a material
   regression in the other;
4. remains finite and deterministic under fixed inference seeds;
5. shows no AUROC/AUPRC change in the frozen base probabilities.

Failure to beat the tuned rule is a valid result: the project should then keep
the deterministic alarm policy and stop L1 development.

## 9. Phase L2: Event-Level Policy Optimization of Model Outputs

L2 is blocked until L1 is complete. It asks whether event-level policy gradients
can improve the score generator itself.

### 9.1 Cheapest viable experiment

- Cache the 896-dimensional fused representation from the frozen model.
- Train only a copied `Linear(896, 2)` classifier as a Bernoulli policy.
- Keep the base classifier as an anchor/reference policy.
- Sample multiple action rollouts from the same recording.
- Use group-relative centered returns as a critic-free baseline.
- Penalize divergence from the base classifier and report calibration drift.

This is conceptually related to GRPO/GiGPO, but it will be implemented for
binary actions rather than forced through `verl-agent` token-generation APIs.

### 9.2 Required controls

- Weighted cross-entropy.
- Focal loss.
- Pairwise ranking/AUC surrogate.
- Differentiable AP/AUPRC surrogate where numerically stable.
- The frozen classifier plus the promoted L1 decision policy.

L2 proceeds to full EEG forward passes only if classifier-only policy training
beats these controls. EfficientNet, Qwen LoRA, and E2/E3/E4 remain frozen during
the first L2 experiment.

## 10. Phase L3: Continual-Learning Label-Budget Policy

L3 is a later research track. Its action chooses whether to request a label,
store a sample, or trigger a classifier-head update under a fixed budget.

Initial methods:

- random sampling;
- uncertainty sampling;
- diversity sampling;
- uncertainty plus diversity heuristic;
- linear contextual bandit.

Do not start with deep RL. CHB-MIT has too few independent subjects for a large
cross-patient meta-policy, and offline sample selection requires expensive
counterfactual retraining. Evaluation must reuse the existing ordered
experience, replay, selection, and future-holdout boundaries.

## 11. Test Plan

### Unit tests

- Artifact schema, hashes, atomic write, and checkpoint identity.
- Observation causality and fixed shape.
- Record-boundary resets for probability history and refractory state.
- Alarm merging and refractory behavior.
- First-hit, duplicate-hit, false-alarm, miss, and latency rewards.
- Zero-event and zero-normal-duration records.
- Enrollment MAD equal to zero.
- Deterministic reset and step behavior for fixed seeds.

### Integration tests

- Existing threshold evaluation equals explicit-action evaluation.
- A prediction artifact can run every baseline and RL environment without
  importing Torch model code.
- Labels and event IDs never appear in observations.
- Final-test loaders fail while the run is in development mode.
- `chb01` and `chb21` cannot enter different strict subject partitions.
- Frozen probability AUROC/AUPRC hashes remain unchanged across policies.

### Experiment audit

- Save the exact split, reward, state contract, seeds, package versions, and
  policy checkpoint hash.
- Save per-record action arrays and accepted-alarm arrays.
- Save development selection tables before creating the final-test result.
- Evaluate the final test exactly once for each frozen method family.

## 12. Deliverables

Phase L1 produces:

- `good/RL/eeg_alarm_policy/` implementation;
- content-addressed prediction artifacts under `artifacts/chbmit/eeg_rl/`;
- deterministic-rule search table;
- supervised temporal-control results;
- RL seed and learning-curve artifacts;
- per-patient event comparison plots;
- `good/RL/EEG_RL_ALARM_POLICY_RESULTS.md`;
- machine-readable result JSON with artifact hashes.

Required result figures:

1. Event sensitivity versus false alarms/hour Pareto plot.
2. Detection-latency versus false alarms/hour plot.
3. Per-patient comparison for fixed rule, supervised MLP, and RL.
4. RL training return with seed dispersion.
5. One representative natural-timeline plot showing probabilities, seizure
   intervals, fixed-rule alarms, and RL alarms.

## 13. Execution Order

1. L0 prediction export and artifact validation.
2. L0 explicit-action evaluator and equivalence tests.
3. L1-A fixed-rule grid.
4. L1-A logistic-regression and small-MLP controls.
5. Freeze the objective, guardrail, and test procedure.
6. L1-B tabular diagnostic and PPO.
7. Final held-out L1 evaluation and report.
8. Decide whether evidence justifies strict Tier B confirmation.
9. Consider L2 only after L1 conclusions are stable.
10. Consider L3 only after the continual-learning protocol is frozen.

## 14. Stop Conditions

Stop RL development and retain deterministic rules when any of the following is
true:

- the tuned fixed rule matches or beats RL;
- gains disappear across policy seeds or patients;
- reward coefficients determine the conclusion more strongly than the policy;
- RL reduces false alarms mainly by missing events;
- the result depends on in-sample base-model probabilities and fails Tier B;
- the policy cannot be reproduced from saved artifacts.

This keeps the research question falsifiable: RL is adopted only when sequential
decision learning adds measurable value beyond tuning the existing alarm logic.
