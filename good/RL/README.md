# EEG Alarm-Policy RL

This directory contains an isolated research package for temporal EEG alarm
decisions. The core `eeg_alarm_policy` package does not import retained
implementations under `good/` or `ai/`; it consumes versioned probability
artifacts only. Read-only adapters under `integrations/` are the explicit
boundary that loads a frozen model and exports those artifacts.

`verl-agent-master/` is retained as external reference source. It is not a
dependency of `eeg_alarm_policy` because its rollout contract is built for
generated language-model tokens rather than binary alarm actions.

Development plan:
[`EEG_RL_ALARM_POLICY_PLAN.md`](EEG_RL_ALARM_POLICY_PLAN.md).

Completed R1-R4 report:
[`EEG_RL_ALARM_POLICY_RESULTS.md`](EEG_RL_ALARM_POLICY_RESULTS.md).

VERL-inspired G0 parity report:
[`G0_VERL_ADVANTAGE_PARITY.md`](G0_VERL_ADVANTAGE_PARITY.md).

Grouped-policy findings:
[`G1_GRPO_FINDINGS.md`](G1_GRPO_FINDINGS.md) and
[`G3_GIGPO_FINDINGS.md`](G3_GIGPO_FINDINGS.md).

## Current result

R1-R4 are complete. The held-out chb22-chb23 probabilities were exported only
after the protocol freeze and reproduce the base-model AUROC `0.9945` and AUPRC
`0.5285` exactly.

- Frozen robust rule: 9/10 events, `0.471` false alarms/hour, but only 2/3
  chb22 events and therefore fails the patient guardrail.
- Frozen 32x32 MLP: 10/10 events and `0.837` false alarms/hour; this is the
  strongest final comparator, but requires confirmation on a new untouched
  cohort before promotion.
- Median-seed PPO failed its validation promotion gate and was not evaluated on
  chb22-chb23. The current evidence does not justify RL over simpler controls.

The next research step is Tier B confirmation with grouped `chb01/chb21` and
out-of-fold policy-training probabilities. L2 event-level model optimization is
paused until the compact MLP result is independently confirmed.

## VERL-inspired binary policy track

G0 is complete. Independent GRPO, RLOO, and exact-anchor GiGPO advantage
functions were compared directly with the unmodified vendored `verl-agent`
sources. All 11 cases passed at absolute tolerance `1e-6`; the maximum observed
error was `1.192e-7`.

G1 and G3 are complete. Record-level GRPO restored exploration with a sparse
alarm initialization but converged to silence on chb21. GiGPO then restored
step-local credit on chb20, but the learned separation did not transfer to
chb21. Neither method passed the development promotion gate, and neither was
evaluated on chb22-chb23.

Run the isolated tests from this directory:

```powershell
conda run --no-capture-output -n qwen35-eeg python -m pytest
```

Run static checks:

```powershell
conda run --no-capture-output -n qwen35-eeg python -m ruff check `
  eeg_alarm_policy tests
```

Rebuild the saved report and figures from immutable R1-R4 artifacts:

```powershell
conda run --no-capture-output -n qwen35-eeg python -m `
  eeg_alarm_policy.build_report `
  --r1 artifacts\chbmit\eeg_rl\rule_search_robust\rule_search_2723796193be.json `
  --r2 artifacts\chbmit\eeg_rl\supervised_controls\supervised_controls_34d4ab7abb92.json `
  --r3 artifacts\chbmit\eeg_rl\phase_r3\phase_r3_bdf87d987ba3.json `
  --r4 artifacts\chbmit\eeg_rl\final_evaluation\r4_final_evaluation_0632a4cc02d5.json `
  --test-artifact-dir artifacts\chbmit\eeg_rl\final_test_predictions
```

Re-run G0 mathematical parity verification:

```powershell
conda run --no-capture-output -n qwen35-eeg python -m `
  eeg_alarm_policy.verify_verl_advantages
```
