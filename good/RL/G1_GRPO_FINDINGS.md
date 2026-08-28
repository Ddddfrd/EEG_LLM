# G1: Record-Grouped GRPO — Findings

Status: **completed, not promoted** (2026-08-28). Machine result:
`artifacts/chbmit/eeg_rl/g1_grpo/g1_grpo_9439626b912e.json`.

## Setup (frozen before the run)

- Training: chb20 (29 EDF records, 24,828 windows). Selection read: chb21 only.
  chb22-23 untouched.
- Environment: one record = one episode; `AlarmRewardConfig` identical to phase
  R3 (hit +1, miss −1, FA −0.02, latency −0.001/min, duplicate −0.01,
  refractory 300 s).
- Advantage: record return standardized across same-record rollouts (GRPO,
  `grpo_outcome_advantage`, parity-verified in G0 to 1.2e-7).
- Deployment readout (pre-declared): deterministic alarm iff logit ≥ 0 (p ≥ 0.5).
- Config: 8 epochs, 8 rollouts/group, 2 update epochs, minibatch 2048,
  lr 3e-4, clip 0.2, entropy 0.01, seeds (11, 22, 33, 44, 55); selection = median
  chb21 J seed.

## Finding 1 — dense random init is a GRPO dead zone

At the default random init the actor alarms at p ≈ 0.5-0.59 everywhere. Under
the 300 s refractory this saturates: every rollout of a record accepts the same
alarm schedule (first alarm after each refractory gap), so all within-group
returns came out bit-identical:

| init alarm p | active groups (variance > 0) |
|---|---|
| 0.53-0.59 (random init) | **0 / 29** |
| 0.27 | 3 / 29 |
| 0.12 | 29 / 29 |
| 0.047 | 29 / 29 (highest mean group std 0.16) |
| 0.018 / 0.007 | 29 / 29 |

Fix: `GRPOConfig.init_logit_bias = -3.0` (sparse alarm start) restores the
group-relative signal. Recorded in the result payload and covered by
`tests/test_grpo_training.py`.

## Finding 2 — record-level outcome credit is diluted ~60:1

With signal restored, training dynamics are healthy but the policy drifts
silent: entropy falls every epoch (e.g. seed 11: 0.217 → 0.145), mean trajectory
return barely moves (−0.393 → −0.367), and after 8 epochs the ictal/normal logit
separation is only **0.15** (ictal −4.33 vs normal −4.48) with no logit crossing
0 anywhere. The mechanism: an event record has ~900 steps of which ~14 are
ictal; the record-level outcome advantage (±1 hit/miss) is broadcast uniformly
over all steps, so the alarm-relevant credit is ~1.6 % of the gradient signal,
while the "staying silent saves 0.02 FA" signal is consistent on all 23 normal
records. Silence wins. This is a credit-assignment limitation, not optimization
noise — and it is exactly what GiGPO's step-level grouping (G3) targets.

## Formal result (5 seeds)

| seed | J (chb21) | sensitivity | FA/h | entropy ep1→ep8 |
|---|---|---|---|---|
| 11 | 0.0 | 0.0 | 0.0 | 0.217 → 0.145 |
| 22 | 0.0 | 0.0 | 0.0 | 0.178 → 0.110 |
| 33 (median, selected) | 0.0 | 0.0 | 0.0 | 0.185 → 0.103 |
| 44 | 0.0 | 0.0 | 0.0 | 0.160 → 0.074 |
| 55 | 0.0 | 0.0 | 0.0 | 0.192 → 0.105 |

Development comparators: robust fixed rule J = 0.9904, logistic regression
J = 0.9882, R3 PPO selected seed J = 0.2374. `promoted: false` — record-level
GRPO is far below every baseline, with low seed variance (all seeds identical:
uniform silent policy).

## Conclusions

1. G1 is a clean negative: episode-level group advantage alone cannot train
   this task; the failure is attributable to credit dilution (measured), not to
   the advantage math (G0 parity) or exploration (fixed and verified active).
2. The correct next step per the G-plan is G3 (step-level grouping) — or
   treating G2 (RLOO, also episode-level) as a low-information control.
3. Per the frozen protocol, GRPO does not enter any final-test evaluation.

Verification: 56 tests passing, ruff clean. CPU-only throughout (no GPU touched,
per the L0 contract). Wall clock ≈ 9 min for 5 seeds.
