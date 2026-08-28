# G3: GiGPO Step-Level Credit Assignment — Findings

Status: **completed, not promoted** (2026-08-28). Machine result:
`artifacts/chbmit/eeg_rl/g3_gigpo/g3_gigpo_14a8fdd69696.json`.

## Setup (frozen before the run)

- Identical to G1 (chb20 train / chb21 select / chb22-23 untouched, same
  `AlarmRewardConfig`, same sparse init `init_logit_bias=-3`, same PPO update,
  5 seeds 11-55, median chb21 J selection) — only the advantage changed.
- Advantage = episode + 1.0 × step (GiGPO Eq. 8):
  - episode: record return standardized across same-record rollouts
    (exactly G1's `grpo_outcome_advantage`, std-normalized);
  - step: return-to-go (γ=1, Eq. 5) standardized within step groups
    (verl-agent default mode `mean_std_norm`).
- Step anchor = `(record_group, row)`, **an intentional deviation** from
  verl-agent's exact-observation hash. Reason: our 14-dim observation embeds
  this rollout's own alarm history (seconds-since-alarm, refractory
  remaining), so exact matching would fragment precisely at the divergent
  alarm rows where the counterfactual comparison is needed. The `(record,
  row)` anchor keeps every step group at rollouts_per_group members.
  Documented in the result payload (`protocol.anchor_deviation`).
- Step-level math reuses the G0 parity-verified ports (`group_advantages.py`);
  the PPO update loop was factored into `grpo_training.ppo_update_epochs`
  (behavior-preserving; G1 determinism tests still green).

## Finding 1 — step-level credit fixes P9 on the training subject

The step signal is strong and concentrated where it matters:

| metric (8 epochs) | G1 (episode only) | G3 (episode + step) |
|---|---|---|
| mean \|step adv\| | — (uniform broadcast) | 0.74-0.77 every epoch |
| active step groups | — | 96-98% (group std > 0) |
| chb20 ictal/normal logit separation (seed 11) | 0.15 | **0.95** |

chb20 ictal mean logit rose from −4.33 (G1) to −3.20 while normal fell to
−4.15: the hit/miss outcome now reaches the alarm rows through the
return-to-go instead of being diluted over ~900 steps. Credit dilution
(P9) is fixed as a training-signal problem.

## Finding 2 — the bottleneck moved to cross-subject generalization

On the selection subject the same policy reverses: chb21 separation is
**−0.27** (ictal −4.41 vs normal −4.14), no logit crosses the 0 readout
threshold anywhere, deterministic readout gives 0 alarms, J = 0 on all
seeds. The mechanism is a distribution shift the single-subject policy
cannot see past:

| | chb20 (train) | chb21 (select) |
|---|---|---|
| enrollment median | 0.1023 | 0.0072 (~14× lower) |
| enrollment q95 | 0.6035 | 0.1230 |
| ictal mean prob | 0.6366 | 0.6375 (similar!) |

The 8 history features are raw probabilities. A policy that has only seen
chb20 reads chb21's typical background (0.007-0.12) as "deeply normal" and
never learned scale invariance, because the enrollment features give it no
incentive to generalize beyond the one distribution it trained on. R2 was also
trained on chb20, not pooled across chb01-19, but its supervised per-window
labels and temporal features transfer to chb21 at J ≈ 0.988. The comparison
therefore shows that useful transfer is possible; it does not establish that a
pooled warm-start model already exists.

## Formal result (5 seeds, frozen defaults: weight 1.0, γ 1.0, mean_std_norm)

| seed | J (chb21) | entropy ep1→ep8 | return ep1→ep8 | \|step adv\| ep1→ep8 |
|---|---|---|---|---|
| 11 | 0.0 | 0.204 → 0.098 | −0.393 → −0.359 | 0.746 → 0.744 |
| 22 | 0.0 | 0.171 → 0.071 | −0.444 → −0.333 | 0.752 → 0.759 |
| 33 (median, selected) | 0.0 | 0.168 → 0.086 | −0.399 → −0.343 | 0.740 → 0.765 |
| 44 | 0.0 | 0.151 → 0.049 | −0.395 → −0.331 | 0.740 → 0.758 |
| 55 | 0.0 | 0.185 → 0.089 | −0.372 → −0.335 | 0.735 → 0.751 |

`promoted: false` (bar: robust 0.9904 / supervised 0.9882). Low seed
variance — all seeds converge to the same silent-on-chb21 outcome.

## Conclusions

1. G3 did what it was designed to do: step-level grouping restores dense,
   alarm-localized credit (P9 closed). The failure mode changed from
   "no learning signal" to "learned signal does not transfer across
   subjects".
2. Single-subject RL training is now the demonstrated bottleneck. Before G4,
   a genuine pooled chb01-19 MLP and corresponding out-of-fold probability
   artifacts must be built; the existing R2 MLP is chb20-only. G4 may then
   test that warm start with a KL constraint so step-local RL updates do not
   destroy the supervised behavior. G2 (RLOO, episode-level) stays skipped.
3. If G4 also fails to beat the supervised MLP on chb21, the RL route stops
   per the frozen protocol and the MLP remains the candidate.

Verification: 62 tests passing (6 new), ruff clean, CPU-only throughout
(L0 contract). Wall clock ≈ 12 min for 5 seeds.
