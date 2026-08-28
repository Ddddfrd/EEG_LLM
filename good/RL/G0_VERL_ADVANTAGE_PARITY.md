# G0 verl-agent Advantage Parity

All independent binary-policy advantage calculations match the unmodified verl-agent reference functions at absolute tolerance `1e-6`.

| Case | Shape | Max absolute error | Passed |
|---|---:|---:|:---:|
| grpo_std=True_cross_steps=True | 8x3 | 0.000e+00 | yes |
| grpo_std=False_cross_steps=True | 8x3 | 0.000e+00 | yes |
| grpo_std=True_cross_steps=False | 8x3 | 0.000e+00 | yes |
| grpo_std=False_cross_steps=False | 8x3 | 0.000e+00 | yes |
| rloo_cross_steps=True | 8x3 | 1.192e-07 | yes |
| rloo_cross_steps=False | 8x3 | 0.000e+00 | yes |
| gigpo_mean_norm | 8x3 | 0.000e+00 | yes |
| gigpo_mean_std_norm | 8x3 | 0.000e+00 | yes |
| grpo_singleton | 1x2 | 0.000e+00 | yes |
| rloo_singleton | 1x2 | 0.000e+00 | yes |
| gigpo_singleton | 1x2 | 0.000e+00 | yes |

## Verified semantics

- GRPO outcome centering with optional sample-standard-deviation scaling.
- RLOO leave-one-out baseline with and without repeated-step deduplication.
- GiGPO episode advantage plus exact-anchor step-level advantage.
- Response masking, zero-variance groups, repeated trajectory IDs, and singleton groups.

## Important reference behavior

The reference sums token-level rewards before applying the response mask. Upstream reward tensors must therefore already be zero outside valid positions. For a singleton episode group, GRPO and RLOO preserve the raw episode score because the reference uses mean `0` and standard deviation `1`; singleton GiGPO step advantage is zero. The EEG adapter will preserve these semantics unless a separately named ablation intentionally changes them.

Result SHA256: `3c649e4207466d320f181b7e78fc93ce5f42866f2864d9472921b6224051bac2`
