# Constraint-Faithful RelGRPO v1

This directory records the first matched smoke comparison for the relational
policy-optimization prototype. Both runs start from the same certified SFT v4
adapter and use the same 800-row pair-blocked training parquet, batch size 4,
rollout group size 4, learning rate `1e-5`, data order, and 10 training steps.
The only intended training difference is the relational-advantage branch.

## Heldout result

Evaluation uses `data/counterfactual/heldout_atomic_pairs_v2.jsonl`, 534 pairs,
temperature 0, and the local cached Qwen3-1.7B base snapshot. `PRA` here is the
strict paired robust accuracy (both sides of a pair correct).

| initialization / method | original | counterfactual | PRA | certification | persistence error |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT v4 corrective | 93.82% | 78.09% | 73.03% | 77.84% | 18.73% |
| Independent CF-GRPO, step 10 | 87.64% | 87.83% | **76.97%** | 87.82% | 8.80% |
| Relational-advantage prototype, step 10 | 84.46% | 88.01% | 74.34% | 88.03% | 8.24% |
| Relational-residual v2, step 10 | 86.14% | 88.20% | 75.84% | 88.04% | 8.24% |

The relational branch is operational and improves counterfactual accuracy over
SFT, but this smoke run does **not** establish superiority over the matched
independent baseline. Its lower original-side accuracy is consistent with the
current prototype utility applying a shared negative signal when only the
counterfactual side fails. Do not use this result as evidence for the final
method claim; redesign the side/residual utility before longer training.

The v2 asymmetric residual fixes the clear v1 failure mode and recovers most of
the gap (75.84% vs 76.97% matched Independent), but it still does not beat the
Independent smoke at this sample size. The direction is therefore technically
viable but not yet paper-ready: the next experiment should use intent-level
relation certificates and a strictly matched multi-seed/longer-step comparison,
not claim a win from these smoke runs.

## Reproduction artifacts

- `smoke_step_10/lora_adapter/` — relational prototype adapter.
- `smoke_step_10/evaluation/counterfactual_heldout_v2_metrics.json` — relational metrics.
- `independent_smoke_step_10/lora_adapter/` — matched independent adapter.
- `independent_smoke_step_10/evaluation/counterfactual_heldout_v2_metrics.json` — independent metrics.
- `residual_v2_smoke_step_10/lora_adapter/` — asymmetric residual v2 adapter.
- `residual_v2_smoke_step_10/evaluation/counterfactual_heldout_v2_metrics.json` — v2 metrics.

Raw FSDP checkpoints and optimizer states remain under `/overlay/shopagent_rl_artifacts/`
and are intentionally not part of the reproducibility artifact.
