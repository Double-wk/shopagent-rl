# Versioned Experiment Outputs

Active, reportable assets are grouped by training stage and version.  Within
each version, `model/`, `evaluation/`, and `logs/` keep the checkpoint,
Final-200 result, and corresponding run records together.

Status synchronized on 2026-08-17. Generated checkpoint README files are model artifacts and
are not the source of truth; use this file and the repository root `README.md` for current claims.

```text
outputs/
├── base/v1/evaluation/                 # Base Final-200 baseline
├── sft/
│   ├── v1/
│   │   ├── model/training_output/       # SFT checkpoints + LoRA adapter
│   │   ├── evaluation/                  # Final-200 result and per-task traces
│   │   └── logs/                        # SFT training and evaluation logs
│   ├── v2_paired/                       # paired-option SFT + Final-200/counterfactual reports
│   ├── v3_certified/                    # failed summary-shortcut ablation (retained for audit)
│   └── v4_certified_corrective/         # running corrective adapter/evaluation destination
├── counterfactual/                      # atomic probes and metrics, including heldout-v2 diagnostics
└── grpo/v1/
    ├── model/checkpoint_step_200/       # env16 final FSDP checkpoint + LoRA
    ├── evaluation/                      # env16 step-200 Final-200 result
    └── logs/                            # training chain and Final-200 log
```

## Current comparable Final-200 reports

All use the 10-turn, 512-token protocol.

| Stage | Location | strict success | `r_hard` | completion | product rate |
|---|---|---:|---:|---:|---:|
| Base v1 | `base/v1/evaluation/` | 0% | 0 | 0% | 0% |
| SFT v1 | `sft/v1/evaluation/` | 17.0% | 0.2010 | 39.5% | 25.5% |
| GRPO v1 (env16, step 200) | `grpo/v1/evaluation/` | 8.5% | 0.1183 | 32.5% | 19.0% |
| SFT v2 Paired | `sft/v2_paired/evaluation/` | 25.0% | 0.278 | 52.0% | 27.5% |
| GRPO Paired-C1-hard | `grpo/paired_c1_hard/evaluation/` | **40.0%** | **0.425** | **90.0%** | **45.0%** |

## Current counterfactual status

- Paired-C1-hard option-swap paired robust: **73.1%**; price counterfactual accuracy: **0%**.
- Certified SFT v3 heldout-v2: option cf **82%**, natural price cf **0%**, price commit persistence **94.27%**.
- Restoring the v3-only budget summary produces price cf **100%** but original accuracy **32.55%**;
  this is a shortcut diagnostic, not a reportable natural-input result.
- v4 corrective output remains pending. Do not place a v4 number here until the detached evaluation
  chain finishes and writes the heldout-v2 and gated Final-200 metrics.

The multi-GB `actor/*.pt` FSDP resume state in GRPO v1 stays local and is
ignored by Git. The portable compressed LoRA adapter in `artifact/adapter/` is
versioned and can be restored with `scripts/restore_large_artifacts.sh`.

Historical Hydra launch folders remain under `outputs/2026-08-12/`; older
experiments, including the SFT-90 checkpoint at `../archive/sft_legacy/`,
remain under `../archive/`.
