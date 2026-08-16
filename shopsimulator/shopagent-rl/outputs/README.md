# Versioned Experiment Outputs

Active, reportable assets are grouped by training stage and version.  Within
each version, `model/`, `evaluation/`, and `logs/` keep the checkpoint,
Final-200 result, and corresponding run records together.

```text
outputs/
├── base/v1/evaluation/                 # Base Final-200 baseline
├── sft/
│   ├── v1/
│   │   ├── model/training_output/       # SFT checkpoints + LoRA adapter
│   │   ├── evaluation/                  # Final-200 result and per-task traces
│   │   └── logs/                        # SFT training and evaluation logs
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

The multi-GB `actor/*.pt` FSDP resume state in GRPO v1 stays local and is
ignored by Git. The portable compressed LoRA adapter in `artifact/adapter/` is
versioned and can be restored with `scripts/restore_large_artifacts.sh`.

Historical Hydra launch folders remain under `outputs/2026-08-12/`; older
experiments, including the SFT-90 checkpoint at `../archive/sft_legacy/`,
remain under `../archive/`.
