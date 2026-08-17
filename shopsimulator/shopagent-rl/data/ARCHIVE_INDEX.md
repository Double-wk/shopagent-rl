# Data Organization

Current dataset definitions and experiment-ready counts live in [`../DATA.md`](../DATA.md).
This file only explains storage/retention boundaries; historical directories are not rewritten
when the active Certified pipeline changes.

## Canonical inputs

- `sft_train.jsonl`: current 3,793-record SFT training set.
- `trajectories_raw/gpt-5.6-terra/trajectories_raw.jsonl`: complete teacher collection.
- `trajectories_raw/gpt-5.6-terra_trainonly/trajectories_raw.jsonl`: train-only view consumed by `scripts/build_sft_data.sh`.
- `trajectories_sft/`: validated per-task trajectories used to build the current SFT set.
- `sft_certified_corrective_train.jsonl`: natural-format paired corrective records.
- `sft_train_certified_corrective_mix.jsonl`: 10,057-record v4 corrective SFT mixture.
- `grpo_certified_natural_train.parquet`: no-summary paired/environment GRPO mixture gated on v4 evaluation.

## Retained historical data

- `trajectories_raw_old/`: earlier teacher collections and SFT snapshots. These files are retained for reproducibility and are not current training inputs.
