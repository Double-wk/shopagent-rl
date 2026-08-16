# Data Organization

## Canonical inputs

- `sft_train.jsonl`: current 3,793-record SFT training set.
- `trajectories_raw/gpt-5.6-terra/trajectories_raw.jsonl`: complete teacher collection.
- `trajectories_raw/gpt-5.6-terra_trainonly/trajectories_raw.jsonl`: train-only view consumed by `scripts/build_sft_data.sh`.
- `trajectories_sft/`: validated per-task trajectories used to build the current SFT set.

## Retained historical data

- `trajectories_raw_old/`: earlier teacher collections and SFT snapshots. These files are retained for reproducibility and are not current training inputs.
