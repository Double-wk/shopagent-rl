# Archived Hydra Run Snapshots

Hydra creates `outputs/YYYY-MM-DD/HH-MM-SS/` for every `main_ppo` launch. Each
snapshot contains the resolved configuration, command-line overrides, and an
empty `main_ppo.log`; operational logs are kept under `run/` instead.

- `2026-08-10/`: historical GRPO configuration trials.
- `2026-08-12/`: ROCm smoke tests, action/turn-boundary tests, and the 50-step
  diagnostic run started at `10-10-44`, which ended with HIP OOM after step 13.

These snapshots are for configuration audit only and are not active outputs.
