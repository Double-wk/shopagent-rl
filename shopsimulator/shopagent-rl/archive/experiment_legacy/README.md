# Experiment Archive

This directory contains historical experiment artifacts that are retained for reproducibility.

## Layout

- `2026-08-09/results/`: previous checkpoints and evaluation attempts. These are not the current SFT or evaluation outputs.
- `2026-08-09/logs/`: legacy training logs moved out of `experiment/outputs/`.
- `2026-08-12/results/`: the historical SFT90 Final-200 report, moved from
  `outputs/` because it used the older 10 turn × 256 token protocol.

Current artifacts remain in `experiment/outputs/`:

- `sft_new3793/`: active SFT output directory.
- `eval_base_final10*`: current Base 10-step report and per-task traces.

The SFT90 report's original adapter weight is unavailable (only
metadata/tokenizer state remains under `archive/sft_legacy/checkpoint_90/checkpoint-90/`),
so it is retained as an audit record only and must not be used as a reproducible
or active baseline.
