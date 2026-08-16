# GRPO env16 global step 200

Completed ShopSimulator GRPO run using the env16 service profile.

- Base model: `Qwen/Qwen3-1.7B-Base`
- Initial adapter: `outputs/sft/v1/model/training_output/lora_adapter`
- Training recipe: `TRAIN_BATCH=4`, `ROLLOUT_N=4`, `PPO_MINI_BATCH=4`,
  `GPU_MEM_UTIL=0.25`, `LR=1e-5`
- Final state: 200/200 steps completed on 2026-08-12 21:43 UTC.
- Resume-capable FSDP checkpoint: `outputs/grpo/v1/model/checkpoint_step_200/`
- Portable inference/evaluation adapter: `adapter/`

`adapter/adapter_model.safetensors` has SHA-256
`0d6162d596b92a307a5811f90830e3905961da950ab8308c7838c24bfc540d53`.
It is also stored as `adapter_model.safetensors.gz` for repository transport;
run `bash scripts/restore_large_artifacts.sh` after a fresh clone to restore it.

`provenance/` contains the complete training log, the exact resume script,
FSDP/LoRA metadata, dataloader state, and the LoRA-export log. The final run
contains a small number of ShopSim rollout errors (`can only join an iterable`)
which received zero reward; they did not prevent the run from reaching step 200.

`provenance/pre50_runs/` preserves the five completed/attempted GRPO recovery
logs leading up to step 50. The final uninterrupted step-50-to-200 run is kept
as `provenance/training.log`.
