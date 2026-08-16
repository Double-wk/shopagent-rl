# GRPO global step 20

This directory contains the latest completed ShopSim GRPO checkpoint artifact.

- Base model: `Qwen/Qwen3-1.7B-Base`
- Source checkpoint: `/overlay/shop_A_grpo_outputs/grpo_50_batch6_mini3/global_step_20`
- Training run: `TRAIN_BATCH=6`, `ROLLOUT_N=4`, `PPO_MINI_BATCH=3`
- The run resumed at step 10 and reached step 23 before a HIP out-of-memory
  failure. Therefore `global_step_20` is the latest complete checkpoint.
- `adapter/` is the extracted LoRA adapter and can be passed to
  `scripts/run_eval.sh --adapter` together with the base model.
- `provenance/` contains the LoRA/FSDP metadata, rollout state, and the full
  training log. The original 8.3GB FSDP model shard remains on the overlay;
  the portable LoRA adapter is the uploaded model weight.

The evaluation artifacts will be added alongside this directory after the
20-turn SFT and GRPO comparisons finish.
