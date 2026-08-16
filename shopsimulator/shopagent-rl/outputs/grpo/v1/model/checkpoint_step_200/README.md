# GRPO env16 global step 200 checkpoint

This is the final resume-capable FSDP checkpoint from the completed 200-step
ShopSimulator GRPO run. Continue training with:

```bash
TOTAL_STEPS=<new-total> bash scripts/run_grpo.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/workspace/shopsimulator/shopagent-rl/outputs/grpo/v1/model/checkpoint_step_200
```

For evaluation or inference, use `lora_adapter/` instead of loading the FSDP
checkpoint directly. Its provenance and compressed portable copy are in
`artifact/`.
