# RelGRPO v2 residual smoke step 10 adapter

This is a portable PEFT adapter exported from the veRL/FSDP checkpoint. The
weights are gzip-compressed and split below GitHub's 100 MB file limit.

Restore in this directory:

```bash
cat adapter_model.safetensors.gz.part* | gzip -d > adapter_model.safetensors
sha256sum -c adapter_model.safetensors.sha256
```

The base model is the cached Qwen3-1.7B-Base snapshot recorded in the parent
experiment README. Raw FSDP/optimizer checkpoints remain on `/overlay`.
