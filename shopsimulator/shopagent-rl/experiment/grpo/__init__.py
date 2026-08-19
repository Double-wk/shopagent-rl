# GRPO integration with veRL. After the 2026-08-08 downgrade (transformers
# 5.14.1 -> 4.57.6, huggingface_hub -> 0.36.2, --no-deps), shopsim now imports the
# full veRL chain, so this runs in shopsim (the single env for SFT+GRPO+eval).
# opd-rocm (also transformers 4.57.6) works identically if ever needed.
