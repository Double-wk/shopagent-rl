# GRPO v2a — Signal-quality validation

This version starts from SFT v1, not GRPO v1.  It is a 50-step diagnostic run
before any long GRPO training.

| Setting | v1 | v2a |
|---|---:|---:|
| distinct tasks per step | 4 | 2 |
| rollouts per task | 4 | 8 |
| total rollouts per step | 16 | 16 |
| temperature | 0.70 | 0.85 |
| observation cap | 1800 chars | 1400 chars |

The v2 service fixes concurrent same-task rollouts sharing a SimServer browser
session.  In v1 this caused `KeyError: None` when one rollout overwrote another
rollout's selected product/options.

Training checkpoints live at `/overlay/shopagent_rl_grpo_outputs/grpo/v2/`.
After the run, copy only the selected final checkpoint and portable adapter to
`model/`; retain train/eval logs in `logs/` and reports in `evaluation/`.

Acceptance criteria before a longer v2 run:

- zero-advantage steps below 10%;
- within-task tied-reward groups below 30%;
- response-length caps below 3%;
- no service-side session/state errors.

## v2b formal run

`scripts/run_grpo_v2b_200_b4n8_env32.sh` starts the 200-step run from SFT v1
with 4 distinct tasks × 8 rollouts = 32 trajectories per step.  It requires a
32-slot pack_api pool and writes raw resume checkpoints to
`/overlay/shopagent_rl_grpo_outputs/grpo/v2/full_200_b4_n8_env32/`.

v2a completed its 50-step gate with zero service-side session errors and
5/50 (10%) zero-advantage steps.  Its mean response-cap rate was 6.63%, so
v2b reduces the observation cap from 1400 to 1200 characters.
