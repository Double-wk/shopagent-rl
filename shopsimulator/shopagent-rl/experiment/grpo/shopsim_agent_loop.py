"""ShopSim agent loop for veRL multi-turn GRPO (vLLM async backend).

Drives the pack_api ShopSimulator env (:5000) turn-by-turn through veRL's
AsyncLLMServerManager (vLLM rollout). The policy emits free-text turns
(`Thought: ... Action: search[...] / click[...]`); each is parsed into an env
action by ShopSimEnv, the env returns a text observation that is fed back as the
NEXT user turn (response_mask=0 -- not policy tokens), and on the terminal step
(buy / cap) the segmented shaped reward (shop_env.reward.shaped) is returned as
`AgentLoopOutput.reward_score`. veRL places that at the last response token
(agent_loop.py:569-575) and feeds it to compute_grpo_outcome_advantage
(core_algos.py:264) -- outcome-reward GRPO, exactly the project target.

Wired in with zero veRL source changes:
  * registration: configs/shopsim_agent_loop.yaml (a YAML with
    `_target_: experiment.grpo.shopsim_agent_loop.ShopsimAgentLoop`), pointed to
    by `actor_rollout_ref.rollout.agent.agent_loop_config_path`;
  * selection: `actor_rollout_ref.rollout.agent.default_agent_loop: shopsim_agent`;
  * the dataset row carries `task_id` (a column) + `messages=[system]` so veRL
    passes `raw_prompt=[system]` and `task_id` as kwargs to run().

Env: this module runs under shopsim (transformers 4.57.6 after the 2026-08-08
downgrade; opd-rocm is identical). The sys.path insert below makes the shop_A
package (shop_env.*) importable from veRL's worker cwd (the verl repo root).
The env16 pack_api pool caps concurrent live trajectories; reset() retries on
        "Unable to get available environment" (same backoff as teacher/collect.py), so
        a rollout batch larger than the pool drains in waves.
"""
from __future__ import annotations

import os
import re
import sys
import time
from typing import Any
from uuid import uuid4

# veRL's worker process cwd is the verl repo root, NOT shop_A -- make shop_env.*
# importable regardless. (This module only ever runs under opd-rocm.)
_SHOP_A_ROOT = "/workspace/shopsimulator/shopagent-rl"
if _SHOP_A_ROOT not in sys.path:
    sys.path.insert(0, _SHOP_A_ROOT)

from verl.experimental.agent_loop.agent_loop import (  # noqa: E402
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

from shop_env.client import ShopEnvClient  # noqa: E402
from shop_env.wrapper import ShopSimEnv, parse_model_action  # noqa: E402
from shop_env.obs_format import format_observation  # noqa: E402
from shop_env import reward as R  # noqa: E402
from experiment.counterfactual_grading import grade_response  # noqa: E402

# Per-dimension shaped-reward weights for GRPO credit assignment (shop_env/reward.py).
# Attribute / spec weighted high -- the long-horizon discriminative skill.
SHOPSIM_REWARD_WEIGHTS = {"r_type": 0.20, "r_att": 0.30, "r_option": 0.30, "r_price": 0.20}

# C1 budget gate on the terminal reward (shop_env/reward.py `shaped`): the
# counterfactual probe showed the vanilla weighted sum trains price-blind
# committing. "none" (default) keeps the v2b behaviour bit-exact.
SHOPSIM_BUDGET_MODE = os.environ.get("SHOPSIM_REWARD_BUDGET_MODE", "none")
SHOPSIM_BUDGET_PENALTY = float(os.environ.get("SHOPSIM_REWARD_BUDGET_PENALTY", "0.5"))
SHOPSIM_CERTIFIED_REWARD_WEIGHT = float(os.environ.get("SHOPSIM_CERTIFIED_REWARD_WEIGHT", "1.0"))
SHOPSIM_CF_LENIENT_REWARD = float(os.environ.get("SHOPSIM_CF_LENIENT_REWARD", "0.5"))

# Match the first complete, dispatchable action.  This is deliberately stricter
# than the environment's permissive parser: it provides the exact token boundary
# after which continuations are no longer part of the SFT-style assistant turn.
_COMPLETE_ACTION_RE = re.compile(r"(?:Action:\s*)?(?:search|click)\s*\[.*?\]", re.IGNORECASE | re.DOTALL)

_POOL_BUSY = "Unable to get available environment"


@register("shopsim_agent")
class ShopsimAgentLoop(AgentLoopBase):
    """Multi-turn shopping agent over the pack_api env pool."""

    def __init__(self, *args, **kwargs):
        # veRL 0.8.0: AgentLoopBase 用 __init__(self, trainer_config, server_manager,
        # tokenizer, processor, dataset_cls, data_config) 设实例属性 (self.config /
        # self.rollout_config / self.tokenizer / self.apply_chat_template_kwargs /
        # self.system_prompt / self.loop), 取代 0.7.0 的 @classmethod init_class(cls,...)
        # 类属性机制。子类 __init__ 调 super() 后从 self.rollout_config 读自己的配置。
        super().__init__(*args, **kwargs)
        mt = self.rollout_config.multi_turn
        self.max_turns = int(mt.max_assistant_turns)        # == env max_steps
        self.response_length = int(self.rollout_config.response_length)  # hard stop for the response region
        # A single assistant turn only needs to emit one Thought/Action pair.  If
        # this is left at the full response budget, vLLM can spend thousands of
        # tokens rambling before the environment gets a chance to advance.
        self.turn_max_tokens = max(32, int(os.environ.get("SHOPSIM_TURN_MAX_TOKENS", "160")))
        self.obs_max_chars = max(600, int(os.environ.get("SHOPSIM_OBS_MAX_CHARS", "1800")))
        # provenance: reward shaping variant must be visible in every run log
        print(f"[ShopsimAgentLoop] reward budget_mode={SHOPSIM_BUDGET_MODE} "
              f"budget_penalty={SHOPSIM_BUDGET_PENALTY} weights={SHOPSIM_REWARD_WEIGHTS}",
              flush=True)
        # Every assistant message in the SFT data is delimited by Qwen's chat
        # template terminator.  The async GRPO rollout must use the same boundary:
        # without it vLLM keeps sampling after a valid Action until max_tokens,
        # then those unrelated suffix tokens receive policy loss.
        self.im_end_id = int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self.eos_id = int(self.tokenizer.eos_token_id)
        self.stop_token_ids = [x for x in dict.fromkeys((self.im_end_id, self.eos_id)) if x >= 0]
        # Chat-template system prefix, reused to strip the leading system block when
        # re-tokenizing an appended user turn (mirrors ToolAgentLoop in veRL).
        self.system_prompt_ids = self.tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **self.apply_chat_template_kwargs
        )
        self.base_url = os.environ.get("SHOP_ENV_BASE_URL", "http://127.0.0.1:5000")
        self.if_persona = bool(kwargs.get("if_persona", False))
        print(f"[ShopsimAgentLoop] __init__: max_turns={self.max_turns} "
              f"response_length={self.response_length} turn_max_tokens={self.turn_max_tokens} "
              f"stop_token_ids={self.stop_token_ids} base_url={self.base_url}")

    # -------------------------------------------------------------- helpers
    async def _reset_with_retry(self, env: ShopSimEnv, task_id: int,
                                retries: int = 5, backoff: float = 6.0):
        """reset() with backoff -- absorbs the transient 'pool full' error so a
        rollout batch larger than the env16 pool drains in waves instead of failing."""
        last = None
        for attempt in range(retries):
            try:
                return await self.loop.run_in_executor(None, lambda: env.reset(task_id))
            except RuntimeError as e:
                if _POOL_BUSY not in str(e):
                    raise
                last = e
                await _async_sleep(self.loop, backoff * (attempt + 1))
        raise last

    async def _encode_user_turn(self, text: str) -> list[int]:
        ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        # strip the duplicated leading system-template block
        return [int(x) for x in ids[len(self.system_prompt_ids):]]

    def _tokens_through_first_action(self, toks: list[int]) -> list[int]:
        """Return the sampled-token prefix ending at the first complete action.

        Decoding token prefixes (rather than re-encoding a character substring)
        preserves the exact token IDs sampled by vLLM, so actor log-probabilities
        and policy loss remain aligned.  A turn cap is at most 160 tokens, making
        this simple boundary scan negligible beside generation time.
        """
        for end in range(1, len(toks) + 1):
            text = self.tokenizer.decode(toks[:end], skip_special_tokens=True)
            match = _COMPLETE_ACTION_RE.search(text)
            if match and parse_model_action(text[:match.end()]) is not None:
                return toks[:end]
        return []

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    async def _run_counterfactual(
        self, raw_prompt: list[dict[str, Any]], sampling_params: dict[str, Any], **kwargs
    ) -> AgentLoopOutput:
        """Run and programmatically grade one validated counterfactual state."""
        prompt_ids = [int(value) for value in await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                raw_prompt,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )]
        turn_sampling_params = dict(sampling_params)
        turn_sampling_params["max_tokens"] = min(self.turn_max_tokens, self.response_length)
        turn_sampling_params["stop_token_ids"] = self.stop_token_ids
        started = time.monotonic()
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=turn_sampling_params,
        )
        gen_time = time.monotonic() - started
        sampled = [int(value) for value in output.token_ids]
        response_ids = self._tokens_through_first_action(sampled)
        if response_ids:
            text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )
            side = {
                "expected_action_intents": self._string_list(kwargs.get("expected_action_intents")),
                "allowed_actions": self._string_list(kwargs.get("allowed_actions")),
            }
            grade = grade_response(text, side)
            if grade["correct_strict"]:
                reward_score = SHOPSIM_CERTIFIED_REWARD_WEIGHT
            elif grade["correct_lenient"]:
                reward_score = SHOPSIM_CERTIFIED_REWARD_WEIGHT * SHOPSIM_CF_LENIENT_REWARD
            else:
                reward_score = 0.0
            response_mask = [1] * len(response_ids)
        else:
            text = self.tokenizer.decode(sampled, skip_special_tokens=True) if sampled else ""
            grade = {
                "action": None,
                "correct_strict": False,
                "correct_lenient": False,
                "unparseable": True,
            }
            response_ids = sampled[: self.response_length] or [self.eos_id]
            response_mask = [0] * len(response_ids)
            reward_score = 0.0

        pair_id = str(kwargs.get("pair_id") or "")
        side_name = str(kwargs.get("side") or "")
        intervention_type = str(kwargs.get("intervention_type") or "")
        print(
            f"[ShopsimAgentLoop] certified pair_id={pair_id} side={side_name} "
            f"type={intervention_type} action={grade.get('action')} "
            f"strict={grade['correct_strict']} lenient={grade['correct_lenient']} "
            f"reward={reward_score:.4f}",
            flush=True,
        )
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            reward_score=reward_score,
            num_turns=1,
            metrics=AgentLoopMetrics(generate_sequences=gen_time),
            extra_fields={
                "pair_id": pair_id,
                "side": side_name,
                "intervention_type": intervention_type,
                "counterfactual_grade": grade,
                "response_preview": " ".join(text.split())[:180],
            },
        )

    # -------------------------------------------------------------- main loop
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        if str(kwargs.get("sample_mode") or "environment") == "counterfactual":
            # ``raw_prompt`` is supplied by veRL in kwargs.  Remove it before
            # forwarding the remaining row metadata because it is already the
            # explicit first argument to ``_run_counterfactual``.
            cf_kwargs = dict(kwargs)
            cf_kwargs.pop("raw_prompt", None)
            return await self._run_counterfactual(raw_prompt, sampling_params, **cf_kwargs)
        task_id = kwargs.get("task_id")
        if task_id is None:
            task_id = kwargs.get("extra_info", {}).get("task_id")
        assert task_id is not None, "ShopsimAgentLoop requires a `task_id` column in the dataset row"

        env = ShopSimEnv(ShopEnvClient(self.base_url),
                         max_steps=self.max_turns, if_persona=self.if_persona)
        prefix_ids: list[int] = []
        response_mask: list[int] = []      # 1 = policy token, 0 = env obs token
        reward_score = 0.0
        assistant_turns = 0
        action_types: list[str] = []
        legal_steps = 0
        gen_time = 0.0
        prompt_len = 0                     # set inside try; init so the except path stays well-defined
        try:
            obs, _ = await self._reset_with_retry(env, int(task_id))
            # keep only the system message(s) from the dataset, then build the real
            # first user turn = instruction + landing page (collect.py convention).
            sys_msgs = [m for m in raw_prompt if m.get("role") == "system"]
            messages = sys_msgs + [{"role": "user", "content": format_observation(obs, max_chars=self.obs_max_chars)}]
            prefix_ids = [int(x) for x in await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs),
            )]
            prompt_len = len(prefix_ids)

            done = False
            for _ in range(self.max_turns):
                if len(response_mask) >= self.response_length:
                    break  # response region full -- stop generating (veRL truncates/pads)
                t0 = time.monotonic()
                # vLLM's async server consumes (pops) max_tokens from the dict.
                # Use a fresh per-turn copy and explicitly cap generation by both
                # the per-turn budget and the remaining response-region budget.
                turn_sampling_params = dict(sampling_params)
                remaining = self.response_length - len(response_mask)
                turn_sampling_params["max_tokens"] = max(1, min(self.turn_max_tokens, remaining))
                turn_sampling_params["stop_token_ids"] = self.stop_token_ids
                out = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prefix_ids,
                    sampling_params=turn_sampling_params,
                )
                gen_time += time.monotonic() - t0
                toks = [int(x) for x in out.token_ids]
                assistant_turns += 1

                clean_toks = self._tokens_through_first_action(toks)
                if not clean_toks:
                    # No complete action: do not train the policy on unbounded
                    # prose/garbage.  Keep the sampled text only as masked
                    # diagnostics and end this zero-reward rollout.
                    prefix_ids += toks
                    response_mask += [0] * len(toks)
                    print(
                        f"[ShopsimAgentLoop] task_id={task_id} turn={assistant_turns} "
                        f"format_failure=no_complete_action tokens={len(toks)}",
                        flush=True,
                    )
                    break

                # The action is the supervised boundary from SFT.  Any vLLM
                # continuation after it (often to the cap) is discarded from both
                # context and policy loss, eliminating reward/loss decoupling.
                discarded = len(toks) - len(clean_toks)
                toks = clean_toks
                prefix_ids += toks
                response_mask += [1] * len(toks)
                if discarded:
                    print(
                        f"[ShopsimAgentLoop] task_id={task_id} turn={assistant_turns} "
                        f"trimmed_after_action kept={len(toks)} discarded={discarded}",
                        flush=True,
                    )

                # Keep the serialized conversation structurally identical to SFT
                # even if the backend did not return the stop token in token_ids.
                # This separator is context only, never a policy-loss token.
                if not toks or toks[-1] != self.im_end_id:
                    prefix_ids.append(self.im_end_id)
                    response_mask.append(0)

                asst_text = await self.loop.run_in_executor(
                    None, lambda t=toks: self.tokenizer.decode(t, skip_special_tokens=True)
                )
                _obs, _r, step_done, info = await self.loop.run_in_executor(
                    None, lambda: env.step(asst_text)
                )
                if info.get("action_type"):
                    action_types.append(str(info["action_type"]))
                if info.get("legal", False):
                    legal_steps += 1
                if step_done or not info.get("action_type"):
                    preview = " ".join(asst_text.replace("\n", " ").split())[:180]
                    print(
                        f"[ShopsimAgentLoop] task_id={task_id} turn={assistant_turns} "
                        f"action={info.get('action_type')} legal={info.get('legal')} "
                        f"done={step_done} tokens={len(toks)} text={preview!r}",
                        flush=True,
                    )
                if step_done:
                    done = True
                    # Terminal credit. reward_detail is populated by ShopSimEnv.step
                    # ONLY on a real Buy (env_done == True); on a step-cap / env_over
                    # termination it is {} and shaped() returns 0. We deliberately
                    # do NOT couple action-format legality into the terminal reward
                    # -- a Buy the env accepted is credited even if the action string
                    # was oddly formatted; format adherence is already shaped
                    # implicitly by the step budget (malformed -> no progress -> cap
                    # -> 0 reward). illegal/None -> default False -> no zeroing.
                    reward_score = R.shaped(
                        info.get("reward_detail", {}),
                        weights=SHOPSIM_REWARD_WEIGHTS,
                        budget_mode=SHOPSIM_BUDGET_MODE,
                        budget_penalty=SHOPSIM_BUDGET_PENALTY,
                    )
                    break
                # feed the env observation back as the next user turn (masked out of policy loss)
                obs_ids = await self._encode_user_turn(format_observation(_obs, max_chars=self.obs_max_chars))
                prefix_ids += obs_ids
                response_mask += [0] * len(obs_ids)

            if not done:
                reward_score = 0.0   # never reached a terminal buy -> no success reward
        except Exception as e:
            # A single trajectory's failure must NOT abort the whole rollout batch:
            # asyncio.gather propagates the first raised exception, which would kill
            # every other in-flight trajectory in the step. Emit a zero-reward,
            # structurally well-formed output so that one sample simply gets no
            # gradient instead of crashing the run.
            print(f"[ShopsimAgentLoop] task_id={task_id} rollout failed: {e!r}")
            reward_score = 0.0
            if not prefix_ids:
                # reset / first-prompt build failed -> synthesize a minimal valid
                # prompt so postprocess pads a clean (fully-masked) sample.
                sys_msgs = [m for m in raw_prompt if m.get("role") == "system"] or [{"role": "system", "content": ""}]
                prefix_ids = [
                    int(x) for x in self.tokenizer.apply_chat_template(
                        sys_msgs, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs)
                ]
                prompt_len = len(prefix_ids)
            # veRL's postprocessor expects tokenizer.pad(..., return_tensors="pt")
            # to receive at least one ID. A completely failed reset previously
            # returned an empty list, for which some tokenizers return a Python
            # list and crash the whole Ray batch. This EOS is masked from policy
            # loss and exists only to preserve the Tensor contract.
            if len(prefix_ids) == prompt_len:
                prefix_ids.append(self.eos_id)
                response_mask.append(0)
        finally:
            try:
                await self.loop.run_in_executor(None, env.release)
            except Exception:
                pass

        # Truncate the response region to response_length (mirrors ToolAgentLoop's
        # `[: self.response_length]`). tokenizer.pad(padding="max_length") only pads
        # shorter sequences -- it never truncates -- so an un-truncated overshoot
        # (a long single-turn generation pushing past response_length) would leave
        # response_ids longer than the pad target and break the downstream batch cat.
        response_ids = prefix_ids[prompt_len:][: self.response_length]
        metrics = AgentLoopMetrics(generate_sequences=gen_time)
        print(
            f"[ShopsimAgentLoop] task_id={task_id} summary turns={assistant_turns} "
            f"actions={','.join(action_types) or 'none'} legal={legal_steps} "
            f"response_tokens={len(response_ids)} reward={reward_score:.4f}",
            flush=True,
        )
        return AgentLoopOutput(
            prompt_ids=prefix_ids[:prompt_len],
            response_ids=response_ids,
            response_mask=response_mask[: self.response_length],
            reward_score=reward_score,
            num_turns=assistant_turns,
            metrics=metrics,
            # Environment rows carry no intervention pair, but the keys must still be
            # present: `_get_gen_batch` pops every non-reward column out of the driver
            # batch, so the pair metadata only survives a step if the agent loop hands
            # it back. Emitting empty strings keeps paired GRPO's contract check happy
            # on environment-only blocks (`add_joint_certified_bonus` then sees zero
            # complete relations and adds no bonus). Omitting them crashed the paired
            # run at the first environment block.
            extra_fields={
                "pair_id": "",
                "side": "",
                "intervention_type": "",
                # Keep the environment and certified-counterfactual output
                # schemas identical before DataProto.concat across workers.
                "counterfactual_grade": {},
                "response_preview": "",
            },
        )


async def _async_sleep(loop, seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)
