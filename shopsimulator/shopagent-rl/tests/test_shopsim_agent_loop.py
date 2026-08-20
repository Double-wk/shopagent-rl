from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from experiment.grpo import shopsim_agent_loop
from experiment.grpo.shopsim_agent_loop import ShopsimAgentLoop


class ShopsimAgentLoopTests(IsolatedAsyncioTestCase):
    async def test_counterfactual_prompt_is_not_forwarded_twice(self) -> None:
        agent = object.__new__(ShopsimAgentLoop)
        agent._run_counterfactual = AsyncMock(return_value="result")
        prompt = [{"role": "user", "content": "test"}]

        result = await agent.run(
            {"temperature": 0.7},
            raw_prompt=prompt,
            sample_mode="counterfactual",
            pair_id="pair-1",
        )

        self.assertEqual(result, "result")
        agent._run_counterfactual.assert_awaited_once_with(
            prompt,
            {"temperature": 0.7},
            sample_mode="counterfactual",
            pair_id="pair-1",
        )

    async def test_environment_rollout_reports_empty_pair_metadata(self) -> None:
        """Paired GRPO reads pair metadata off the agent-loop output, not the dataset.

        `RayPPOTrainer._get_gen_batch` pops every non-reward column out of the
        driver batch before rollout, so `compute_advantage` only sees pair
        metadata that the agent loop hands back. Environment rows have no pair,
        but must still report the keys or the paired run dies on its first
        environment block.
        """
        agent = object.__new__(ShopsimAgentLoop)
        agent.base_url = "http://localhost:5000"
        agent.max_turns = 10
        agent.if_persona = False
        agent.response_length = 8192
        agent.eos_id = 151643
        agent.apply_chat_template_kwargs = {}
        agent.tokenizer = MagicMock()
        agent.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        # Fail the reset so the rollout takes the guarded zero-reward path; the
        # return contract is shared with a successful trajectory.
        agent._reset_with_retry = AsyncMock(side_effect=RuntimeError("env pool empty"))

        with patch.object(shopsim_agent_loop, "ShopSimEnv"), patch.object(
            shopsim_agent_loop, "ShopEnvClient"
        ):
            output = await agent.run(
                {"temperature": 0.7},
                raw_prompt=[{"role": "system", "content": "sys"}],
                sample_mode="environment",
                task_id=12141,
            )

        self.assertEqual(output.reward_score, 0.0)
        for key in ("pair_id", "side", "intervention_type"):
            self.assertIn(key, output.extra_fields)
            self.assertEqual(output.extra_fields[key], "")
        self.assertEqual(output.extra_fields["counterfactual_grade"], {})
        self.assertEqual(output.extra_fields["response_preview"], "")


if __name__ == "__main__":
    import unittest

    unittest.main()
