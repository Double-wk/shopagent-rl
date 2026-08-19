from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

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


if __name__ == "__main__":
    import unittest

    unittest.main()
