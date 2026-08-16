from __future__ import annotations

import unittest

from shop_env.wrapper import Action, ShopSimEnv, truncate_after_first_action


class WrapperProtocolTests(unittest.TestCase):
    def test_truncates_hallucinated_following_turn(self) -> None:
        raw = (
            "Thought: 选择目标规格。\nAction: click[M]"
            "\n<|im_end|>\n<|im_start|>user\nInstruction: fake"
            "\nAction: click[buy now]"
        )
        self.assertEqual(
            truncate_after_first_action(raw),
            "Thought: 选择目标规格。\nAction: click[M]",
        )

    def test_keeps_format_failure_for_diagnostics(self) -> None:
        raw = "Thought: 还需要查看详情。"
        self.assertEqual(truncate_after_first_action(raw), raw)

    def test_fuzzy_click_is_canonicalized_before_dispatch(self) -> None:
        env = ShopSimEnv(client=None)  # type: ignore[arg-type]
        env._last_clickables = ["buy now"]
        env._last_has_search = False
        captured: list[str] = []

        class Client:
            def interact(self, _idx, payload):
                captured.append(payload)
                return {"done": False, "over": False, "reward": 0.0, "instruction": ""}

        env.client = Client()  # type: ignore[assignment]
        env.env_idx = 0
        env.step("Action: click[buy now ]")
        self.assertEqual(captured, [Action("click", "buy now").to_env()])

    def test_search_when_unavailable_is_illegal(self) -> None:
        env = ShopSimEnv(client=None)  # type: ignore[arg-type]
        env._last_has_search = False
        captured: list[str] = []

        class Client:
            def interact(self, _idx, payload):
                captured.append(payload)
                return {"done": False, "over": False, "reward": 0.0, "instruction": ""}

        env.client = Client()  # type: ignore[assignment]
        env.env_idx = 0
        _obs, _reward, _done, info = env.step("Action: search[袜子]")
        self.assertFalse(info["legal"])
        self.assertEqual(captured, ["search[袜子]"])


if __name__ == "__main__":
    unittest.main()
