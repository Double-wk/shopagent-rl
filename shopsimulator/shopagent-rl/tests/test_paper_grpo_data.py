from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import unittest

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data/grpo_certified_paper_v1_800_pairblocked.summary.json"


class PaperGrpoDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        cls.parquet = ROOT / cls.summary["parquet"]
        cls.rows = pq.read_table(cls.parquet).to_pylist()

    def test_hash_and_composition(self) -> None:
        self.assertEqual(hashlib.sha256(self.parquet.read_bytes()).hexdigest(),
                         self.summary["parquet_sha256"])
        self.assertEqual(len(self.rows), 800)
        self.assertEqual(Counter(row["sample_mode"] for row in self.rows),
                         {"environment": 400, "counterfactual": 400})
        self.assertEqual(Counter(
            row["intervention_type"] for row in self.rows if row["intervention_type"]
        ), {
            "price_above_budget": 200,
            "option_swap": 120,
            "nuisance_display_note": 80,
        })
        self.assertLessEqual(self.summary["max_prompt_tokens"], 2048)

    def test_pair_blocks_are_complete(self) -> None:
        for start in range(0, len(self.rows), 4):
            block = self.rows[start:start + 4]
            relation_ids = {row["relation_id"] for row in block if row["relation_id"]}
            if not relation_ids:
                self.assertTrue(all(row["sample_mode"] == "environment" for row in block))
                continue
            self.assertEqual(len(relation_ids), 2)
            for relation_id in relation_ids:
                sides = {row["side"] for row in block if row["relation_id"] == relation_id}
                self.assertEqual(sides, {"original", "counterfactual"})

    def test_training_partitions_and_final_test_are_disjoint(self) -> None:
        environment = {row["task_id"] for row in self.rows if row["sample_mode"] == "environment"}
        paired = {row["task_id"] for row in self.rows if row["sample_mode"] == "counterfactual"}
        final = {
            json.loads(line)["task_id"]
            for line in (ROOT / self.summary["final_test"]).read_text(encoding="utf-8").splitlines()
        }
        self.assertFalse(environment & paired)
        self.assertFalse((environment | paired) & final)
        self.assertEqual(self.summary["final_test_product_overlap"], 0)

    def test_launcher_pins_the_input_and_supports_matched_modes(self) -> None:
        launcher = (ROOT / "scripts/run_paper_grpo_smoke.sh").read_text(encoding="utf-8")
        self.assertIn(self.summary["parquet_sha256"], launcher)
        self.assertIn("explicit_relation)", launcher)
        self.assertIn("residual)", launcher)
        self.assertIn("TOTAL_STEPS:-10", launcher)


if __name__ == "__main__":
    unittest.main()
