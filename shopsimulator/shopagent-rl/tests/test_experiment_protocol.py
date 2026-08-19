from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ExperimentProtocolTests(unittest.TestCase):
    def test_horizon10_clean_sft_starts_from_base(self) -> None:
        config = yaml.safe_load((ROOT / "configs/sft_horizon10_clean_v1.yaml").read_text())
        self.assertEqual(config["data"]["train_file"], str(ROOT / "data/sft_train_horizon10.jsonl"))
        self.assertNotIn("init_adapter", config["model"])
        self.assertEqual(config["model"]["revision"], "ea980cb0a6c2ae4b936e82123acc929f1cec04c1")
        self.assertIn("v6_horizon10_clean_from_base", config["output_dir"])
        self.assertTrue(config["train"]["gradient_checkpointing"])

    def test_grpo_uses_the_pinned_base_snapshot(self) -> None:
        launcher = (ROOT / "scripts/run_horizon10_clean_grpo.sh").read_text()
        revision = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
        self.assertIn(f"snapshots/{revision}", launcher)
        self.assertIn('export MODEL_PATH="${MODEL_PATH:-$BASE_SNAPSHOT}"', launcher)

    def test_manifest_data_hashes_and_rows(self) -> None:
        manifest = yaml.safe_load(
            (ROOT / "experiments/manifests/horizon10_clean_v1.yaml").read_text()
        )
        for section in ("sft", "grpo"):
            spec = manifest[section]
            path = ROOT / spec["data"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, spec["sha256"])
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as handle:
                    rows = sum(1 for line in handle if line.strip())
            else:
                import pyarrow.parquet as pq

                rows = pq.read_metadata(path).num_rows
            self.assertEqual(rows, spec["rows"])


if __name__ == "__main__":
    unittest.main()
