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
        self.assertEqual(config["train"]["batch_size"], 2)
        self.assertEqual(config["train"]["grad_accum"], 4)
        self.assertTrue(config["train"]["group_by_length"])

    def test_grpo_uses_the_pinned_base_snapshot(self) -> None:
        launcher = (ROOT / "scripts/run_horizon10_clean_grpo.sh").read_text()
        revision = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
        self.assertIn(f"snapshots/{revision}", launcher)
        self.assertIn('export MODEL_PATH="${MODEL_PATH:-$BASE_SNAPSHOT}"', launcher)
        self.assertIn(
            "outputs/sft/v6_horizon10_clean_from_base/model/training_output/lora_adapter",
            launcher,
        )
        self.assertIn("grpo_certified_natural_800_pairblocked.parquet", launcher)
        self.assertNotIn("v4_certified_corrective", launcher)

    def test_grpo_methods_only_toggle_the_pair_objective(self) -> None:
        launcher = (ROOT / "scripts/run_horizon10_clean_grpo.sh").read_text()
        self.assertIn("independent) PAIRED_OBJECTIVE=False", launcher)
        self.assertIn("paired) PAIRED_OBJECTIVE=True", launcher)
        self.assertIn(
            'algorithm.paired_intervention.enabled="$PAIRED_OBJECTIVE"',
            launcher,
        )
        self.assertIn('GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.35}"', launcher)
        self.assertIn(
            'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"',
            launcher,
        )

        common_launcher = (ROOT / "scripts/run_grpo.sh").read_text()
        self.assertIn(
            'actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS"',
            common_launcher,
        )

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

    def test_manifest_tracks_sft_and_matched_grpo_separately(self) -> None:
        manifest = yaml.safe_load(
            (ROOT / "experiments/manifests/horizon10_clean_v1.yaml").read_text()
        )
        self.assertEqual(manifest["status"], "in_progress")
        self.assertEqual(manifest["sft"]["status"], "completed")
        self.assertEqual(manifest["grpo"]["independent"]["status"], "running")
        self.assertEqual(manifest["grpo"]["paired"]["status"], "pending")
        self.assertTrue(manifest["grpo"]["independent"]["smoke_status"] == "passed")

    def test_clean_sft_evaluation_entrypoint_is_serial_and_pinned(self) -> None:
        script = (ROOT / "scripts/eval_horizon10_clean_sft.sh").read_text()
        self.assertIn("heldout_atomic_pairs_v2.jsonl", script)
        self.assertIn("run_counterfactual_eval.sh", script)
        self.assertIn("run_eval.sh", script)
        self.assertIn("SFT_HORIZON10_CLEAN_V1", script)
        self.assertIn("max_turns 10", script)


if __name__ == "__main__":
    unittest.main()
