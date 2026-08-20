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
        self.assertEqual(
            manifest["grpo"]["independent"]["status"], "completed_but_weights_lost"
        )
        self.assertEqual(manifest["grpo"]["paired"]["status"], "completed_negative")
        self.assertTrue(manifest["grpo"]["independent"]["smoke_status"] == "passed")

    def test_manifest_records_the_overlay_artifact_loss(self) -> None:
        """The single-seed Independent result must not be silently extendable.

        Its adapter died with /overlay on 2026-08-19, so a rerun produces a
        different model; mixing its seeds with the recorded 0.35 would be wrong.
        """
        manifest = yaml.safe_load(
            (ROOT / "experiments/manifests/horizon10_clean_v1.yaml").read_text()
        )
        loss = manifest["artifact_loss"]
        self.assertEqual(manifest["grpo"]["independent"]["evaluation"]["seeds"], 1)
        self.assertTrue(any("export_step_200" in item for item in loss["lost"]))
        self.assertTrue(
            any("v6_horizon10_clean_from_base" in item for item in loss["survived"])
        )

    def test_artifact_policy_splits_by_reproducibility(self) -> None:
        """Bulk checkpoints may sit on ephemeral /overlay; results may not.

        The 2026-08-19 loss was not caused by using /overlay for checkpoints —
        it was caused by the exported adapter living there too. Raw FSDP state is
        recoverable by rerunning; an adapter whose eval numbers are already
        published is not.
        """
        config = (ROOT / "configs/grpo.yaml").read_text()
        self.assertIn(
            "default_local_dir: /overlay/shopagent_rl_artifacts/grpo_runs", config
        )

        manifest = yaml.safe_load(
            (ROOT / "experiments/manifests/horizon10_clean_v1.yaml").read_text()
        )
        grpo = manifest["grpo"]
        for key in ("independent_output", "paired_output"):
            self.assertTrue(grpo[key].startswith("/overlay/"), key)
        # Anything the paper cites must be a repo-relative path under outputs/.
        self.assertTrue(grpo["adapter_export_root"].startswith("outputs/"))
        for key, value in grpo["independent"]["evaluation"].items():
            if isinstance(value, str):
                self.assertTrue(value.startswith("outputs/"), f"{key}={value}")
        for key, value in manifest["sft"]["evaluation"].items():
            if isinstance(value, str):
                self.assertTrue(value.startswith("outputs/"), f"{key}={value}")

    def test_paths_module_is_the_only_interpreter_source(self) -> None:
        """No script may hardcode an interpreter that a machine move invalidates."""
        paths = (ROOT / "scripts/paths.sh").read_text()
        self.assertIn("SHOPAGENT_PY:-/workspace/miniconda3/envs/shopsim/bin/python", paths)
        self.assertIn("SHOPAGENT_ARTIFACT_ROOT:-/overlay/", paths)

        for script in sorted((ROOT / "scripts").glob("*.sh")):
            if script.name == "paths.sh":
                continue
            body = script.read_text()
            self.assertNotIn(
                "PY=/overlay/miniconda3", body, f"{script.name} hardcodes a dead interpreter"
            )

    def test_clean_sft_evaluation_entrypoint_is_serial_and_pinned(self) -> None:
        script = (ROOT / "scripts/eval_horizon10_clean_sft.sh").read_text()
        self.assertIn("heldout_atomic_pairs_v2.jsonl", script)
        self.assertIn("run_counterfactual_eval.sh", script)
        self.assertIn("run_eval.sh", script)
        self.assertIn("SFT_HORIZON10_CLEAN_V1", script)
        self.assertIn("max_turns 10", script)


if __name__ == "__main__":
    unittest.main()
