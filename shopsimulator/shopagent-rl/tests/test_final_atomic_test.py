from __future__ import annotations

import hashlib
import gzip
import json
from collections import Counter
from pathlib import Path
import unittest

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data/counterfactual/final_atomic_test_v1.jsonl"
SUMMARY = ROOT / "data/counterfactual/final_atomic_test_v1.summary.json"
EXCLUSIONS = ROOT / "experiments/splits/final_atomic_test_v1_exclusions.txt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gitignored_paths() -> set[str]:
    """Repo-relative paths this project deliberately keeps out of Git.

    Read straight from .gitignore rather than shelling out to `git check-ignore`
    so the audit works in a source tree without a working Git binary.
    """
    ignore_file = ROOT / ".gitignore"
    if not ignore_file.exists():
        return set()
    return {
        stripped
        for line in ignore_file.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    }


def _task_ids(path: Path) -> set[int]:
    if path.suffix == ".parquet":
        table = pq.read_table(path, columns=["task_id"])
        return {int(value.as_py()) for value in table.column("task_id") if value.as_py() is not None}
    text = path.read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list) and all(isinstance(value, (int, float)) for value in payload):
        return {int(value) for value in payload}
    result: set[int] = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row.get("task_id"), int):
            result.add(row["task_id"])
    return result


class FinalAtomicTestProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = [json.loads(line) for line in PAIRS.read_text(encoding="utf-8").splitlines()]
        cls.summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    def test_split_is_sealed_and_hash_pinned(self) -> None:
        self.assertTrue(self.summary["sealed"])
        self.assertEqual(self.summary["split"], "final-atomic-test-v1")
        self.assertEqual(self.summary["output_sha256"], _sha256(PAIRS))
        self.assertEqual(self.summary["rows"], len(self.records))

    def test_balanced_composition_and_unique_tasks_products(self) -> None:
        counts = Counter(record["intervention_type"] for record in self.records)
        self.assertEqual(counts, {
            "price_above_budget": 150,
            "option_swap": 75,
            "nuisance_display_note": 75,
        })
        self.assertEqual(len({record["task_id"] for record in self.records}), 300)
        self.assertEqual(len({record["source"]["asin"] for record in self.records}), 300)

    def test_relations_are_explicit_and_ordered(self) -> None:
        expected = {
            "price_above_budget": ["COMMIT", "SEARCH_ALTERNATIVE"],
            "option_swap": ["COMMIT", "SELECT_TARGET_OPTION"],
            "nuisance_display_note": ["COMMIT", "COMMIT"],
        }
        for record in self.records:
            self.assertEqual(record["split"], "final-atomic-test-v1")
            self.assertEqual(record["expected_relation"], expected[record["intervention_type"]])

    def test_no_task_overlap_with_any_predating_artifact(self) -> None:
        final_tasks = {record["task_id"] for record in self.records}
        products_path = ROOT.parent / "ShopSimulator/shop_env/data/items_eval_train.json.gz"
        with gzip.open(products_path, "rt", encoding="utf-8") as handle:
            products = json.load(handle)
        final_products = {record["source"]["asin"] for record in self.records}

        audited: list[str] = []
        unauditable: list[str] = []
        for line in EXCLUSIONS.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            path = ROOT / value
            if not path.exists():
                # Some predating artifacts are deliberately not committed (e.g. the
                # full candidate pool behind the frozen 800-row set). Those cannot be
                # audited from a fresh clone, but a *tracked* artifact going missing
                # must still fail loudly, so only gitignored paths are tolerated.
                self.assertIn(
                    value,
                    _gitignored_paths(),
                    f"{value} is listed as a predating artifact but is neither present "
                    f"nor gitignored, so provenance cannot be verified",
                )
                unauditable.append(value)
                continue
            tasks = _task_ids(path)
            self.assertFalse(final_tasks & tasks, value)
            used_products = {
                str(products[task_id].get("asin", ""))
                for task_id in tasks if 0 <= task_id < len(products)
            }
            self.assertFalse(final_products & used_products, value)
            audited.append(value)

        # Guard against the audit quietly shrinking to nothing.
        self.assertGreater(len(audited), len(unauditable), "too few artifacts audited")
        if unauditable:
            print(
                f"\n[provenance] audited {len(audited)} artifacts; "
                f"{len(unauditable)} not in this clone: {', '.join(unauditable)}"
            )


if __name__ == "__main__":
    unittest.main()
