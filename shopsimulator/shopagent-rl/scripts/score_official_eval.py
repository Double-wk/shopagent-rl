"""Score official-harness outputs with the official get_score.py semantics.

Usage:
  /overlay/miniconda3/envs/shopsim/bin/python scripts/score_official_eval.py \
      <output_dir_with_task_jsons> [more_dirs...]

Reuses ShopSimulator/get_score.py::calculate_metrics verbatim so numbers are
byte-identical to the paper's harness.
"""
import sys
import os

sys.path.insert(0, "/workspace/shopsimulator/ShopSimulator")
from get_score import calculate_metrics, print_metrics  # noqa: E402


def main() -> None:
    for out_dir in sys.argv[1:]:
        files = sorted(
            os.path.join(out_dir, f)
            for f in os.listdir(out_dir)
            if f.endswith(".json")
        )
        print(f"=== {out_dir}: {len(files)} episodes ===")
        metrics = calculate_metrics(files)
        print_metrics(metrics)


if __name__ == "__main__":
    main()
