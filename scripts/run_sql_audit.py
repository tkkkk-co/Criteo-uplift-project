"""Run read-only SQL reconciliation of a frozen Criteo test experiment."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from criteo_uplift.runtime import configure_runtime

configure_runtime()

from criteo_uplift.sql_audit import run_sql_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "artifacts/conversion_1000000_seed42"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/sql_audit_v1",
        help="new output directory; existing directories and data/ are rejected",
    )
    args = parser.parse_args()
    result = run_sql_audit(args.source, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "n_eligible": result["n_eligible"],
                "primary_budget": result["primary_budget"],
                "frozen_selected_policy": result["frozen_selected_policy"],
                "maximum_absolute_sql_python_difference": result[
                    "maximum_absolute_sql_python_difference"
                ],
                "source_unchanged": result["source_unchanged"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
