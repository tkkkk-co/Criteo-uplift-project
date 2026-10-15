#!/usr/bin/env python3
"""Run the fixed development-only sensitivity study without reusing old test rows.

Example: python run_stability.py
Use a fresh --output directory for each run; existing artifacts are protected.
"""

from criteo_uplift.runtime import configure_runtime

configure_runtime()

import argparse
from pathlib import Path

from criteo_uplift.data import ROOT
from criteo_uplift.stability import DEFAULT_FRACTIONS, DEFAULT_SEEDS, run_stability


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "artifacts/cache/sample_1000000_seed42.npz"
    )
    parser.add_argument(
        "--base-run", type=Path, default=ROOT / "artifacts/conversion_1000000_seed42"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/stability_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--fractions", type=float, nargs="+", default=DEFAULT_FRACTIONS)
    parser.add_argument("--trees", type=int, default=150)
    parser.add_argument("--primary-budget", type=float, default=0.2)
    parser.add_argument(
        "--outcome", choices=["conversion", "visit"], default="conversion"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    output = run_stability(**vars(parse_args()))
    print(f"Completed development-only sensitivity study: {output}")
