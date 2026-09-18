#!/usr/bin/env python3
"""
Median accuracy, completeness, precision and recall per (method, scene).

Reads the overall_evaluation_metrics.csv that analysis/mapping/evaluate_maps.py writes and
collapses the per-trial rows into one row per method and scene.

    python analysis/compute_mapping_summary.py
    python analysis/compute_mapping_summary.py --input <dir>/overall_evaluation_metrics.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / "eval_results" / "map_comparison" / "maps"

METRICS = ["accuracy", "completeness", "precision", "recall"]

#: Worst attainable value per metric, used to score a run that never produced a map.
#: accuracy/completeness are unbounded distances, so a failure is +inf; precision/recall
#: are bounded in [0, 1], so a failure is 0.
FAILURE_VALUE = {
    "accuracy": np.inf,
    "completeness": np.inf,
    "precision": 0.0,
    "recall": 0.0,
}

#: Methods whose mapper is deterministic and so was run once on purpose. Their missing
#: trials are a deliberate choice, not failures, and must not be counted against them.
SINGLE_RUN_METHODS = ("turtlmap", "turtlmap_our_depths")


def pad_failed_trials(df, expected_trials, single_run_methods):
    """Add a worst-case row for every run that was expected but produced no map.

    A crashed run is a result, not a missing sample: dropping it and taking the median of
    whatever survived rewards a method for failing. Padding to `expected_trials` with the
    worst attainable value keeps the median honest -- with two failures out of five, the
    median becomes the worst of the three that did finish.
    """
    padded = [df]
    for (method, scene), group in df.groupby(["method", "scene"]):
        if method in single_run_methods:
            continue
        missing = expected_trials - len(group)
        if missing <= 0:
            continue
        print(f"  {method}/{scene}: {len(group)}/{expected_trials} runs produced a map, "
              f"scoring {missing} failure(s) as worst-case")
        padded.append(pd.DataFrame([
            {"method": method, "scene": scene, "exp_number": f"failed_{i}", **FAILURE_VALUE}
            for i in range(missing)
        ]))
    return pd.concat(padded, ignore_index=True)


def compute_mapping_summary(input_csv_path, output_csv_path,
                            expected_trials=5, single_run_methods=SINGLE_RUN_METHODS):
    """Collapse per-trial map metrics to a median per (method, scene)."""
    df = pd.read_csv(input_csv_path)

    scored = df.groupby(["method", "scene"]).size().rename("num_scored").reset_index()
    df = pad_failed_trials(df, expected_trials, single_run_methods)

    summary = df.groupby(["method", "scene"]).agg(
        {m: "median" for m in METRICS} | {"exp_number": "count"}).reset_index()

    summary.columns = ["method", "scene"] + [f"median_{m}" for m in METRICS] + ["num_trials"]
    summary = summary.merge(scored, on=["method", "scene"], how="left")
    summary = summary.sort_values(["method", "scene"]).reset_index(drop=True)

    summary.to_csv(output_csv_path, index=False)
    print(f"Summary saved to {output_csv_path}")
    print("\nSummary statistics:")
    print(summary.to_string(index=False))

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                        default=DEFAULT_DIR / "overall_evaluation_metrics.csv")
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to mapping_summary.csv beside --input")
    parser.add_argument("--expected-trials", type=int, default=5,
                        help="Runs each method was given. Any shortfall is scored as a "
                             "failure at the worst attainable value, not dropped.")
    parser.add_argument("--single-run-methods", nargs="*", default=list(SINGLE_RUN_METHODS),
                        help="Methods run once by design (deterministic mapper); their "
                             "missing trials are not treated as failures.")
    args = parser.parse_args()

    output = args.output or args.input.parent / "mapping_summary.csv"
    compute_mapping_summary(args.input, output, args.expected_trials,
                            tuple(args.single_run_methods))


if __name__ == "__main__":
    main()
