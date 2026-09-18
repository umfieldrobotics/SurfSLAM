#!/usr/bin/env python3
"""
SLAM Trajectory Evaluation Script
==================================
Evaluates multiple SLAM algorithms across sequences using evo metrics
plus a custom completeness metric.

Usage:
    python evaluate_slam.py --config eval_config.yaml
    python evaluate_slam.py --config eval_config.yaml --output results.csv
"""

import argparse
import copy
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

# evo imports
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import lie_algebra
from evo import EvoException

sys.path.append(str(Path(__file__).resolve().parents[1]))
from surfslam.common.dataset_paths import load_config  # noqa: E402


def compute_completeness(
    traj_gt: PoseTrajectory3D,
    traj_est: PoseTrajectory3D,
    time_threshold_s: float = 0.1) -> Tuple[float, int, int]:
    """
    Compute trajectory completeness metric.

    Uses EVO's associate_trajectories to determine what proportion of GT poses
    were successfully matched to estimated poses within the time threshold.
    Completeness is the percentage of GT timestamps that have a match.
    """
    if len(traj_est.timestamps) == 0:
        return 0.0, 0, len(traj_gt.timestamps)

    # Use EVO's association to find matches
    traj_gt_sync, traj_est_sync = sync.associate_trajectories(
        traj_est, traj_gt, max_diff=1)

    num_gt_total = len(traj_est.timestamps)
    num_matched = len(traj_est_sync.timestamps)

    completeness = num_matched / num_gt_total if num_gt_total > 0 else 0.0
    return completeness, num_matched, num_gt_total


def compute_ape_metrics(
    traj_gt: PoseTrajectory3D,
    traj_est: PoseTrajectory3D,
    pose_relation: str = "trans_part",
    align: bool = True,
    align_origin: bool = False,
    correct_scale: bool = False,
    time_threshold_s: float = 0.1) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, any]]]:

    traj_gt_sync, traj_est_sync = sync.associate_trajectories(
        traj_gt, traj_est, max_diff=time_threshold_s)
    
    if len(traj_est_sync.timestamps) < 2:
        warnings.warn("Not enough synchronized poses for APE computation")
        return None, None

    alignment_info = None
    if align or correct_scale:
        traj_est_aligned = copy.deepcopy(traj_est_sync)
        r, t, s = traj_est_aligned.align(
            traj_gt_sync,
            correct_scale=correct_scale)

        # Store alignment information
        alignment_info = {
            "rotation": r.tolist(),
            "translation": t.tolist(),
            "scale": float(s),
            "correct_scale": correct_scale,
            "align": align
        }
    else:
        traj_est_aligned = traj_est_sync

    if pose_relation == "trans_part":
        pose_rel = metrics.PoseRelation.translation_part
    elif pose_relation == "rot_part":
        pose_rel = metrics.PoseRelation.rotation_angle_deg
    elif pose_relation == "full":
        pose_rel = metrics.PoseRelation.full_transformation
    else:
        pose_rel = metrics.PoseRelation.translation_part

    ape_metric = metrics.APE(pose_rel)
    ape_metric.process_data((traj_gt_sync, traj_est_aligned))

    stats = ape_metric.get_all_statistics()

    return {
        "ape_rmse": stats["rmse"],
        "ape_mean": stats["mean"],
        "ape_median": stats["median"],
        "ape_std": stats["std"],
        "ape_min": stats["min"],
        "ape_max": stats["max"],
        "ape_sse": stats["sse"],
        "num_poses_aligned": len(traj_est_aligned.timestamps),
        "aligned_completeness": len(traj_gt_sync.timestamps)/len(traj_gt.timestamps)}, alignment_info



def compute_rpe_metrics(
    traj_gt: PoseTrajectory3D,
    traj_est: PoseTrajectory3D,
    delta: float = 1.0,
    delta_unit: str = "m",
    align: bool = True,
    correct_scale: bool = False,
    time_threshold_s: float = 0.1) -> Optional[Dict[str, float]]:
    
    traj_gt_sync, traj_est_sync = sync.associate_trajectories(
        traj_gt, traj_est, max_diff=time_threshold_s)

    if len(traj_est_sync.timestamps) < 2:
        return None

    if align or correct_scale:
        traj_est_aligned = copy.deepcopy(traj_est_sync)
        traj_est_aligned.align(
            traj_gt_sync,
            correct_scale=correct_scale)
    else:
        traj_est_aligned = traj_est_sync

    unit_map = {
        "m": metrics.Unit.meters,
        "rad": metrics.Unit.radians,
        "deg": metrics.Unit.degrees,
        "f": metrics.Unit.frames}

    rpe_metric = metrics.RPE(
        metrics.PoseRelation.translation_part,
        delta=delta,
        delta_unit=unit_map.get(delta_unit, metrics.Unit.meters),
        all_pairs=False)
    rpe_metric.process_data((traj_gt_sync, traj_est_aligned))

    stats = rpe_metric.get_all_statistics()

    return {
        "rpe_rmse": stats["rmse"],
        "rpe_mean": stats["mean"],
        "rpe_median": stats["median"],
        "rpe_std": stats["std"],
        "rpe_min": stats["min"],
        "rpe_max": stats["max"],}

def find_trial_trajectories(
    alg_config: dict,
    sequence_name: str) -> List[Tuple[str, str]]:
    base_path = Path(alg_config["base_path"])

    # Result trees predating a sequence rename keep the old directory name. An alias lets
    # the config stay on canonical sequence names while still finding those runs
    # (e.g. monohansett_engine was called monohansett_start before 8970b4b).
    dir_name = alg_config.get("sequence_aliases", {}).get(sequence_name, sequence_name)
    trial_pattern = alg_config["trial_pattern"].format(sequence=dir_name)
    traj_glob = alg_config["traj_glob"]

    trial_dirs = sorted(glob.glob(str(base_path / trial_pattern)))

    results = []
    for trial_dir in trial_dirs:
        trial_name = Path(trial_dir).name

        # A trial that died mid-run can leave a complete-looking trajectory behind,
        # since it is written before the statistics file. Don't score those.
        if os.path.exists(os.path.join(trial_dir, "FAILED.txt")):
            warnings.warn(f"Skipping failed trial {trial_dir}")
            continue

        traj_files = glob.glob(os.path.join(trial_dir, traj_glob))

        if traj_files:
            results.append((trial_name, traj_files[0]))
        else:
            warnings.warn(
                f"No trajectory found in {trial_dir} matching {traj_glob}")

    return results


def evaluate_single_trial(
    traj_gt: PoseTrajectory3D,
    traj_est_path: str,
    config: dict,
    time_threshold_s: float = 0.1,
    correct_scale: bool = False,
    time_mult: float = 1.0,
    alignment_dir: Optional[str] = None,
    sequence_name: str = "",
    algorithm_name: str = "",
    trial_name: str = "") -> Optional[Dict[str, float]]:
    traj_est = file_interface.read_tum_trajectory_file(traj_est_path)
    if traj_est is None:
        return None

    # Apply timestamp multiplier if specified
    if time_mult != 1.0:
        traj_est.timestamps = traj_est.timestamps * time_mult

    results = {}

    ape_settings = config.get("ape", {})

    completeness, matched, total = compute_completeness(
        traj_gt, traj_est,
        time_threshold_s=time_threshold_s)
    results["completeness"] = completeness
    results["completeness_matched"] = matched
    results["completeness_total"] = total
    results["est_trajectory_length"] = len(traj_est.timestamps)

    try:
        results["gt_path_length_m"] = float(traj_gt.path_length)
        results["est_path_length_m"] = float(traj_est.path_length)
    except:
        pass

    alignment_info = None
    try:
        ape_results, alignment_info = compute_ape_metrics(
            traj_gt, traj_est,
            pose_relation=ape_settings.get("pose_relation", "trans_part"),
            align=ape_settings.get("align", True),
            align_origin=ape_settings.get("align_origin", False),
            correct_scale=correct_scale,
            time_threshold_s=time_threshold_s)
    except EvoException:
        ape_results = None

    if ape_results:
        results.update(ape_results)

    # Save alignment if we have one
    if alignment_info and alignment_dir:
        # Create hierarchical directory structure: alignments/algorithm/sequence/
        algo_seq_dir = os.path.join(alignment_dir, algorithm_name, sequence_name)
        os.makedirs(algo_seq_dir, exist_ok=True)

        alignment_filename = f"{trial_name}_alignment.json"
        alignment_path = os.path.join(algo_seq_dir, alignment_filename)

        # Add metadata
        alignment_data = {
            "sequence": sequence_name,
            "algorithm": algorithm_name,
            "trial": trial_name,
            "trajectory_path": traj_est_path,
            **alignment_info
        }

        with open(alignment_path, 'w') as f:
            json.dump(alignment_data, f, indent=2)

    try:
        rpe_results = compute_rpe_metrics(
            traj_gt, traj_est,
            align=ape_settings.get("align", True),
            correct_scale=correct_scale,
            time_threshold_s=time_threshold_s)
    except EvoException:
        rpe_results = {}

    if rpe_results:
        results.update(rpe_results)

    return results


def run_evaluation(config: dict, alignment_dir: Optional[str] = None) -> pd.DataFrame:
    all_results = []

    sequences = config["sequences"]
    algorithms = config["algorithms"]

    # Create alignment directory if specified
    if alignment_dir:
        os.makedirs(alignment_dir, exist_ok=True)
        print(f"Saving alignments to: {alignment_dir}\n")

    for seq_config in sequences:
        seq_name = seq_config["name"]
        gt_path = seq_config["ground_truth"]

        # Get sequence-specific time threshold (default to 0.1)
        seq_time_threshold = seq_config.get("time_threshold_s", 0.1)

        print(f"\n{'='*60}")
        print(f"Processing sequence: {seq_name}")
        print(f"  Time threshold: {seq_time_threshold}s")
        print(f"{'='*60}")

        traj_gt = file_interface.read_tum_trajectory_file(gt_path)
        if traj_gt is None:
            print(f"  [ERROR] Failed to load ground truth: {gt_path}")
            continue

        print(f"  Ground truth loaded: {len(traj_gt.timestamps)} poses")

        for alg_config in algorithms:
            alg_name = alg_config["name"]

            # Get algorithm-specific correct_scale setting (default to False)
            alg_correct_scale = alg_config.get("correct_scale", False)
            # Get algorithm-specific time multiplier (default to 1.0)
            alg_time_mult = alg_config.get("time_mult", 1.0)

            print(f"\n  Algorithm: {alg_name} (correct_scale={alg_correct_scale}, time_mult={alg_time_mult})")

            trials = find_trial_trajectories(alg_config, seq_name)

            if not trials:
                print(f"    [WARNING] No trials found")
                continue

            print(f"    Found {len(trials)} trials")

            for trial_name, traj_path in trials:
                print(f"    Evaluating {trial_name}...", end=" ")

                trial_results = evaluate_single_trial(
                    traj_gt, traj_path, config,
                    time_threshold_s=seq_time_threshold,
                    correct_scale=alg_correct_scale,
                    time_mult=alg_time_mult,
                    alignment_dir=alignment_dir,
                    sequence_name=seq_name,
                    algorithm_name=alg_name,
                    trial_name=trial_name)

                if trial_results:
                    trial_results["sequence"] = seq_name
                    trial_results["algorithm"] = alg_name
                    trial_results["trial"] = trial_name
                    trial_results["traj_path"] = traj_path
                    trial_results["time_threshold_s"] = seq_time_threshold
                    trial_results["correct_scale"] = alg_correct_scale
                    all_results.append(trial_results)
                    ape_rmse = trial_results.get('ape_rmse', 'N/A')
                    ape_rmse_str = f"{ape_rmse:.4f}" if isinstance(ape_rmse, (int, float)) else ape_rmse
                    print(
                        f"APE RMSE: {ape_rmse_str}, "
                        f"Completeness: {trial_results.get('completeness', 0)*100:.1f}%")
                else:
                    print("[FAILED]")

    return pd.DataFrame(all_results)


def compute_aggregated_stats(df: pd.DataFrame, config: dict, expected_trials: int = None) -> pd.DataFrame:
    """
    Compute aggregated statistics including median with expected_trials handling.

    Args:
        df: DataFrame with raw trial results
        config: Configuration dict containing sequences and algorithms
        expected_trials: Number of expected trials. If provided, missing trials
                        are counted as infinity (or -infinity for metrics where higher is better)
    """
    # Get all algorithm/sequence pairs from config
    all_sequences = [seq["name"] for seq in config["sequences"]]
    all_algorithms = [alg["name"] for alg in config["algorithms"]]

    # Determine numeric columns from data (if any exists)
    if not df.empty:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ["completeness_matched", "completeness_total"]
        agg_cols = [c for c in numeric_cols if c not in exclude_cols]
    else:
        # Default metrics if no data exists
        agg_cols = ["completeness", "ape_rmse", "ape_mean", "ape_median",
                    "ape_std", "rpe_rmse", "rpe_mean", "rpe_median", "rpe_std"]

    # Metrics where higher is better (use -inf for missing)
    higher_is_better = {"completeness"}

    # Create a row for every algorithm/sequence pair
    agg_stats = []
    for seq in all_sequences:
        for alg in all_algorithms:
            # Get data for this specific pair
            if not df.empty:
                group = df[(df["sequence"] == seq) & (df["algorithm"] == alg)]
            else:
                group = pd.DataFrame()

            num_trials = len(group)

            stats = {
                "sequence": seq,
                "algorithm": alg,
                "num_trials": num_trials
            }

            if expected_trials is not None:
                stats["expected_trials"] = expected_trials
                stats["missing_trials"] = max(0, expected_trials - num_trials)

            for col in agg_cols:
                if not group.empty:
                    values = group[col].dropna()
                else:
                    values = pd.Series([], dtype=float)

                if len(values) == 0:
                    # No successful trials - report NaN/inf
                    stats[f"{col}_mean"] = np.nan
                    stats[f"{col}_std"] = np.nan
                    stats[f"{col}_median"] = np.nan
                else:
                    # Compute mean and std from available values
                    stats[f"{col}_mean"] = values.mean()
                    stats[f"{col}_std"] = values.std()

                    # Compute median with padding if expected_trials is provided
                    if expected_trials is not None:
                        values_list = values.tolist()
                        missing = expected_trials - len(values_list)

                        if missing > 0:
                            # Pad with infinity/negative infinity
                            if col in higher_is_better:
                                # For metrics where higher is better, missing = -inf
                                padded_values = values_list + [-np.inf] * missing
                            else:
                                # For metrics where lower is better, missing = +inf
                                padded_values = values_list + [np.inf] * missing
                            stats[f"{col}_median"] = np.median(padded_values)
                        else:
                            # Have all expected trials (or more)
                            stats[f"{col}_median"] = np.median(values_list[:expected_trials])
                    else:
                        # No expected_trials - just compute median of available values
                        stats[f"{col}_median"] = values.median()

            agg_stats.append(stats)

    return pd.DataFrame(agg_stats)


def save_results(
    df_raw: pd.DataFrame,
    df_agg: pd.DataFrame,
    output_dir: str,
    output_format: str = "csv"):
    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(output_dir, f"results_raw.{output_format}")
    if output_format == "csv":
        df_raw.to_csv(raw_path, index=False)
    else:
        df_raw.to_json(raw_path, orient="records", indent=2)
    print(f"\nRaw results saved to: {raw_path}")

    agg_path = os.path.join(output_dir, f"results_aggregated.{output_format}")
    if output_format == "csv":
        df_agg.to_csv(agg_path, index=False)
    else:
        df_agg.to_json(agg_path, orient="records", indent=2)
    print(f"Aggregated results saved to: {agg_path}")

    if not df_agg.empty:
        summary_cols = ["sequence", "algorithm", "num_trials"]

        # Add expected_trials and missing_trials if they exist
        if "expected_trials" in df_agg.columns:
            summary_cols.extend(["expected_trials", "missing_trials"])

        # Add key metrics: mean, std, and median
        metric_cols = [
            "ape_rmse_mean", "ape_rmse_std", "ape_rmse_median",
            "rpe_rmse_mean", "rpe_rmse_std", "rpe_rmse_median",
            "completeness_mean", "completeness_std", "completeness_median"
        ]
        summary_cols.extend([c for c in metric_cols if c in df_agg.columns])

        # Filter to only existing columns
        summary_cols = [c for c in summary_cols if c in df_agg.columns]
        summary = df_agg[summary_cols].copy()

        summary_path = os.path.join(output_dir, "summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SLAM trajectories using evo metrics")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--output", "-o",default=None,help="Output directory (overrides config)")
    parser.add_argument("--format", "-f",choices=["csv", "json"],default="csv",help="Output format (default: csv)")
    parser.add_argument("--alignments", "-a",default=None,help="Directory to save alignment transformations (default: output_dir/alignments)")

    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    output_dir = args.output or config.get("output_dir", "./eval_results")
    expected_trials = config.get("expected_trials", None)

    # Set alignment directory
    if args.alignments is not None:
        alignment_dir = args.alignments
    else:
        # Default to output_dir/alignments
        alignment_dir = os.path.join(output_dir, "alignments")

    if expected_trials is not None:
        print(f"Expected trials per algorithm/sequence: {expected_trials}")

    df_raw = run_evaluation(config, alignment_dir=alignment_dir)

    if df_raw.empty:
        print("\n[WARNING] No results computed. Will generate empty aggregated stats.")

    df_agg = compute_aggregated_stats(df_raw, config, expected_trials=expected_trials)

    save_results(df_raw, df_agg, output_dir, args.format)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
