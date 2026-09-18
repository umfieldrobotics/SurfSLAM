#!/usr/bin/env python3
"""
Plot ablation-run trajectories aligned against ground truth.

Reuses the same association/alignment logic as evaluate_trajectories.py so
the plotted trajectories match the numbers in results_raw.csv. For each
sequence in the eval config, draws ground truth plus every trial's estimate
(aligned to GT) on a top-down (X-Y) view.

Usage:
    python analysis/plot_ablation_trajectories.py analysis/ablation_eval_cfg.yaml \
        --output eval_results/ablations/trajectories.png
"""

import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evo.core import sync
from evo.tools import file_interface

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))
from surfslam.common.dataset_paths import load_config  # noqa: E402
from evaluate_trajectories import find_trial_trajectories  # noqa: E402


def load_aligned_trial(traj_gt, traj_est_path: str, time_threshold_s: float, correct_scale: bool):
    traj_est = file_interface.read_tum_trajectory_file(traj_est_path)
    traj_gt_sync, traj_est_sync = sync.associate_trajectories(
        traj_gt, traj_est, max_diff=time_threshold_s)
    if len(traj_est_sync.timestamps) < 2:
        return None
    traj_est_aligned = copy.deepcopy(traj_est_sync)
    traj_est_aligned.align(traj_gt_sync, correct_scale=correct_scale)
    return traj_est_aligned


def main():
    parser = argparse.ArgumentParser(description="Plot ablation trajectories against ground truth")
    parser.add_argument("config", help="Path to the same YAML config used by evaluate_trajectories.py")
    parser.add_argument("--output", "-o", default=None, help="Output image path (default: <output_dir>/trajectories.png)")
    parser.add_argument("--cols", type=int, default=2, help="Number of subplot columns")
    args = parser.parse_args()

    config = load_config(args.config)
    sequences = config["sequences"]
    algorithms = config["algorithms"]

    n = len(sequences)
    cols = args.cols
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows), squeeze=False)

    for idx, seq_config in enumerate(sequences):
        ax = axes[idx // cols][idx % cols]
        seq_name = seq_config["name"]
        gt_path = seq_config["ground_truth"]
        time_threshold_s = seq_config.get("time_threshold_s", 0.1)

        traj_gt = file_interface.read_tum_trajectory_file(gt_path)
        if traj_gt is None:
            ax.set_title(f"{seq_name} [GT LOAD FAILED]")
            continue

        gt_xyz = traj_gt.positions_xyz
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], color="black", linewidth=2.2, label="Ground truth", zorder=5)

        found_any = False
        for alg_config in algorithms:
            alg_name = alg_config["name"]
            correct_scale = alg_config.get("correct_scale", False)
            trials = find_trial_trajectories(alg_config, seq_name)

            colors = plt.cm.viridis([i / max(len(trials) - 1, 1) for i in range(len(trials))])
            for trial_idx, (trial_name, traj_path) in enumerate(trials):
                traj_est_aligned = load_aligned_trial(
                    traj_gt, traj_path, time_threshold_s, correct_scale)
                if traj_est_aligned is None:
                    continue
                found_any = True
                est_xyz = traj_est_aligned.positions_xyz
                ax.plot(
                    est_xyz[:, 0], est_xyz[:, 1],
                    color=colors[trial_idx], alpha=0.75, linewidth=1.1,
                    label=f"{alg_name} {trial_name}" if len(algorithms) > 1 else trial_name)

        ax.set_title(seq_name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        if not found_any:
            ax.text(0.5, 0.5, "No trials found", transform=ax.transAxes,
                    ha="center", va="center", color="red")
        ax.legend(fontsize=7, loc="best")

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.tight_layout()

    output_dir = config.get("output_dir", "./eval_results")
    output_path = args.output or str(Path(output_dir) / "trajectories.png")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved trajectory plot to: {output_path}")


if __name__ == "__main__":
    main()
