#!/usr/bin/env python3
"""
SLAM Statistics Summary Script
==============================
Summarizes pipeline statistics, runtime, and optimization timing from SLAM runs.

Usage:
    python summarize_statistics.py --config summary_config.yaml
    python summarize_statistics.py /path/to/run/directory
    python summarize_statistics.py /path/to/experiment/  # summarizes all trials
    python summarize_statistics.py --config cfg.yaml --latex  # output LaTeX table
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from surfslam.common.dataset_paths import load_config  # noqa: E402


# LaTeX table row definitions: (group_name, display_name, column_key, unit, format_spec)
# For rows with "_mean" suffix, we'll also look for "_std" to format as "mean $\pm$ std"
LATEX_TABLE_ROWS = [
    # Counts group
    ("Counts", "Stereo Frames", "stereo_processed", "", ".0f"),
    ("Counts", "Registrations", "registration_processed", "", ".0f"),
    # Timing group (mean +/- std format)
    ("Timing", "Stereo Inference", "stereo_inference_mean_ms", "ms", ".1f"),
    ("Timing", "Optimization", "opt_mean_ms", "ms", ".1f"),
    ("Timing", "Total Runtime", "runtime_without_overhead_sec", "s", ".1f"),
    # Drop rates group
    ("Drops", "Overall", "overall_drop_rate", r"\%", ".1f"),
    ("Drops", "Stereo", "stereo_drop_rate", r"\%", ".1f"),
    ("Drops", "Registration", "registration_drop_rate", r"\%", ".1f"),
    # Signal drops group
    ("Signal Drops", "Stereo Input", "stereo_input_dropped", "", ".0f"),
    ("Signal Drops", "Frame", "frame_dropped", "", ".0f"),
    ("Signal Drops", "Registration", "registration_result_dropped", "", ".0f"),
]


def parse_pipeline_statistics(filepath: str) -> Optional[Dict]:
    """Parse pipeline_statistics.txt file."""
    if not os.path.exists(filepath):
        return None

    stats = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse frame counts
    patterns = {
        'stereo_input': r'Stereo frames input:\s+(\d+)',
        'stereo_processed': r'Stereo frames processed:\s+(\d+)',
        'registration_processed': r'Registration frames processed:\s+(\d+)',
        'stereo_drop_rate': r'Stereo processing drop rate:\s+([\d.]+)%',
        'registration_drop_rate': r'Registration drop rate:\s+([\d.]+)%',
        'overall_drop_rate': r'Overall drop rate:\s+([\d.]+)%',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1)
            stats[key] = int(value) if key.endswith('_count') or key.endswith('_processed') or key == 'stereo_input' else float(value)

    # Parse signal drop statistics
    signal_pattern = r'(\w+):\n\s+Total dropped:\s+(\d+)\n\s+Number of slots:\s+(\d+)\n\s+Max age \(seconds\):\s+([\d.]+|None)'
    for match in re.finditer(signal_pattern, content):
        signal_name = match.group(1)
        stats[f'{signal_name}_dropped'] = int(match.group(2))
        stats[f'{signal_name}_slots'] = int(match.group(3))
        max_age = match.group(4)
        stats[f'{signal_name}_max_age'] = float(max_age) if max_age != 'None' else None

    return stats


def parse_runtime(filepath: str) -> Optional[Dict]:
    """Parse runtime.txt file."""
    if not os.path.exists(filepath):
        return None

    stats = {}

    with open(filepath, 'r') as f:
        for line in f:
            if 'With Overhead' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    stats['runtime_with_overhead_sec'] = float(match.group(1))
            elif 'Without Overhead' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    stats['runtime_without_overhead_sec'] = float(match.group(1))

    return stats


def parse_optimization_timing(filepath: str) -> Optional[Dict]:
    """Parse optimization timing file (surfslam_live_optimization_timing.txt)."""
    if not os.path.exists(filepath):
        return None

    times_ms = []
    timestamps = []
    modes = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 4:
                try:
                    timestamp = float(parts[0])
                    elapsed_ms = float(parts[2])
                    opt_mode = parts[3]

                    timestamps.append(timestamp)
                    times_ms.append(elapsed_ms)
                    modes[opt_mode] = modes.get(opt_mode, 0) + 1
                except (ValueError, IndexError):
                    continue

    if not times_ms:
        return None

    times_ms = np.array(times_ms)

    stats = {
        'opt_count': len(times_ms),
        'opt_mean_ms': float(np.mean(times_ms)),
        'opt_std_ms': float(np.std(times_ms)),
        'opt_median_ms': float(np.median(times_ms)),
        'opt_min_ms': float(np.min(times_ms)),
        'opt_max_ms': float(np.max(times_ms)),
        'opt_total_ms': float(np.sum(times_ms)),
    }

    # Add mode breakdown
    for mode, count in modes.items():
        stats[f'opt_{mode}_count'] = count

    # Compute duration of run from timestamps
    if len(timestamps) >= 2:
        stats['opt_duration_sec'] = timestamps[-1] - timestamps[0]

    return stats


def parse_stereo_timing(filepath: str) -> Optional[Dict]:
    """Parse process_stereo_timing.csv file for stereo inference timing."""
    if not os.path.exists(filepath):
        return None

    try:
        df = pd.read_csv(filepath)
    except Exception:
        return None

    if df.empty or 'stage' not in df.columns or 'duration_sec' not in df.columns:
        return None

    stats = {}

    # Extract stereo_inference timing
    stereo_df = df[df['stage'] == 'stereo_inference']
    if not stereo_df.empty:
        times_ms = stereo_df['duration_sec'].values * 1000  # convert to ms
        stats['stereo_inference_count'] = len(times_ms)
        stats['stereo_inference_mean_ms'] = float(np.mean(times_ms))
        stats['stereo_inference_std_ms'] = float(np.std(times_ms))
        stats['stereo_inference_median_ms'] = float(np.median(times_ms))
        stats['stereo_inference_min_ms'] = float(np.min(times_ms))
        stats['stereo_inference_max_ms'] = float(np.max(times_ms))
        stats['stereo_inference_total_ms'] = float(np.sum(times_ms))

    # Also extract process_stereo_total if present
    total_df = df[df['stage'] == 'process_stereo_total']
    if not total_df.empty:
        times_ms = total_df['duration_sec'].values * 1000
        stats['process_stereo_total_mean_ms'] = float(np.mean(times_ms))
        stats['process_stereo_total_std_ms'] = float(np.std(times_ms))

    return stats if stats else None


def find_run_directories(base_path: str) -> List[str]:
    """Find all run directories containing pipeline_statistics.txt."""
    base = Path(base_path)

    # Check if base_path itself is a run directory
    if (base / 'pipeline_statistics.txt').exists():
        return [str(base)]

    # Search for run directories
    run_dirs = []

    # Pattern 1: Direct trial directories (trial_0, trial_1, etc.)
    for trial_dir in sorted(base.glob('trial_*')):
        if (trial_dir / 'pipeline_statistics.txt').exists():
            run_dirs.append(str(trial_dir))

    # Pattern 2: Config directories with trials (config_0/trial_0, etc.)
    for config_dir in sorted(base.glob('config_*')):
        for trial_dir in sorted(config_dir.glob('trial_*')):
            if (trial_dir / 'pipeline_statistics.txt').exists():
                run_dirs.append(str(trial_dir))

    # Pattern 3: Nested experiment structure
    if not run_dirs:
        for stats_file in base.rglob('pipeline_statistics.txt'):
            run_dirs.append(str(stats_file.parent))

    return sorted(set(run_dirs))


def summarize_single_run(run_dir: str) -> Optional[Dict]:
    """Summarize statistics for a single run directory."""
    run_path = Path(run_dir)

    result = {'run_dir': run_dir}

    # Extract trial/config info from path
    parts = run_path.parts
    for i, part in enumerate(parts):
        if part.startswith('trial_'):
            result['trial'] = part
        elif part.startswith('config_'):
            result['config'] = part

    # Parse pipeline statistics
    pipeline_stats = parse_pipeline_statistics(run_path / 'pipeline_statistics.txt')
    if pipeline_stats:
        result.update(pipeline_stats)

    # Parse runtime
    runtime_stats = parse_runtime(run_path / 'runtime.txt')
    if runtime_stats:
        result.update(runtime_stats)

    # Parse optimization timing
    opt_timing_path = run_path / 'trajectory' / 'surfslam_live_optimization_timing.txt'
    opt_stats = parse_optimization_timing(opt_timing_path)
    if opt_stats:
        result.update(opt_stats)

    # Parse stereo timing
    stereo_timing_path = run_path / 'process_stereo_timing.csv'
    stereo_stats = parse_stereo_timing(stereo_timing_path)
    if stereo_stats:
        result.update(stereo_stats)

    return result


def compute_aggregated_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated statistics across all runs."""
    if df.empty:
        return pd.DataFrame()

    # Numeric columns to aggregate
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['stereo_input', 'stereo_processed', 'registration_processed']
    agg_cols = [c for c in numeric_cols if c not in exclude_cols]

    agg_stats = {'num_runs': len(df)}

    for col in agg_cols:
        values = df[col].dropna()
        if len(values) > 0:
            agg_stats[f'{col}_mean'] = values.mean()
            agg_stats[f'{col}_std'] = values.std()
            agg_stats[f'{col}_median'] = values.median()
            agg_stats[f'{col}_min'] = values.min()
            agg_stats[f'{col}_max'] = values.max()

    return pd.DataFrame([agg_stats])


def print_summary(df: pd.DataFrame, df_agg: pd.DataFrame):
    """Print a formatted summary to console."""
    print("\n" + "=" * 80)
    print("STATISTICS SUMMARY")
    print("=" * 80)

    print(f"\nNumber of runs analyzed: {len(df)}")

    if df.empty:
        print("No data to summarize.")
        return

    # Key metrics to display
    key_metrics = [
        ('Drop Rates', [
            ('overall_drop_rate', 'Overall Drop Rate', '%'),
            ('stereo_drop_rate', 'Stereo Drop Rate', '%'),
            ('registration_drop_rate', 'Registration Drop Rate', '%'),
        ]),
        ('Runtime', [
            ('runtime_without_overhead_sec', 'Runtime (no overhead)', 's'),
            ('runtime_with_overhead_sec', 'Runtime (with overhead)', 's'),
        ]),
        ('Optimization Timing', [
            ('opt_mean_ms', 'Mean Opt Time', 'ms'),
            ('opt_median_ms', 'Median Opt Time', 'ms'),
            ('opt_max_ms', 'Max Opt Time', 'ms'),
            ('opt_total_ms', 'Total Opt Time', 'ms'),
            ('opt_count', 'Opt Iterations', ''),
        ]),
    ]

    for section_name, metrics in key_metrics:
        print(f"\n{section_name}:")
        print("-" * 40)

        for col, display_name, unit in metrics:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    if len(values) > 1:
                        print(f"  {display_name}: {mean_val:.2f} +/- {std_val:.2f} {unit}")
                    else:
                        print(f"  {display_name}: {mean_val:.2f} {unit}")

    # Signal drop summary
    signal_names = ['stereo_input', 'backend_input', 'frame', 'completed_frame', 'registration_result']
    has_signal_data = any(f'{s}_dropped' in df.columns for s in signal_names)

    if has_signal_data:
        print(f"\nSignal Drops:")
        print("-" * 40)
        for signal in signal_names:
            dropped_col = f'{signal}_dropped'
            if dropped_col in df.columns:
                values = df[dropped_col].dropna()
                if len(values) > 0:
                    total = values.sum()
                    mean = values.mean()
                    print(f"  {signal}: {mean:.1f} avg ({total:.0f} total)")


def generate_latex_table(method_stats: Dict[str, pd.DataFrame], output_path: str) -> str:
    """
    Generate a LaTeX table comparing statistics across methods.

    Args:
        method_stats: Dict mapping method name to DataFrame of aggregated stats
        output_path: Path to save the LaTeX file

    Returns:
        LaTeX table string
    """
    method_names = list(method_stats.keys())
    num_methods = len(method_names)

    # Build the table
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Pipeline Statistics Comparison}")
    lines.append(r"\label{tab:pipeline_stats}")

    # Column spec: group | stat | method1 | method2 | ...
    col_spec = "ll" + "r" * num_methods
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row
    header = "Group & Metric & " + " & ".join(method_names) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Group rows by group name for multirow
    current_group = None
    group_row_count = 0

    # Pre-count rows per group
    group_counts = {}
    for group, _, _, _, _ in LATEX_TABLE_ROWS:
        group_counts[group] = group_counts.get(group, 0) + 1

    for i, (group, display_name, col_key, unit, fmt) in enumerate(LATEX_TABLE_ROWS):
        # Determine if this is first row of a new group
        if group != current_group:
            if current_group is not None:
                lines.append(r"\midrule")
            current_group = group
            group_row_count = group_counts[group]
            group_cell = rf"\multirow{{{group_row_count}}}{{*}}{{{group}}}"
        else:
            group_cell = ""

        # Build value cells for each method
        value_cells = []
        for method_name in method_names:
            df = method_stats[method_name]

            # Check if this is a timing column that should show mean +/- std
            if col_key.endswith('_mean_ms') or col_key == 'opt_mean_ms':
                # Get mean and std
                mean_col = col_key
                std_col = col_key.replace('_mean_ms', '_std_ms') if col_key.endswith('_mean_ms') else 'opt_std_ms'

                mean_val = df[mean_col].values[0] if mean_col in df.columns and not df.empty else None
                std_val = df[std_col].values[0] if std_col in df.columns and not df.empty else None

                if mean_val is not None and not np.isnan(mean_val):
                    if std_val is not None and not np.isnan(std_val):
                        val_str = f"${mean_val:{fmt}} \\pm {std_val:{fmt}}$ {unit}"
                    else:
                        val_str = f"${mean_val:{fmt}}$ {unit}"
                else:
                    val_str = "---"
            else:
                # Regular column - just show the value
                if col_key in df.columns and not df.empty:
                    val = df[col_key].values[0]
                    if val is not None and not np.isnan(val):
                        val_str = f"${val:{fmt}}$ {unit}" if unit else f"${val:{fmt}}$"
                    else:
                        val_str = "---"
                else:
                    val_str = "---"

            value_cells.append(val_str)

        row = f"{group_cell} & {display_name} & " + " & ".join(value_cells) + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_content = "\n".join(lines)

    # Save to file
    with open(output_path, 'w') as f:
        f.write(latex_content)

    return latex_content


def compute_method_aggregates(df: pd.DataFrame, method_name: str) -> pd.DataFrame:
    """Compute aggregated statistics for a single method across all its runs."""
    if df.empty:
        return pd.DataFrame()

    # For the latex table, we need per-run values aggregated
    # Most values should be summed (counts) or averaged (rates, times)
    agg_dict = {'method': method_name, 'num_runs': len(df)}

    # Columns to sum (counts)
    sum_cols = ['stereo_processed', 'registration_processed', 'stereo_input',
                'stereo_input_dropped', 'frame_dropped', 'registration_result_dropped',
                'backend_input_dropped', 'completed_frame_dropped']

    # Columns to average (rates, times)
    avg_cols = ['overall_drop_rate', 'stereo_drop_rate', 'registration_drop_rate',
                'opt_mean_ms', 'opt_std_ms', 'opt_median_ms', 'opt_max_ms',
                'stereo_inference_mean_ms', 'stereo_inference_std_ms',
                'runtime_without_overhead_sec', 'runtime_with_overhead_sec']

    for col in sum_cols:
        if col in df.columns:
            agg_dict[col] = df[col].sum()

    for col in avg_cols:
        if col in df.columns:
            agg_dict[col] = df[col].mean()

    return pd.DataFrame([agg_dict])


def find_method_run_directories(method_config: dict) -> List[str]:
    """Find run directories for a method based on config patterns."""
    base_path = Path(os.path.expanduser(method_config["base_path"]))
    trial_pattern = method_config.get("trial_pattern", "{sequence}/trial_*")
    sequences = method_config.get("sequences", ["*"])

    run_dirs = []
    for seq in sequences:
        pattern = trial_pattern.format(sequence=seq)
        for trial_dir in sorted(base_path.glob(pattern)):
            if (trial_dir / 'pipeline_statistics.txt').exists():
                run_dirs.append(str(trial_dir))

    # Fallback: search recursively if pattern didn't match
    if not run_dirs:
        run_dirs = find_run_directories(str(base_path))

    return run_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Summarize SLAM pipeline statistics, runtime, and optimization timing")
    parser.add_argument("path", nargs='?', default=None,
                        help="Path to run directory or experiment directory")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to YAML configuration file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for CSV files")
    parser.add_argument("--format", "-f", choices=["csv", "json"], default="csv",
                        help="Output format (default: csv)")
    parser.add_argument("--latex", "-l", action="store_true",
                        help="Generate LaTeX table comparing methods")

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = args.output
    elif args.config:
        output_dir = os.path.dirname(os.path.abspath(args.config))
    else:
        output_dir = '.'

    os.makedirs(output_dir, exist_ok=True)

    # Check if config has methods (for multi-method comparison)
    if args.config:
        config = load_config(args.config)

        methods = config.get('methods', None)

        if methods:
            # Multi-method mode: process each method separately
            method_stats = {}
            all_raw_results = []

            for method_config in methods:
                method_name = method_config["name"]
                print(f"\n{'=' * 60}")
                print(f"Processing method: {method_name}")
                print(f"{'=' * 60}")

                run_dirs = find_method_run_directories(method_config)

                if not run_dirs:
                    print(f"  Warning: No run directories found for {method_name}")
                    continue

                print(f"  Found {len(run_dirs)} run(s)")

                results = []
                for run_dir in run_dirs:
                    print(f"    Processing: {run_dir}")
                    result = summarize_single_run(run_dir)
                    if result:
                        result['method'] = method_name
                        results.append(result)
                        all_raw_results.append(result)

                df = pd.DataFrame(results)
                if not df.empty:
                    method_stats[method_name] = compute_method_aggregates(df, method_name)
                    print_summary(df, method_stats[method_name])

            # Save combined raw results
            if all_raw_results:
                df_all = pd.DataFrame(all_raw_results)
                raw_path = os.path.join(output_dir, f"statistics_raw.{args.format}")
                if args.format == "csv":
                    df_all.to_csv(raw_path, index=False)
                else:
                    df_all.to_json(raw_path, orient="records", indent=2)
                print(f"\nRaw statistics saved to: {raw_path}")

            # Generate LaTeX table if requested
            if args.latex and method_stats:
                latex_path = os.path.join(output_dir, "statistics_table.tex")
                latex_content = generate_latex_table(method_stats, latex_path)
                print(f"\nLaTeX table saved to: {latex_path}")
                print("\nLaTeX table content:")
                print("-" * 60)
                print(latex_content)

            print("\nDone!")
            return

        # Fall back to simple paths mode
        paths = config.get('paths', [])
        if isinstance(paths, str):
            paths = [paths]
    elif args.path:
        paths = [args.path]
    else:
        parser.print_help()
        sys.exit(1)

    # Simple mode: analyze all paths together
    all_run_dirs = []
    for path in paths:
        path = os.path.expanduser(path)
        run_dirs = find_run_directories(path)
        all_run_dirs.extend(run_dirs)
        if not run_dirs:
            print(f"Warning: No run directories found in {path}")

    if not all_run_dirs:
        print("Error: No run directories found.")
        sys.exit(1)

    print(f"Found {len(all_run_dirs)} run(s) to analyze")

    # Summarize each run
    results = []
    for run_dir in all_run_dirs:
        print(f"  Processing: {run_dir}")
        result = summarize_single_run(run_dir)
        if result:
            results.append(result)

    df = pd.DataFrame(results)
    df_agg = compute_aggregated_stats(df)

    # Print summary
    print_summary(df, df_agg)

    # Save raw results
    raw_path = os.path.join(output_dir, f"statistics_raw.{args.format}")
    if args.format == "csv":
        df.to_csv(raw_path, index=False)
    else:
        df.to_json(raw_path, orient="records", indent=2)
    print(f"\nRaw statistics saved to: {raw_path}")

    # Save aggregated results
    if not df_agg.empty:
        agg_path = os.path.join(output_dir, f"statistics_aggregated.{args.format}")
        if args.format == "csv":
            df_agg.to_csv(agg_path, index=False)
        else:
            df_agg.to_json(agg_path, orient="records", indent=2)
        print(f"Aggregated statistics saved to: {agg_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
