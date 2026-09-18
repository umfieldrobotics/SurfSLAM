#!/usr/bin/env python3
"""
Convert SLAM evaluation results to LaTeX table format.

Usage:
    python results_to_latex.py results_aggregated.csv --metric ape_rmse_median --output table.tex
    python results_to_latex.py results_aggregated.csv -m ape_rmse_median -o table.tex --our-method surfslam
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Mapping from sequence names in config to table column names
SEQUENCE_MAPPING = {
    # Renamed from monohansett_start in 8970b4b ("rename configs to match paper");
    # the old names are kept so previously-generated result CSVs still tabulate.
    "monohansett_engine": ("Engine", "Full"),
    "monohansett_engine_visual": ("Engine", "Vis."),
    "monohansett_start": ("Engine", "Full"),
    "monohansett_start_visual": ("Engine", "Vis."),
    "monohansett_boiler": ("Boiler", "Full"),
    "monohansett_boiler_visual": ("Boiler", "Vis."),
    "monohansett_long": ("Long", "Full"),
    "monohansett_long_visual": ("Long", "Vis."),
}


# Per-algorithm presentation: display name, bibtex key, and platform column.
# Row order here is the row order in the table; "our" methods still move below the rule.
METHOD_DISPLAY = {
    "droidslam":   ("DROID-SLAM", "teed2021droid",   "CPU+GPU"),
    "mast3r_slam": ("MASt3R SLAM", "murai2025mast3r", "CPU+GPU"),
    "orbslam3":    ("ORB-SLAM3", "orbslam3",         "CPU"),
    "svin2":       ("SVIn2", "svin2",                "CPU"),
    "turtlmap":    ("TURTLMap", "song2024turtlmap",  "CPU"),
    "vggt_slam":   ("VGGT-SLAM", None,               "CPU+GPU"),
    "surfslam":    ("Ours", None,                    "CPU+GPU"),
    # Same configuration as `surfslam`, run on the vehicle's onboard Jetson
    # (${slam_results}/jetson_resubmission_trials, see analysis/traj_eval_cfg.yaml).
    "surfslam_jetson": ("Ours (Jetson)", None,       "Jetson"),
}

DEFAULT_CAPTION = (
    "Estimated trajectories are compared against the ground truth. The reported metric is "
    "Root Mean Square Absolute Pose Error (APE). Each result is the median of five runs. "
    "MASt3R-SLAM is a monocular method, so scale-invariant trajectory evaluations were "
    "conducted for MASt3R-SLAM only. Lower is better."
)


def method_label(alg: str) -> str:
    """Display name plus \\cite{...}, falling back to the raw algorithm name."""
    entry = METHOD_DISPLAY.get(alg)
    if entry is None:
        print(f"warning: no METHOD_DISPLAY entry for {alg!r}; using the raw name and no "
              "platform (add one to keep the table consistent)", file=sys.stderr)
        return alg.replace("_", "\\_")
    display, cite, _ = entry
    return f"{display} \\cite{{{cite}}}" if cite else display


def method_platform(alg: str) -> str:
    entry = METHOD_DISPLAY.get(alg)
    return entry[2] if entry else "--"


def load_results(csv_path: str) -> pd.DataFrame:
    """Load aggregated results CSV."""
    return pd.read_csv(csv_path)


def is_failure(value: float) -> bool:
    """Check if a value represents a failure (NaN or inf)."""
    return pd.isna(value) or np.isinf(value)


def format_value(value: float, precision: int = 3) -> str:
    """Format a numeric value for display."""
    if is_failure(value):
        return "fail"
    return f"{value:.{precision}f}"


def get_column_groups(no_vis: bool = False) -> List[Tuple[str, List[str]]]:
    """Get the column groups for the table header."""
    if no_vis:
        return [
            ("Engine", ["Full"]),
            ("Boiler", ["Full"]),
            ("Long", ["Full"]),
        ]
    return [
        ("Engine", ["Full", "Vis."]),
        ("Boiler", ["Full", "Vis."]),
        ("Long", ["Full", "Vis."]),
    ]


def create_pivot_table(
    df: pd.DataFrame,
    metric: str,
    sequence_mapping: Dict[str, Tuple[str, str]],
    no_vis: bool = False
) -> pd.DataFrame:
    """
    Create a pivot table with algorithms as rows and (scene, subseq) as columns.

    Args:
        df: DataFrame with aggregated results
        metric: Metric column name (e.g., 'ape_rmse_median')
        sequence_mapping: Mapping from sequence names to (scene, subseq) tuples
        no_vis: If True, exclude visualization subsequences

    Returns:
        Pivot table with MultiIndex columns
    """
    # Filter to only sequences we care about
    df_filtered = df[df['sequence'].isin(sequence_mapping.keys())].copy()

    # Add scene and subsequence columns
    df_filtered['scene'] = df_filtered['sequence'].map(lambda x: sequence_mapping[x][0])
    df_filtered['subseq'] = df_filtered['sequence'].map(lambda x: sequence_mapping[x][1])

    # Filter out Vis. subsequences if --no-vis is enabled
    if no_vis:
        df_filtered = df_filtered[df_filtered['subseq'] != "Vis."]

    # Create pivot table
    pivot = df_filtered.pivot_table(
        values=metric,
        index='algorithm',
        columns=['scene', 'subseq'],
        aggfunc='first'
    )

    # Ensure column order matches our desired order
    if no_vis:
        column_order = [
            ("Engine", "Full"),
            ("Boiler", "Full"),
            ("Long", "Full"),
        ]
    else:
        column_order = [
            ("Engine", "Full"), ("Engine", "Vis."),
            ("Boiler", "Full"), ("Boiler", "Vis."),
            ("Long", "Full"), ("Long", "Vis."),
        ]

    # Reindex to ensure all columns exist (fill with NaN if missing)
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(column_order), fill_value=np.nan)

    return pivot


def find_best_and_second_best(
    values: List[float],
    lower_is_better: bool = True
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find indices of best and second-best values.

    Args:
        values: List of numeric values
        lower_is_better: If True, smaller values are better

    Returns:
        (best_idx, second_best_idx) or (None, None) if not enough valid values
    """
    # Filter out failures
    valid_indices = [i for i, v in enumerate(values) if not is_failure(v)]

    if len(valid_indices) < 2:
        if len(valid_indices) == 1:
            return valid_indices[0], None
        return None, None

    valid_values = [(i, values[i]) for i in valid_indices]

    # Sort by value
    if lower_is_better:
        valid_values.sort(key=lambda x: x[1])
    else:
        valid_values.sort(key=lambda x: x[1], reverse=True)

    best_idx = valid_values[0][0]
    second_best_idx = valid_values[1][0]

    return best_idx, second_best_idx


def generate_latex_table(
    pivot: pd.DataFrame,
    our_methods: Optional[List[str]] = None,
    lower_is_better: bool = True,
    precision: int = 3,
    caption: str = "SLAM Results",
    label: str = "tab:slam_results",
    no_vis: bool = False
) -> str:
    """
    Generate LaTeX table from pivot table.

    Args:
        pivot: Pivot table with algorithms as rows
        our_methods: Name(s) of "our" method(s) to place in separate section at bottom
        lower_is_better: If True, lower values are better (for highlighting)
        precision: Number of decimal places
        caption: Table caption
        label: LaTeX label
        no_vis: If True, generate table without Vis. columns

    Returns:
        LaTeX table as string
    """
    lines = []
    column_groups = get_column_groups(no_vis)

    lines.append("\\begin{table}[htb]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")

    # Method, Platform, then one column per scene (two when Vis. columns are kept).
    per_scene = 1 if no_vis else 2
    col_spec = "l||" + "|".join(["c"] * (1 + len(column_groups) * per_scene))
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    if no_vis:
        # One column per scene: a single header row, each scene boxed like the paper table.
        cells = []
        for i, (scene, _) in enumerate(column_groups):
            align = "c" if i == len(column_groups) - 1 else "c|"
            cells.append(f"\\multicolumn{{1}}{{{align}}}{{{scene}}}")
        lines.append("Method & Platform & " + " & ".join(cells) + " \\\\")
        lines.append("\\hline")
    else:
        header1 = "\\multirow{2}{*}{Method} & \\multirow{2}{*}{Platform}"
        for scene, subseqs in column_groups:
            header1 += f" & \\multicolumn{{{len(subseqs)}}}{{c}}{{{scene}}}"
        header1 += " \\\\"
        lines.append(header1)
        header2 = "            & "
        for scene, subseqs in column_groups:
            for subseq in subseqs:
                header2 += f" & {subseq}           "
        header2 += "\\\\\\hline"
        lines.append(header2)

    # Separate "our" methods if specified
    algorithms = list(pivot.index)
    # Present in METHOD_DISPLAY order so the table row order is stable and matches the paper.
    order = {a: i for i, a in enumerate(METHOD_DISPLAY)}
    algorithms.sort(key=lambda a: (order.get(a, len(order)), a))
    if our_methods:
        our_algorithms = [m for m in our_methods if m in algorithms]
        other_algorithms = [a for a in algorithms if a not in our_algorithms]
    else:
        other_algorithms = algorithms
        our_algorithms = []

    # Generate data rows
    for alg in other_algorithms:
        row_values = []
        for col_idx, (scene, subseq) in enumerate(pivot.columns):
            value = pivot.loc[alg, (scene, subseq)]

            # Get ALL values for this column (including our method if it exists)
            all_column_values = [pivot.loc[a, (scene, subseq)] for a in algorithms]
            best_idx, second_idx = find_best_and_second_best(all_column_values, lower_is_better)

            # Get the position of current algorithm in full algorithms list
            alg_idx = algorithms.index(alg)

            # Format the value
            formatted = format_value(value, precision)

            # Add highlighting
            if not is_failure(value):
                if best_idx is not None and alg_idx == best_idx:
                    formatted = f"\\textbf{{{formatted}}}"
                elif second_idx is not None and alg_idx == second_idx:
                    formatted = f"\\underline{{{formatted}}}"

            row_values.append(formatted)

        row = (f"{method_label(alg)} & {method_platform(alg)} & "
               + " & ".join(row_values) + " \\\\")
        lines.append(row)

    # Add separator and "our" method rows if specified
    if our_algorithms:
        lines.append("\\hline")
        for our_algorithm in our_algorithms:
            row_values = []

            for col_idx, (scene, subseq) in enumerate(pivot.columns):
                value = pivot.loc[our_algorithm, (scene, subseq)]

                # Get ALL values for this column (including our methods)
                all_column_values = [pivot.loc[a, (scene, subseq)] for a in algorithms]
                best_idx, second_idx = find_best_and_second_best(all_column_values, lower_is_better)

                # Get the position of our algorithm in full algorithms list
                our_idx = algorithms.index(our_algorithm)

                # Format the value
                formatted = format_value(value, precision)

                # Add highlighting
                if not is_failure(value):
                    if best_idx is not None and our_idx == best_idx:
                        formatted = f"\\textbf{{{formatted}}}"
                    elif second_idx is not None and our_idx == second_idx:
                        formatted = f"\\underline{{{formatted}}}"

                row_values.append(formatted)

            # Only the last "our" row closes the table; several of them (e.g. desktop and
            # Jetson) belong in one block, not separated by a rule each.
            is_last = our_algorithm == our_algorithms[-1]
            row = (f"{method_label(our_algorithm)} & {method_platform(our_algorithm)} & "
                   + " & ".join(row_values) + (" \\\\\\hline" if is_last else " \\\\"))
            lines.append(row)

    # End table. Caption is emitted above the tabular; only the label follows it.
    lines.append("\\end{tabular}")
    lines.append("\\vspace{0.2cm}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SLAM evaluation results to LaTeX table")
    parser.add_argument("results_csv", help="Path to results_aggregated.csv")
    parser.add_argument(
        "--metric", "-m",
        default="ape_rmse_median",
        help="Metric to display (default: ape_rmse_median)")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output .tex file (default: print to stdout)")
    parser.add_argument(
        "--our-method",
        nargs="+",
        default=None,
        help="Name(s) of 'our' method(s) to place in separate section at bottom")
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Higher values are better (default: lower is better)")
    parser.add_argument(
        "--precision", "-p",
        type=int,
        default=3,
        help="Number of decimal places (default: 3)")
    parser.add_argument(
        "--caption",
        default=DEFAULT_CAPTION,
        help="Table caption")
    parser.add_argument(
        "--label",
        default="tab:slam_results",
        help="LaTeX label")
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip Vis. subsequences and only show Full columns (3 columns total)")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_csv}", file=sys.stderr)
    df = load_results(args.results_csv)

    # Check if metric exists
    if args.metric not in df.columns:
        print(f"ERROR: Metric '{args.metric}' not found in results.", file=sys.stderr)
        print(f"Available metrics: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Create pivot table
    print(f"Creating pivot table for metric: {args.metric}", file=sys.stderr)
    pivot = create_pivot_table(df, args.metric, SEQUENCE_MAPPING, no_vis=args.no_vis)

    # Generate LaTeX
    print("Generating LaTeX table...", file=sys.stderr)
    latex = generate_latex_table(
        pivot,
        our_methods=args.our_method,
        lower_is_better=not args.higher_is_better,
        precision=args.precision,
        caption=args.caption,
        label=args.label,
        no_vis=args.no_vis
    )

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(latex)
        print(f"LaTeX table saved to: {args.output}", file=sys.stderr)
    else:
        print(latex)

    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
