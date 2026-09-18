#!/usr/bin/env python3
r"""
Convert ablation results (results_aggregated.csv / summary.csv from
evaluate_trajectories.py) into a LaTeX table: scenes as rows, ablation
conditions (No DVL, No Barometer, ...) as grouped columns, each showing
APE RMSE (mean +/- std) and completeness.

--style checkmarks instead emits the compact sensor-toggle form: one row per
sensor combination with \cmark / \xmark, scenes as value columns. That layout
drops completeness (identical across ablations for a given scene, since it
measures coverage rather than accuracy) and the std column.

Sequence names are expected to follow the cfg/tbnms_ablations/ convention:
    monohansett_<scene>_no_<sensor>   e.g. monohansett_boiler_no_dvl
    monohansett_<scene>                (unablated baseline -> "Full")

Usage:
    python analysis/ablation_summary_to_latex.py eval_results/ablations_boiler_engine/results_aggregated.csv
    python analysis/ablation_summary_to_latex.py results_aggregated.csv --output table.tex
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

SEQUENCE_RE = re.compile(r"^monohansett_(?P<scene>[a-zA-Z0-9]+)(?:_no_(?P<ablation>[a-zA-Z0-9]+))?$")

SCENE_DISPLAY = {
    "boiler": "Boiler",
    "engine": "Engine",
    "long": "Long",
    "start": "Engine",
}

ABLATION_DISPLAY = {
    "dvl": "No DVL",
    "barometer": "No Barometer",
    None: "Full",
}

#: For --style checkmarks: the sensor column each ablation turns off, in table order.
SENSOR_COLUMNS = ["Camera", "DVL", "Baro."]
ABLATION_DISABLES = {
    "Full": set(),
    "No DVL": {"DVL"},
    "No Barometer": {"Baro."},
    "No Camera": {"Camera"},
}

#: Label given to rows pulled in by --no-camera-algorithm.
NO_CAMERA = "No Camera"

DEFAULT_CHECKMARK_CAPTION = (
    "Ablation results for disabling individual sensors from the SLAM pipeline. Disabling the "
    "camera corresponds to running TURTLMap \\cite{song2024turtlmap}. The reported metric is "
    "RMS APE; each result is the median of five runs."
)


def parse_sequence(name: str) -> Optional[Tuple[str, str]]:
    m = SEQUENCE_RE.match(name)
    if not m:
        return None
    scene = SCENE_DISPLAY.get(m.group("scene"), m.group("scene").capitalize())
    ablation = ABLATION_DISPLAY.get(m.group("ablation"), f"No {m.group('ablation')}")
    return scene, ablation


def fmt_mean_std(mean: float, std: float, precision: int) -> str:
    if pd.isna(mean) or np.isinf(mean):
        return "--"
    if pd.isna(std) or np.isinf(std):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def emit(lines, output) -> None:
    """Write the table to `output`, or stdout when it is None."""
    latex = "\n".join(lines)
    if output:
        Path(output).write_text(latex)
        print(f"LaTeX table saved to: {output}", file=sys.stderr)
    else:
        print(latex)


def build_checkmark_table(df, scenes, ablations, precision, caption, label,
                          statistic="mean") -> list:
    """Sensor combinations as rows, scenes as columns.

    The unablated ("Full") condition is emitted last, below a rule, mirroring how
    analysis/results_to_latex.py separates "Ours" from the baselines. Best value per
    scene is bold, second best underlined; ties for best are all bold.
    """
    ape_col = f"ape_rmse_{statistic}"

    def value_for(scene, ablation):
        match = df[(df["scene"] == scene) & (df["ablation"] == ablation)]
        if match.empty:
            return np.nan
        v = match.iloc[0].get(ape_col, np.nan)
        return np.nan if (pd.isna(v) or np.isinf(v)) else float(v)

    rows = [a for a in ablations if a != "Full"]
    if "Full" in ablations:
        rows.append("Full")

    # Rank per scene so bold/underline reflect the whole column.
    best, second = {}, {}
    for scene in scenes:
        vals = {a: value_for(scene, a) for a in rows}
        vals = {a: v for a, v in vals.items() if not pd.isna(v)}
        if not vals:
            continue
        ordered = sorted(set(vals.values()))
        best[scene] = {a for a, v in vals.items() if v == ordered[0]}
        if len(ordered) > 1:
            second[scene] = {a for a, v in vals.items() if v == ordered[1]}

    lines = [
        "% \\usepackage{pifont} for \\ding; define once in your preamble:",
        "% \\newcommand{\\cmark}{\\ding{51}}",
        "% \\newcommand{\\xmark}{\\ding{55}}",
        "\\begin{table}",
        f"\\caption{{{caption}}}",
        "\\centering",
        f"\\begin{{tabular}}{{{'c' * len(SENSOR_COLUMNS)}||{'c' * len(scenes)}}}",
        "\\toprule",
        " & ".join(SENSOR_COLUMNS + scenes) + " \\\\",
        "\\midrule",
    ]

    for ablation in rows:
        disabled = ABLATION_DISABLES.get(ablation)
        if disabled is None:
            print(f"warning: no sensor mapping for {ablation!r}; skipping row "
                  "(add it to ABLATION_DISABLES)", file=sys.stderr)
            continue
        if ablation == "Full" and len(rows) > 1:
            lines.append("\\midrule")
        cells = ["\\xmark" if col in disabled else "\\cmark" for col in SENSOR_COLUMNS]
        for scene in scenes:
            v = value_for(scene, ablation)
            if pd.isna(v):
                cells.append("--")
            elif ablation in best.get(scene, ()):
                cells.append(f"\\textbf{{{v:.{precision}f}}}")
            elif ablation in second.get(scene, ()):
                cells.append(f"\\underline{{{v:.{precision}f}}}")
            else:
                cells.append(f"{v:.{precision}f}")
        lines.append(" & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", f"\\label{{{label}}}", "\\end{table}"]
    return lines


def main():
    parser = argparse.ArgumentParser(description="Convert ablation results CSV to a LaTeX table")
    parser.add_argument("results_csv", help="Path to results_aggregated.csv (or summary.csv)")
    parser.add_argument("--output", "-o", default=None, help="Output .tex file (default: print to stdout)")
    parser.add_argument("--precision", "-p", type=int, default=3, help="Decimal places for APE/RPE (default: 3)")
    parser.add_argument("--scene-order", nargs="+", default=["Engine", "Boiler", "Long"],
                        help="Scene column order. Defaults to the same order as "
                             "analysis/results_to_latex.py so the paper's two tables agree; "
                             "they previously disagreed, which transposed copied values.")
    parser.add_argument("--statistic", choices=["mean", "median"], default="mean",
                        help="Which across-trials statistic to tabulate. They agree to ~2%% on "
                             "converged runs but diverge on unstable ones, so it applies "
                             "table-wide rather than per-row.")
    parser.add_argument("--style", choices=["scenes", "checkmarks"], default="scenes",
                        help="scenes: scenes as rows, ablations as grouped columns (default). "
                             "checkmarks: sensor toggles as rows, scenes as columns.")
    parser.add_argument("--no-camera-algorithm", default=None,
                        help="Algorithm whose un-ablated runs supply the camera-off row "
                             "(e.g. turtlmap, which sets areCamsUsed false). Its baseline "
                             "sequences become the 'No Camera' ablation.")
    parser.add_argument("--algorithm", default=None,
                        help="Which algorithm column to tabulate. Required when the CSV holds more "
                             "than one, since a scene/ablation cell would otherwise be ambiguous.")
    parser.add_argument("--ablation-order", nargs="+",
                        default=["Full", "No Barometer", "No DVL", "No Camera"],
                         help="Column order for ablation conditions")
    parser.add_argument("--caption", default=None,
                        help="Table caption. Defaults per --style, since the checkmark layout "
                             "describes sensors while the scenes layout describes conditions.")
    parser.add_argument("--label", default="tab:ablation_results", help="LaTeX label")
    args = parser.parse_args()

    if args.caption is None:
        args.caption = (DEFAULT_CHECKMARK_CAPTION if args.style == "checkmarks"
                        else "DVL / barometer ablation results")

    print(f"Loading results from: {args.results_csv}", file=sys.stderr)
    df = pd.read_csv(args.results_csv)

    # Each (scene, ablation) cell must resolve to exactly one row. The CSV carries one row
    # per algorithm, so without filtering the table silently reports whichever algorithm
    # happens to sort first -- and would print NaN for an algorithm that skipped these
    # sequences. Make the choice explicit instead.
    # Held aside before the single-algorithm filter below, then relabelled as an ablation.
    no_camera_rows = None
    if args.no_camera_algorithm is not None:
        if "algorithm" not in df.columns:
            parser.error("--no-camera-algorithm needs an 'algorithm' column in the CSV")
        no_camera_rows = df[df["algorithm"] == args.no_camera_algorithm].copy()
        if no_camera_rows.empty:
            parser.error(
                f"--no-camera-algorithm {args.no_camera_algorithm!r} matched no rows. "
                f"Available: {', '.join(sorted(df['algorithm'].dropna().unique()))}")

    if "algorithm" in df.columns:
        available = sorted(df["algorithm"].dropna().unique())
        if args.algorithm is not None:
            if args.algorithm not in available:
                parser.error(f"--algorithm {args.algorithm!r} not in CSV. Available: {', '.join(available)}")
            df = df[df["algorithm"] == args.algorithm].copy()
        elif len(available) > 1:
            parser.error(
                f"{args.results_csv} contains {len(available)} algorithms "
                f"({', '.join(available)}); pass --algorithm to choose one.")

    parsed = df["sequence"].apply(parse_sequence)
    df = df[parsed.notna()].copy()
    df["scene"], df["ablation"] = zip(*parsed[parsed.notna()])

    if no_camera_rows is not None:
        nc_parsed = no_camera_rows["sequence"].apply(parse_sequence)
        no_camera_rows = no_camera_rows[nc_parsed.notna()].copy()
        no_camera_rows["scene"], no_camera_rows["ablation"] = zip(*nc_parsed[nc_parsed.notna()])
        # Only its un-ablated runs are meaningful here: that algorithm is itself the
        # camera-off condition, so its own DVL/barometer ablations would be a different
        # (two-sensor) cell than this table's one-sensor-at-a-time rows.
        no_camera_rows = no_camera_rows[no_camera_rows["ablation"] == "Full"].copy()
        if no_camera_rows.empty:
            parser.error(
                f"--no-camera-algorithm {args.no_camera_algorithm!r} has no un-ablated "
                "(monohansett_<scene>) rows to draw from")
        no_camera_rows["ablation"] = NO_CAMERA
        df = pd.concat([df, no_camera_rows], ignore_index=True)

    # Explicit, never alphabetical: when this table's column order silently disagreed with
    # analysis/results_to_latex.py, numbers copied between the two tables got transposed.
    scene_order = args.scene_order
    present = set(df["scene"].unique())
    scenes = [s for s in scene_order if s in present]
    scenes += sorted(s for s in present if s not in scene_order)
    ablations = [a for a in args.ablation_order if a in set(df["ablation"])]
    ablations += [a for a in df["ablation"].unique() if a not in ablations]

    if args.style == "checkmarks":
        lines = build_checkmark_table(df, scenes, ablations, args.precision,
                                      args.caption, args.label, args.statistic)
        emit(lines, args.output)
        return

    lines = []
    lines.append("\\begin{table}")
    lines.append("\\centering")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    col_spec = "l||" + "|".join(["cc"] * len(ablations))
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    header1 = "\\multirow{2}{*}{Scene}"
    for ablation in ablations:
        header1 += f" & \\multicolumn{{2}}{{c}}{{{ablation}}}"
    header1 += " \\\\"
    lines.append(header1)

    header2 = "           "
    for _ in ablations:
        header2 += " & APE RMSE (m) & Completeness (\\%)"
    header2 += " \\\\\\hline"
    lines.append(header2)

    for scene in scenes:
        row_values = []
        for ablation in ablations:
            match = df[(df["scene"] == scene) & (df["ablation"] == ablation)]
            if match.empty:
                row_values.append("--")
                row_values.append("--")
                continue
            row = match.iloc[0]
            ape = fmt_mean_std(row.get(f"ape_rmse_{args.statistic}", np.nan),
                               row.get("ape_rmse_std", np.nan), args.precision)
            completeness_mean = row.get(f"completeness_{args.statistic}", np.nan)
            completeness = "--" if pd.isna(completeness_mean) else f"{completeness_mean * 100:.1f}"
            row_values.append(ape)
            row_values.append(completeness)
        row_str = f"{scene} & " + " & ".join(row_values) + " \\\\"
        lines.append(row_str)

    lines.append("\\hline")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(f"\\caption{{{args.caption}}}")
    lines.append(f"\\label{{{args.label}}}")
    lines.append("\\end{table}")

    emit(lines, args.output)


if __name__ == "__main__":
    main()
