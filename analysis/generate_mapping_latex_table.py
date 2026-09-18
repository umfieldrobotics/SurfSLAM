#!/usr/bin/env python3
"""
Generate LaTeX table from mapping_summary.csv with formatting for best/second-best results.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def format_value(value, is_best, is_second_best):
    """Format a value with bold for best, underline for second best."""
    if pd.isna(value):
        return '-'

    # Reachable only if over half a method's runs failed, since failures are scored at the
    # worst attainable value (+inf for distances) before the median is taken.
    if np.isinf(value):
        return '\\multicolumn{1}{c}{---}'

    formatted = f"{value:.2f}"
    
    if is_best:
        return f"\\textbf{{{formatted}}}"
    elif is_second_best:
        return f"\\underline{{{formatted}}}"
    else:
        return formatted

def get_best_and_second_best(values, lower_is_better=True):
    """
    Identify best and second best indices.
    
    Args:
        values: List of values
        lower_is_better: If True, lower values are better; if False, higher is better
    
    Returns:
        Tuple of (best_idx, second_best_idx)
    """
    valid_values = [(i, v) for i, v in enumerate(values) if not pd.isna(v)]
    
    if len(valid_values) < 2:
        if len(valid_values) == 1:
            return valid_values[0][0], None
        return None, None
    
    # Sort by value
    if lower_is_better:
        sorted_values = sorted(valid_values, key=lambda x: x[1])
    else:
        sorted_values = sorted(valid_values, key=lambda x: x[1], reverse=True)
    
    best_idx = sorted_values[0][0]
    second_best_idx = sorted_values[1][0]
    
    return best_idx, second_best_idx

def generate_latex_table(summary_csv_path, output_tex_path):
    """
    Generate LaTeX table from mapping summary.
    
    Args:
        summary_csv_path: Path to mapping_summary.csv
        output_tex_path: Path to output .tex file
    """
    # Read the summary
    df = pd.read_csv(summary_csv_path)
    
    # Map CSV method names to internal keys. evaluate_maps.py names rows after the
    # algorithm entry in traj_eval_cfg_map.yaml, plus turtlmap_our_depths for the second
    # map TURTLMap contributes.
    method_csv_to_key = {
        'surfslam': '\\algname',
        'droidslam': 'Droid SLAM',
        'mast3r_slam': 'Mast3R SLAM',
        'svin2': 'SVin2',
        'turtlmap': 'Turtl Base',
        'turtlmap_our_depths': 'Turtl Ours',
    }

    # Scene display names. monohansett_start is the pre-8970b4b name for monohansett_engine,
    # kept so previously-generated CSVs still tabulate.
    scene_csv_to_display = {
        'monohansett_boiler': 'Boiler',
        'monohansett_engine': 'Engine',
        'monohansett_start': 'Engine',
        'monohansett_long': 'Long'
    }
    
    # Define metric order and properties
    metrics = [
        ('median_accuracy', 'Acc.', True),      # lower is better
        ('median_completeness', 'Comp.', True),  # lower is better
        ('median_precision', 'Prec.', False),    # higher is better
        ('median_recall', 'Rec.', False)         # higher is better
    ]
    
    # Method order for columns (exactly as in reference table)
    method_order = ['Droid SLAM', 'Mast3R SLAM', 'SVin2', 'Turtl Base', 'Turtl Ours', '\\algname']
    
    # Scene order for rows
    scene_order = ['Boiler', 'Engine', 'Long']
    
    # Build the table content
    table_lines = []
    
    # Add table header - EXACT replica of reference
    table_lines.append("\\begin{table}[t]")
    table_lines.append("\\centering")
    table_lines.append("\\caption{Maps were evaluated using photogrammetry ground-truth. The Turtl Base column indicates accumulating depths from the TURTLMap trajectory estimates using a baseline DEFOM ViT-S stereo depth estimator. In contrast, Turtl Ours indicates using the TURTLMap trajectories and our proposed DEFOM ViT-S stereo depth estimator. TURTLMap's mapper is deterministic, so both Turtl columns are a single run; every other entry is the median of five runs, where a run that failed to produce a reconstruction is scored at the worst attainable value rather than excluded. For accuracy and completion, lower is better. For precision and recall, higher is better.}")
    table_lines.append("\\setlength{\\tabcolsep}{3pt}")
    table_lines.append("\\resizebox{\\columnwidth}{!}{%")
    table_lines.append("\\scriptsize")
    table_lines.append("\\begin{tabular}{c l | c c c c | c c}")
    
    # Add column headers - EXACT replica
    table_lines.append(" &  & Droid & Mast3R & \\multirow{2}{*}{SVin2} & Turtl & Turtl & \\multirow{2}{*}{\\algname} \\\\[-0.2em]")
    table_lines.append(" &  & SLAM & SLAM &  & Base & Ours &  \\\\")
    table_lines.append("\\hline")
    
    # Scene names in the CSV collapse to display names (start and engine are the same
    # scene), so group on the display name rather than iterating the alias map.
    df = df.copy()
    df['scene_display'] = df['scene'].map(scene_csv_to_display)

    unmapped = sorted(set(df.loc[df['scene_display'].isna(), 'scene']))
    if unmapped:
        print(f"warning: ignoring unrecognized scenes {unmapped}")

    # Process each scene
    for scene_display in scene_order:
        # Filter data for this scene
        scene_data = df[df['scene_display'] == scene_display].copy()

        # Create a pivot table for easier access
        scene_pivot = {}
        for _, row in scene_data.iterrows():
            method_key = method_csv_to_key.get(row['method'], row['method'])
            scene_pivot[method_key] = {
                'median_accuracy': row['median_accuracy'],
                'median_completeness': row['median_completeness'],
                'median_precision': row['median_precision'],
                'median_recall': row['median_recall']
            }
        
        # Add multirow for scene name
        table_lines.append(f"\\multirow{{4}}{{*}}{{\\rotatebox{{90}}{{{scene_display}}}}}")
        
        # Add rows for each metric
        for metric_col, metric_label, lower_is_better in metrics:
            # Collect values for all methods
            values = []
            for method in method_order:
                if method in scene_pivot:
                    values.append(scene_pivot[method][metric_col])
                else:
                    values.append(np.nan)
            
            # Find best and second best
            best_idx, second_best_idx = get_best_and_second_best(values, lower_is_better)
            
            # Format values
            formatted_values = []
            for i, value in enumerate(values):
                is_best = (i == best_idx)
                is_second_best = (i == second_best_idx)
                formatted_values.append(format_value(value, is_best, is_second_best))
            
            # Build the row
            row = f" & {metric_label}  & {' & '.join(formatted_values)} \\\\"
            table_lines.append(row)
        
        table_lines.append("\\hline")
    
    # Close the table
    table_lines.append("\\end{tabular}%")
    table_lines.append("}")
    table_lines.append("\\vspace{0.2cm}")
    table_lines.append("\\label{tab:mapping}")
    table_lines.append("\\end{table}")
    
    # Write to file
    with open(output_tex_path, 'w') as f:
        f.write('\n'.join(table_lines))
    
    print(f"LaTeX table saved to {output_tex_path}")
    print("\nTable preview:")
    print('\n'.join(table_lines))
    
    return table_lines

if __name__ == "__main__":
    import argparse

    default_dir = Path(__file__).resolve().parents[1] / "eval_results" / "map_comparison" / "maps"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=default_dir / "mapping_summary.csv",
                        help="mapping_summary.csv from compute_mapping_summary.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to table_mapping.tex at the repo root")
    args = parser.parse_args()

    output = args.output or Path(__file__).resolve().parents[1] / "table_mapping.tex"
    generate_latex_table(args.summary, output)
