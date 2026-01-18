"""Analyze TD3 depth study results from WandB.

This script fetches results from the stoix_td3_depth_study WandB project
and computes mean ± std per (environment, depth) configuration.

Run naming convention: {algo}_{env}_d{depth}_{activation}_utd{utd}_s{seeds}_{timestamp}
Example: td3_halfcheetah_d8_silu_utd16_s42-46_20240118_120000

Usage:
    python experiments/analyze_td3_depth_study.py
    python experiments/analyze_td3_depth_study.py --project stoix_td3_depth_study --latex
"""

import argparse
import re
from collections import defaultdict

import pandas as pd
import numpy as np

try:
    import wandb
except ImportError:
    print("Please install wandb: pip install wandb")
    exit(1)


def parse_run_name(name: str) -> dict | None:
    """Parse run name into components.

    Expected format: {algo}_{env}_d{depth}_{activation}_utd{utd}_s{seeds}_{timestamp}
    Example: td3_halfcheetah_d8_silu_utd16_s42-46_20240118_120000

    Also supports legacy format: td3_depth{D}_{env}_{timestamp}
    """
    # Try new format first
    # Pattern: algo_env_d{depth}_{activation}_utd{utd}_s{seeds}_{timestamp}
    new_pattern = r'^(\w+)_(\w+)_d(\d+)_(\w+)_utd(\d+)_s(\d+)-(\d+)_(\d{8}_\d{6})$'
    match = re.match(new_pattern, name)
    if match:
        return {
            'algo': match.group(1),
            'env': match.group(2),
            'depth': int(match.group(3)),
            'activation': match.group(4),
            'utd': int(match.group(5)),
            'seed_start': int(match.group(6)),
            'seed_end': int(match.group(7)),
            'timestamp': match.group(8),
        }

    # Try legacy format: td3_depth{D}_{env}_{timestamp}
    legacy_pattern = r'^td3_depth(\d+)_(\w+)_(\d{8}_\d{6})$'
    match = re.match(legacy_pattern, name)
    if match:
        return {
            'algo': 'td3',
            'env': match.group(2),
            'depth': int(match.group(1)),
            'activation': 'silu',  # default
            'utd': 16,  # default
            'seed_start': 42,
            'seed_end': 46,
            'timestamp': match.group(3),
        }

    return None


def fetch_depth_study_results(
    project: str = "stoix_td3_depth_study",
    entity: str = None,
) -> pd.DataFrame:
    """Fetch all runs from the depth study project."""
    api = wandb.Api()

    runs = api.runs(f"{entity}/{project}" if entity else project)

    data = []
    for run in runs:
        name = run.name
        parsed = parse_run_name(name)

        if parsed is None:
            print(f"Skipping malformed run name: {name}")
            continue

        summary = run.summary
        episode_return = summary.get("episode_return", None)
        episode_return_std = summary.get("episode_return_std", None)

        if episode_return is not None:
            data.append({
                "run_name": name,
                "algo": parsed['algo'],
                "env": parsed['env'],
                "depth": parsed['depth'],
                "activation": parsed['activation'],
                "utd": parsed['utd'],
                "seeds": f"{parsed['seed_start']}-{parsed['seed_end']}",
                "episode_return": episode_return,
                "episode_return_std": episode_return_std,
                "q_loss": summary.get("q_loss", None),
                "actor_loss": summary.get("actor_loss", None),
                "state": run.state,
            })

    return pd.DataFrame(data)


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std per (environment, depth) configuration."""
    grouped = df.groupby(["env", "depth"])

    summary = []
    for (env, depth), group in grouped:
        returns = group["episode_return"].values

        summary.append({
            "env": env,
            "depth": depth,
            "activation": group["activation"].values[0],
            "utd": group["utd"].values[0],
            "mean_return": returns.mean(),
            "std_return": returns.std() if len(returns) > 1 else 0,
            "n_runs": len(returns),
            "min_return": returns.min(),
            "max_return": returns.max(),
        })

    return pd.DataFrame(summary)


def print_results_table(summary_df: pd.DataFrame):
    """Print a formatted table of results."""
    print("\n" + "=" * 100)
    print("TD3 DEPTH STUDY RESULTS - Baseline for Isometric Network Comparison")
    print("=" * 100)

    envs = sorted(summary_df["env"].unique())
    depths = sorted(summary_df["depth"].unique())

    # Print metadata
    if len(summary_df) > 0:
        activation = summary_df["activation"].values[0]
        utd = summary_df["utd"].values[0]
        print(f"Activation: {activation} | UTD: {utd}")

    print(f"\n{'Environment':<20}", end="")
    for d in depths:
        print(f"{'D=' + str(d):<16}", end="")
    print()
    print("-" * (20 + 16 * len(depths)))

    for env in envs:
        print(f"{env:<20}", end="")
        for depth in depths:
            row = summary_df[(summary_df["env"] == env) & (summary_df["depth"] == depth)]
            if len(row) > 0:
                mean = row["mean_return"].values[0]
                std = row["std_return"].values[0]
                print(f"{mean:>6.0f} ± {std:<5.0f}", end="  ")
            else:
                print(f"{'N/A':<16}", end="")
        print()

    print("-" * (20 + 16 * len(depths)))
    print("\nNote: Values are mean ± std across 5 seeds")


def export_to_latex(summary_df: pd.DataFrame, output_file: str = "td3_depth_study_table.tex"):
    """Export results to LaTeX table format."""
    envs = sorted(summary_df["env"].unique())
    depths = sorted(summary_df["depth"].unique())

    with open(output_file, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{TD3 performance across network depths (mean $\\pm$ std over 5 seeds)}\n")
        f.write("\\label{tab:td3_depth_study}\n")
        f.write("\\begin{tabular}{l" + "c" * len(depths) + "}\n")
        f.write("\\toprule\n")

        f.write("Environment & " + " & ".join([f"D={d}" for d in depths]) + " \\\\\n")
        f.write("\\midrule\n")

        for env in envs:
            row_data = [env.replace("_", "\\_")]
            for depth in depths:
                row = summary_df[(summary_df["env"] == env) & (summary_df["depth"] == depth)]
                if len(row) > 0:
                    mean = row["mean_return"].values[0]
                    std = row["std_return"].values[0]
                    row_data.append(f"${mean:.0f} \\pm {std:.0f}$")
                else:
                    row_data.append("--")
            f.write(" & ".join(row_data) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nLaTeX table exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze TD3 depth study results")
    parser.add_argument("--project", default="stoix_td3_depth_study", help="WandB project name")
    parser.add_argument("--entity", default=None, help="WandB entity")
    parser.add_argument("--latex", action="store_true", help="Export to LaTeX")
    parser.add_argument("--csv", type=str, default=None, help="Export to CSV file")
    args = parser.parse_args()

    print(f"Fetching results from WandB project: {args.project}")
    df = fetch_depth_study_results(args.project, args.entity)

    if len(df) == 0:
        print("No completed runs found!")
        return

    print(f"Found {len(df)} runs")

    summary_df = compute_summary_stats(df)
    print_results_table(summary_df)

    if args.latex:
        export_to_latex(summary_df)

    if args.csv:
        summary_df.to_csv(args.csv, index=False)
        print(f"\nCSV exported to {args.csv}")


if __name__ == "__main__":
    main()
