"""Analyze DQN depth study results from WandB.

This script fetches results from the stoix_dqn_depth_study WandB project
and computes mean ± std per (environment, depth) configuration.

Usage:
    python experiments/analyze_depth_study.py
    python experiments/analyze_depth_study.py --project stoix_dqn_depth_study
"""

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

try:
    import wandb
except ImportError:
    print("Please install wandb: pip install wandb")
    exit(1)


def fetch_depth_study_results(
    project: str = "stoix_dqn_depth_study",
    entity: str = None,
) -> pd.DataFrame:
    """Fetch all runs from the depth study project."""
    api = wandb.Api()

    # Get all runs
    runs = api.runs(f"{entity}/{project}" if entity else project)

    data = []
    for run in runs:
        # Parse run name: dqn_depth{D}_{env}_{timestamp}
        name = run.name
        if not name.startswith("dqn_depth"):
            continue

        # Extract depth and env from name
        parts = name.split("_")
        try:
            depth = int(parts[1].replace("depth", ""))
            # Environment is everything between depth and timestamp
            env_parts = parts[2:-2]  # Skip depth prefix and timestamp
            env = "_".join(env_parts)
        except (IndexError, ValueError):
            print(f"Skipping malformed run name: {name}")
            continue

        # Get final metrics
        summary = run.summary

        # Extract per-seed returns if available
        seed_returns = []
        for key in summary.keys():
            if key.startswith("seed_") and "episode_return" in key:
                seed_returns.append(summary[key])

        # Get aggregate metrics
        episode_return = summary.get("episode_return", None)
        episode_return_std = summary.get("episode_return_std", None)

        if episode_return is not None:
            data.append({
                "run_name": name,
                "env": env,
                "depth": depth,
                "episode_return": episode_return,
                "episode_return_std": episode_return_std,
                "seed_returns": seed_returns,
                "n_seeds": len(seed_returns) if seed_returns else 1,
                "state": run.state,
            })

    return pd.DataFrame(data)


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std per (environment, depth) configuration."""
    # Group by env and depth
    grouped = df.groupby(["env", "depth"])

    summary = []
    for (env, depth), group in grouped:
        returns = group["episode_return"].values

        summary.append({
            "env": env,
            "depth": depth,
            "mean_return": returns.mean(),
            "std_return": returns.std(),
            "n_runs": len(returns),
            "min_return": returns.min(),
            "max_return": returns.max(),
        })

    return pd.DataFrame(summary)


def print_results_table(summary_df: pd.DataFrame):
    """Print a formatted table of results."""
    print("\n" + "=" * 80)
    print("DQN DEPTH STUDY RESULTS - Baseline for Isometric Network Comparison")
    print("=" * 80)

    # Pivot table: environments as rows, depths as columns
    envs = sorted(summary_df["env"].unique())
    depths = sorted(summary_df["depth"].unique())

    # Header
    print(f"\n{'Environment':<20}", end="")
    for d in depths:
        print(f"{'D=' + str(d):<18}", end="")
    print()
    print("-" * (20 + 18 * len(depths)))

    # Data rows
    for env in envs:
        print(f"{env:<20}", end="")
        for depth in depths:
            row = summary_df[(summary_df["env"] == env) & (summary_df["depth"] == depth)]
            if len(row) > 0:
                mean = row["mean_return"].values[0]
                std = row["std_return"].values[0]
                print(f"{mean:>7.1f} ± {std:<6.1f}", end="  ")
            else:
                print(f"{'N/A':<18}", end="")
        print()

    print("-" * (20 + 18 * len(depths)))
    print("\nNote: Values are mean ± std across 5 seeds")


def export_to_latex(summary_df: pd.DataFrame, output_file: str = "depth_study_table.tex"):
    """Export results to LaTeX table format."""
    envs = sorted(summary_df["env"].unique())
    depths = sorted(summary_df["depth"].unique())

    with open(output_file, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{DQN performance across network depths (mean $\\pm$ std over 5 seeds)}\n")
        f.write("\\label{tab:dqn_depth_study}\n")
        f.write("\\begin{tabular}{l" + "c" * len(depths) + "}\n")
        f.write("\\toprule\n")

        # Header
        f.write("Environment & " + " & ".join([f"D={d}" for d in depths]) + " \\\\\n")
        f.write("\\midrule\n")

        # Data rows
        for env in envs:
            row_data = [env.replace("_", "\\_")]
            for depth in depths:
                row = summary_df[(summary_df["env"] == env) & (summary_df["depth"] == depth)]
                if len(row) > 0:
                    mean = row["mean_return"].values[0]
                    std = row["std_return"].values[0]
                    row_data.append(f"${mean:.1f} \\pm {std:.1f}$")
                else:
                    row_data.append("--")
            f.write(" & ".join(row_data) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nLaTeX table exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze DQN depth study results")
    parser.add_argument("--project", default="stoix_dqn_depth_study", help="WandB project name")
    parser.add_argument("--entity", default=None, help="WandB entity (username or team)")
    parser.add_argument("--latex", action="store_true", help="Export to LaTeX")
    parser.add_argument("--csv", type=str, default=None, help="Export to CSV file")
    args = parser.parse_args()

    print(f"Fetching results from WandB project: {args.project}")
    df = fetch_depth_study_results(args.project, args.entity)

    if len(df) == 0:
        print("No completed runs found!")
        return

    print(f"Found {len(df)} runs")

    # Compute summary statistics
    summary_df = compute_summary_stats(df)

    # Print results
    print_results_table(summary_df)

    # Export if requested
    if args.latex:
        export_to_latex(summary_df)

    if args.csv:
        summary_df.to_csv(args.csv, index=False)
        print(f"\nCSV exported to {args.csv}")


if __name__ == "__main__":
    main()
