"""Analyze TD3 UTD ratio study results from WandB.

This script fetches results from the stoix_td3_utd_study WandB project
and computes mean ± std per (environment, UTD) configuration.
Also identifies the best UTD ratio per environment.

Run naming convention: {algo}_{env}_d{depth}_{activation}_utd{utd}_{timestamp}_s{seed}
Example: td3_halfcheetah_d2_silu_utd16_20240118_120000_s42

Usage:
    python experiments/analyze_td3_utd_study.py
    python experiments/analyze_td3_utd_study.py --project stoix_td3_utd_study --latex
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

    Expected format: {algo}_{env}_d{depth}_{activation}_utd{utd}_{timestamp}_s{seed}
    Example: td3_halfcheetah_d2_silu_utd16_20240118_120000_s42

    Also supports legacy format: td3_utd{U}_{env}_{timestamp}
    """
    # Try new format first: {algo}_{env}_d{depth}_{activation}_utd{utd}_{timestamp}_s{seed}
    new_pattern = r'^(\w+)_(\w+)_d(\d+)_(\w+)_utd(\d+)_(\d{8}_\d{6})_s(\d+)$'
    match = re.match(new_pattern, name)
    if match:
        return {
            'algo': match.group(1),
            'env': match.group(2),
            'depth': int(match.group(3)),
            'activation': match.group(4),
            'utd': int(match.group(5)),
            'timestamp': match.group(6),
            'seed': int(match.group(7)),
        }

    # Try legacy format: td3_utd{U}_{env}_{timestamp}
    legacy_pattern = r'^td3_utd(\d+)_(\w+)_(\d{8}_\d{6})$'
    match = re.match(legacy_pattern, name)
    if match:
        return {
            'algo': 'td3',
            'env': match.group(2),
            'depth': 2,  # default
            'activation': 'silu',  # default
            'utd': int(match.group(1)),
            'timestamp': match.group(3),
            'seed': None,
        }

    return None


def fetch_utd_study_results(
    project: str = "stoix_td3_utd_study",
    entity: str = None,
) -> pd.DataFrame:
    """Fetch all runs from the UTD study project."""
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
                "seed": parsed.get('seed'),
                "episode_return": episode_return,
                "episode_return_std": episode_return_std,
                "q_loss": summary.get("q_loss", None),
                "actor_loss": summary.get("actor_loss", None),
                "state": run.state,
            })

    return pd.DataFrame(data)


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std per (environment, UTD) configuration."""
    grouped = df.groupby(["env", "utd"])

    summary = []
    for (env, utd), group in grouped:
        returns = group["episode_return"].values

        summary.append({
            "env": env,
            "utd": utd,
            "depth": group["depth"].values[0],
            "activation": group["activation"].values[0],
            "mean_return": returns.mean(),
            "std_return": returns.std() if len(returns) > 1 else 0,
            "n_runs": len(returns),
            "min_return": returns.min(),
            "max_return": returns.max(),
        })

    return pd.DataFrame(summary)


def find_best_utd_per_env(summary_df: pd.DataFrame) -> dict:
    """Find the best UTD ratio for each environment."""
    best_utd = {}
    for env in summary_df["env"].unique():
        env_data = summary_df[summary_df["env"] == env]
        best_idx = env_data["mean_return"].idxmax()
        best_row = env_data.loc[best_idx]
        best_utd[env] = {
            "utd": int(best_row["utd"]),
            "mean_return": best_row["mean_return"],
            "std_return": best_row["std_return"],
        }
    return best_utd


def print_results_table(summary_df: pd.DataFrame):
    """Print a formatted table of results."""
    print("\n" + "=" * 120)
    print("TD3 UTD RATIO STUDY RESULTS - Finding Optimal Update-to-Data Ratio")
    print("=" * 120)

    envs = sorted(summary_df["env"].unique())
    utds = sorted(summary_df["utd"].unique())

    # Print metadata
    if len(summary_df) > 0:
        depth = summary_df["depth"].values[0]
        activation = summary_df["activation"].values[0]
        print(f"Depth: {depth} | Activation: {activation}")

    print(f"\n{'Environment':<20}", end="")
    for u in utds:
        print(f"{'UTD=' + str(u):<16}", end="")
    print("Best UTD")
    print("-" * (20 + 16 * len(utds) + 10))

    best_utd = find_best_utd_per_env(summary_df)

    for env in envs:
        print(f"{env:<20}", end="")
        env_data = summary_df[summary_df["env"] == env]
        max_mean = env_data["mean_return"].max()

        for utd in utds:
            row = summary_df[(summary_df["env"] == env) & (summary_df["utd"] == utd)]
            if len(row) > 0:
                mean = row["mean_return"].values[0]
                std = row["std_return"].values[0]
                # Highlight best UTD for this environment
                if mean == max_mean:
                    print(f"*{mean:>5.0f} ± {std:<4.0f}*", end=" ")
                else:
                    print(f"{mean:>6.0f} ± {std:<5.0f}", end=" ")
            else:
                print(f"{'N/A':<16}", end="")

        print(f"  {best_utd[env]['utd']}")

    print("-" * (20 + 16 * len(utds) + 10))
    print("\nNote: Values are mean ± std across 5 seeds. * indicates best UTD for that environment.")

    # Print summary of best UTDs
    print("\n" + "=" * 60)
    print("RECOMMENDED UTD RATIOS PER ENVIRONMENT")
    print("=" * 60)
    for env in sorted(best_utd.keys()):
        info = best_utd[env]
        print(f"  {env:<20}: UTD={info['utd']:<3} (return: {info['mean_return']:.0f} ± {info['std_return']:.0f})")

    # Check if there's a consensus
    utd_values = [info["utd"] for info in best_utd.values()]
    most_common_utd = max(set(utd_values), key=utd_values.count)
    count = utd_values.count(most_common_utd)
    print(f"\nMost common best UTD: {most_common_utd} ({count}/{len(utd_values)} environments)")


def export_to_latex(summary_df: pd.DataFrame, output_file: str = "td3_utd_study_table.tex"):
    """Export results to LaTeX table format."""
    envs = sorted(summary_df["env"].unique())
    utds = sorted(summary_df["utd"].unique())
    best_utd = find_best_utd_per_env(summary_df)

    with open(output_file, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{TD3 performance across UTD ratios (mean $\\pm$ std over 5 seeds). Bold indicates best UTD per environment.}\n")
        f.write("\\label{tab:td3_utd_study}\n")
        f.write("\\begin{tabular}{l" + "c" * len(utds) + "c}\n")
        f.write("\\toprule\n")

        f.write("Environment & " + " & ".join([f"UTD={u}" for u in utds]) + " & Best \\\\\n")
        f.write("\\midrule\n")

        for env in envs:
            row_data = [env.replace("_", "\\_")]
            env_data = summary_df[summary_df["env"] == env]
            max_mean = env_data["mean_return"].max()

            for utd in utds:
                row = summary_df[(summary_df["env"] == env) & (summary_df["utd"] == utd)]
                if len(row) > 0:
                    mean = row["mean_return"].values[0]
                    std = row["std_return"].values[0]
                    if mean == max_mean:
                        row_data.append(f"$\\mathbf{{{mean:.0f} \\pm {std:.0f}}}$")
                    else:
                        row_data.append(f"${mean:.0f} \\pm {std:.0f}$")
                else:
                    row_data.append("--")

            row_data.append(str(best_utd[env]["utd"]))
            f.write(" & ".join(row_data) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nLaTeX table exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze TD3 UTD ratio study results")
    parser.add_argument("--project", default="stoix_td3_utd_study", help="WandB project name")
    parser.add_argument("--entity", default=None, help="WandB entity")
    parser.add_argument("--latex", action="store_true", help="Export to LaTeX")
    parser.add_argument("--csv", type=str, default=None, help="Export to CSV file")
    args = parser.parse_args()

    print(f"Fetching results from WandB project: {args.project}")
    df = fetch_utd_study_results(args.project, args.entity)

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
