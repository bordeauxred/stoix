"""Plot TD3 UTD ratio study results with mean ± std.

Creates publication-quality plots showing performance vs UTD ratio.

Usage:
    python experiments/plot_utd_study.py --project stoix_td3_utd_study
    python experiments/plot_utd_study.py --project stoix_td3_utd_study --output plots/td3_utd.pdf
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
except ImportError:
    print("Please install matplotlib: pip install matplotlib")
    exit(1)

try:
    import wandb
except ImportError:
    print("Please install wandb: pip install wandb")
    exit(1)


# Color palette (colorblind-friendly)
COLORS = {
    'halfcheetah': '#1f77b4',
    'ant': '#ff7f0e',
    'hopper': '#2ca02c',
    'pendulum': '#d62728',
    'reacher': '#9467bd',
    'pusher': '#8c564b',
}


def fetch_utd_results(project: str, entity: str = None) -> pd.DataFrame:
    """Fetch all runs from the UTD study project."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}" if entity else project)

    data = []
    for run in runs:
        name = run.name
        if not name.startswith("td3_utd"):
            continue

        parts = name.split("_")
        try:
            utd = int(parts[1].replace("utd", ""))
            env_parts = parts[2:-2]
            env = "_".join(env_parts)
        except (IndexError, ValueError):
            print(f"Skipping malformed run name: {name}")
            continue

        summary = run.summary
        episode_return = summary.get("episode_return", None)

        if episode_return is not None:
            data.append({
                "run_name": name,
                "env": env,
                "utd": utd,
                "episode_return": episode_return,
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
            "mean_return": returns.mean(),
            "std_return": returns.std() if len(returns) > 1 else 0,
            "sem_return": returns.std() / np.sqrt(len(returns)) if len(returns) > 1 else 0,
            "n_runs": len(returns),
        })

    return pd.DataFrame(summary)


def plot_utd_comparison(summary_df: pd.DataFrame, output_path: str, title: str = "UTD Ratio Study"):
    """Create a multi-panel plot showing performance vs UTD for each environment."""
    envs = sorted(summary_df["env"].unique())
    n_envs = len(envs)

    n_cols = min(3, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_envs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, env in enumerate(envs):
        ax = axes[idx]
        env_data = summary_df[summary_df["env"] == env].sort_values("utd")

        utds = env_data["utd"].values
        means = env_data["mean_return"].values
        stds = env_data["std_return"].values

        color = COLORS.get(env, '#333333')

        # Find best UTD for this environment
        best_idx = np.argmax(means)
        best_utd = utds[best_idx]

        # Plot mean with std shading
        ax.plot(utds, means, 'o-', color=color, linewidth=2, markersize=6, label=env)
        ax.fill_between(utds, means - stds, means + stds, color=color, alpha=0.2)

        # Highlight best UTD
        ax.axvline(x=best_utd, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
        ax.scatter([best_utd], [means[best_idx]], color=color, s=100, zorder=5,
                  edgecolors='black', linewidths=1.5, marker='*')

        ax.set_xlabel("UTD Ratio (epochs)", fontsize=10)
        ax.set_ylabel("Episode Return", fontsize=10)
        ax.set_title(f"{env.replace('_', ' ').title()} (best: UTD={best_utd})",
                    fontsize=11, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_xticks(utds)
        ax.set_xticklabels([str(u) for u in utds])
        ax.grid(True, alpha=0.3)

        n_seeds = env_data["n_runs"].values[0]
        ax.text(0.98, 0.02, f"n={n_seeds}", transform=ax.transAxes,
                fontsize=8, ha='right', va='bottom', color='gray')

    for idx in range(n_envs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_combined_utd(summary_df: pd.DataFrame, output_path: str):
    """Create a single plot with all environments showing normalized performance."""
    envs = sorted(summary_df["env"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    best_utds = {}
    for env in envs:
        env_data = summary_df[summary_df["env"] == env].sort_values("utd")

        utds = env_data["utd"].values
        means = env_data["mean_return"].values
        stds = env_data["std_return"].values

        # Normalize
        min_val, max_val = means.min(), means.max()
        if max_val - min_val > 0:
            norm_means = (means - min_val) / (max_val - min_val)
            norm_stds = stds / (max_val - min_val)
        else:
            norm_means = np.ones_like(means)
            norm_stds = np.zeros_like(stds)

        color = COLORS.get(env, '#333333')
        best_utds[env] = utds[np.argmax(means)]

        ax.plot(utds, norm_means, 'o-', color=color, linewidth=2, markersize=5,
                label=f"{env.replace('_', ' ').title()} (best: {best_utds[env]})")
        ax.fill_between(utds, norm_means - norm_stds, norm_means + norm_stds,
                       color=color, alpha=0.15)

    ax.set_xlabel("UTD Ratio (epochs)", fontsize=12)
    ax.set_ylabel("Normalized Return (0=worst, 1=best)", fontsize=12)
    ax.set_title("TD3 Performance vs UTD Ratio (Normalized)", fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted(summary_df["utd"].unique()))
    ax.set_xticklabels([str(u) for u in sorted(summary_df["utd"].unique())])
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()

    base, ext = os.path.splitext(output_path)
    norm_path = f"{base}_normalized{ext}"
    plt.savefig(norm_path, dpi=300, bbox_inches='tight')
    print(f"Saved normalized plot to {norm_path}")
    plt.close()


def plot_best_utd_summary(summary_df: pd.DataFrame, output_path: str):
    """Create a bar chart showing best UTD per environment."""
    envs = sorted(summary_df["env"].unique())

    best_data = []
    for env in envs:
        env_data = summary_df[summary_df["env"] == env]
        best_idx = env_data["mean_return"].idxmax()
        best_row = env_data.loc[best_idx]
        best_data.append({
            "env": env,
            "best_utd": best_row["utd"],
            "mean_return": best_row["mean_return"],
            "std_return": best_row["std_return"],
        })

    best_df = pd.DataFrame(best_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart of best UTDs
    colors = [COLORS.get(env, '#333333') for env in best_df["env"]]
    bars = ax1.bar(range(len(envs)), best_df["best_utd"], color=colors, edgecolor='black', linewidth=1)

    ax1.set_xticks(range(len(envs)))
    ax1.set_xticklabels([e.replace("_", " ").title() for e in best_df["env"]], rotation=45, ha='right')
    ax1.set_ylabel("Best UTD Ratio", fontsize=12)
    ax1.set_title("Optimal UTD Ratio per Environment", fontsize=12, fontweight='bold')
    ax1.set_yscale('log', base=2)
    ax1.set_yticks([1, 2, 4, 8, 16, 32])
    ax1.set_yticklabels(['1', '2', '4', '8', '16', '32'])
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, utd in zip(bars, best_df["best_utd"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{int(utd)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Histogram of best UTDs
    utd_counts = best_df["best_utd"].value_counts().sort_index()
    ax2.bar(range(len(utd_counts)), utd_counts.values, color='steelblue', edgecolor='black', linewidth=1)
    ax2.set_xticks(range(len(utd_counts)))
    ax2.set_xticklabels([str(int(u)) for u in utd_counts.index])
    ax2.set_xlabel("UTD Ratio", fontsize=12)
    ax2.set_ylabel("Number of Environments", fontsize=12)
    ax2.set_title("Distribution of Optimal UTD Ratios", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    base, ext = os.path.splitext(output_path)
    summary_path = f"{base}_best_utd{ext}"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Saved best UTD summary to {summary_path}")
    plt.close()

    # Print text summary
    print("\n" + "=" * 50)
    print("BEST UTD RATIO PER ENVIRONMENT")
    print("=" * 50)
    for _, row in best_df.iterrows():
        print(f"  {row['env']:<20}: UTD={int(row['best_utd']):<3} "
              f"(return: {row['mean_return']:.0f} ± {row['std_return']:.0f})")

    most_common = best_df["best_utd"].mode().values[0]
    count = (best_df["best_utd"] == most_common).sum()
    print(f"\nRecommended default UTD: {int(most_common)} ({count}/{len(envs)} environments)")


def main():
    parser = argparse.ArgumentParser(description="Plot UTD ratio study results")
    parser.add_argument("--project", default="stoix_td3_utd_study", help="WandB project name")
    parser.add_argument("--entity", default=None, help="WandB entity")
    parser.add_argument("--output", default="plots/td3_utd_study.pdf", help="Output path for plots")
    args = parser.parse_args()

    print(f"Fetching results from WandB project: {args.project}")
    df = fetch_utd_results(args.project, args.entity)

    if len(df) == 0:
        print("No completed runs found!")
        return

    print(f"Found {len(df)} runs")

    summary_df = compute_summary_stats(df)
    print("\nSummary statistics:")
    print(summary_df.to_string(index=False))

    # Create all plots
    plot_utd_comparison(summary_df, args.output, title="TD3 Performance vs UTD Ratio")
    plot_combined_utd(summary_df, args.output)
    plot_best_utd_summary(summary_df, args.output)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
