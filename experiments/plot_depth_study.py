"""Plot TD3/DQN depth study results with mean ± std.

Creates publication-quality plots showing performance vs network depth.

Usage:
    python experiments/plot_depth_study.py --project stoix_td3_depth_study
    python experiments/plot_depth_study.py --project stoix_dqn_depth_study --output plots/dqn_depth.pdf
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
    'breakout': '#1f77b4',
    'asterix': '#ff7f0e',
    'space_invaders': '#2ca02c',
    'freeway': '#d62728',
    'cartpole': '#9467bd',
    'acrobot': '#8c564b',
    'mountain_car': '#e377c2',
}


def fetch_study_results(project: str, entity: str = None, prefix: str = "td3_depth") -> pd.DataFrame:
    """Fetch all runs from a depth study project."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}" if entity else project)

    data = []
    for run in runs:
        name = run.name
        if not name.startswith(prefix):
            continue

        parts = name.split("_")
        try:
            # Parse depth from name like "td3_depth8_halfcheetah_20240101_120000"
            depth = int(parts[1].replace("depth", ""))
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
                "depth": depth,
                "episode_return": episode_return,
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
            "mean_return": returns.mean(),
            "std_return": returns.std() if len(returns) > 1 else 0,
            "sem_return": returns.std() / np.sqrt(len(returns)) if len(returns) > 1 else 0,
            "n_runs": len(returns),
        })

    return pd.DataFrame(summary)


def plot_depth_comparison(summary_df: pd.DataFrame, output_path: str, title: str = "Network Depth Study"):
    """Create a multi-panel plot showing performance vs depth for each environment."""
    envs = sorted(summary_df["env"].unique())
    n_envs = len(envs)

    # Determine grid layout
    n_cols = min(3, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_envs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, env in enumerate(envs):
        ax = axes[idx]
        env_data = summary_df[summary_df["env"] == env].sort_values("depth")

        depths = env_data["depth"].values
        means = env_data["mean_return"].values
        stds = env_data["std_return"].values

        color = COLORS.get(env, '#333333')

        # Plot mean with std shading
        ax.plot(depths, means, 'o-', color=color, linewidth=2, markersize=6, label=env)
        ax.fill_between(depths, means - stds, means + stds, color=color, alpha=0.2)

        ax.set_xlabel("Network Depth (layers)", fontsize=10)
        ax.set_ylabel("Episode Return", fontsize=10)
        ax.set_title(env.replace("_", " ").title(), fontsize=11, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_xticks(depths)
        ax.set_xticklabels([str(d) for d in depths])
        ax.grid(True, alpha=0.3)

        # Add sample size annotation
        n_seeds = env_data["n_runs"].values[0]
        ax.text(0.98, 0.02, f"n={n_seeds}", transform=ax.transAxes,
                fontsize=8, ha='right', va='bottom', color='gray')

    # Hide empty subplots
    for idx in range(n_envs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_combined_normalized(summary_df: pd.DataFrame, output_path: str, title: str = "Normalized Performance vs Depth"):
    """Create a single plot with all environments normalized to their best performance."""
    envs = sorted(summary_df["env"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for env in envs:
        env_data = summary_df[summary_df["env"] == env].sort_values("depth")

        depths = env_data["depth"].values
        means = env_data["mean_return"].values
        stds = env_data["std_return"].values

        # Normalize to [0, 1] range based on min/max for this environment
        min_val, max_val = means.min(), means.max()
        if max_val - min_val > 0:
            norm_means = (means - min_val) / (max_val - min_val)
            norm_stds = stds / (max_val - min_val)
        else:
            norm_means = np.ones_like(means)
            norm_stds = np.zeros_like(stds)

        color = COLORS.get(env, '#333333')

        ax.plot(depths, norm_means, 'o-', color=color, linewidth=2, markersize=5,
                label=env.replace("_", " ").title())
        ax.fill_between(depths, norm_means - norm_stds, norm_means + norm_stds,
                       color=color, alpha=0.15)

    ax.set_xlabel("Network Depth (layers)", fontsize=12)
    ax.set_ylabel("Normalized Return (0=worst, 1=best)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted(summary_df["depth"].unique()))
    ax.set_xticklabels([str(d) for d in sorted(summary_df["depth"].unique())])
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()

    # Save figure
    base, ext = os.path.splitext(output_path)
    norm_path = f"{base}_normalized{ext}"
    plt.savefig(norm_path, dpi=300, bbox_inches='tight')
    print(f"Saved normalized plot to {norm_path}")
    plt.close()


def plot_heatmap(summary_df: pd.DataFrame, output_path: str, title: str = "Depth Study Heatmap"):
    """Create a heatmap of normalized performance."""
    envs = sorted(summary_df["env"].unique())
    depths = sorted(summary_df["depth"].unique())

    # Create matrix of normalized returns
    matrix = np.zeros((len(envs), len(depths)))

    for i, env in enumerate(envs):
        env_data = summary_df[summary_df["env"] == env]
        means = []
        for d in depths:
            row = env_data[env_data["depth"] == d]
            if len(row) > 0:
                means.append(row["mean_return"].values[0])
            else:
                means.append(np.nan)
        means = np.array(means)

        # Normalize per environment
        min_val, max_val = np.nanmin(means), np.nanmax(means)
        if max_val - min_val > 0:
            matrix[i, :] = (means - min_val) / (max_val - min_val)
        else:
            matrix[i, :] = 1.0

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels([e.replace("_", " ").title() for e in envs])

    ax.set_xlabel("Network Depth (layers)", fontsize=12)
    ax.set_ylabel("Environment", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Return", fontsize=10)

    # Add text annotations
    for i in range(len(envs)):
        for j in range(len(depths)):
            val = matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    plt.tight_layout()

    base, ext = os.path.splitext(output_path)
    heatmap_path = f"{base}_heatmap{ext}"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {heatmap_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot depth study results")
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--entity", default=None, help="WandB entity")
    parser.add_argument("--output", default="plots/depth_study.pdf", help="Output path for plots")
    parser.add_argument("--prefix", default=None, help="Run name prefix (auto-detected from project name)")
    args = parser.parse_args()

    # Auto-detect prefix from project name
    if args.prefix is None:
        if "td3" in args.project.lower():
            args.prefix = "td3_depth"
        elif "dqn" in args.project.lower():
            args.prefix = "dqn_depth"
        else:
            args.prefix = "depth"

    print(f"Fetching results from WandB project: {args.project}")
    df = fetch_study_results(args.project, args.entity, args.prefix)

    if len(df) == 0:
        print("No completed runs found!")
        return

    print(f"Found {len(df)} runs")

    summary_df = compute_summary_stats(df)
    print("\nSummary statistics:")
    print(summary_df.to_string(index=False))

    # Determine algorithm name for titles
    algo = "TD3" if "td3" in args.project.lower() else "DQN" if "dqn" in args.project.lower() else "RL"

    # Create all plots
    plot_depth_comparison(summary_df, args.output, title=f"{algo} Performance vs Network Depth")
    plot_combined_normalized(summary_df, args.output, title=f"{algo} Normalized Performance vs Depth")
    plot_heatmap(summary_df, args.output, title=f"{algo} Depth Study - Normalized Performance")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
