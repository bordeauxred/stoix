"""Plot multi-seed experiment results from JSON logs.

Creates learning curves with mean ± std across seeds.

Usage:
    python experiments/plot_multiseed_results.py results/json/td3_pendulum_d4_silu_utd4_20260118_184151
    python experiments/plot_multiseed_results.py results/json/td3_* --output plots/td3_comparison.pdf
"""

import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
except ImportError:
    print("Please install matplotlib: pip install matplotlib")
    exit(1)


def load_seed_metrics(group_path: str) -> dict:
    """Load metrics from all seeds in a group directory."""
    seed_dirs = sorted(glob(os.path.join(group_path, "seed_*")))

    if not seed_dirs:
        print(f"No seed directories found in {group_path}")
        return {}

    all_metrics = {}
    seeds = []

    for seed_dir in seed_dirs:
        seed = int(os.path.basename(seed_dir).replace("seed_", ""))
        seeds.append(seed)

        metrics_file = os.path.join(seed_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(np.array(value))

    # Stack arrays across seeds
    stacked = {}
    for key, values in all_metrics.items():
        try:
            stacked[key] = np.stack(values, axis=0)  # (num_seeds, ...)
        except ValueError:
            # Different lengths, skip
            pass

    return {"metrics": stacked, "seeds": seeds, "group": os.path.basename(group_path)}


def plot_learning_curves(data: dict, output_path: str = None, show: bool = True):
    """Plot learning curves with mean ± std across seeds."""
    metrics = data["metrics"]
    group_name = data["group"]
    seeds = data["seeds"]

    # Determine which metrics to plot
    plot_metrics = []
    for key in ["episode_return", "q_loss", "actor_loss", "loss"]:
        if key in metrics and metrics[key].ndim >= 2:
            plot_metrics.append(key)

    if not plot_metrics:
        print("No plottable metrics found")
        return

    n_plots = len(plot_metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    colors = {'episode_return': '#2ecc71', 'q_loss': '#e74c3c', 'actor_loss': '#3498db', 'loss': '#9b59b6'}

    for ax, metric_name in zip(axes, plot_metrics):
        metric_data = metrics[metric_name]  # (num_seeds, num_steps, ...)

        # Aggregate if needed (e.g., across envs)
        while metric_data.ndim > 2:
            metric_data = np.mean(metric_data, axis=-1)

        # metric_data is now (num_seeds, num_steps)
        mean = np.mean(metric_data, axis=0)
        std = np.std(metric_data, axis=0)
        steps = np.arange(len(mean))

        color = colors.get(metric_name, '#333333')

        ax.plot(steps, mean, color=color, linewidth=2, label=f'Mean (n={len(seeds)})')
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2, label='±1 std')

        # Plot individual seeds as thin lines
        for i, seed in enumerate(seeds):
            ax.plot(steps, metric_data[i], color=color, alpha=0.3, linewidth=0.5)

        ax.set_xlabel('Update Step', fontsize=11)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{group_name}\nSeeds: {seeds}', fontsize=11)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(group_paths: list, output_path: str = None, show: bool = True):
    """Plot comparison across multiple experiment groups."""
    all_data = []
    for path in group_paths:
        data = load_seed_metrics(path)
        if data:
            all_data.append(data)

    if not all_data:
        print("No data loaded")
        return

    # Plot episode_return comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))

    for i, data in enumerate(all_data):
        metrics = data["metrics"]
        if "episode_return" not in metrics:
            continue

        returns = metrics["episode_return"]
        while returns.ndim > 2:
            returns = np.mean(returns, axis=-1)

        mean = np.mean(returns, axis=0)
        std = np.std(returns, axis=0)
        steps = np.arange(len(mean))

        label = data["group"]
        ax.plot(steps, mean, color=colors[i], linewidth=2, label=label)
        ax.fill_between(steps, mean - std, mean + std, color=colors[i], alpha=0.2)

    ax.set_xlabel('Update Step', fontsize=12)
    ax.set_ylabel('Episode Return', fontsize=12)
    ax.set_title('Learning Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot multi-seed experiment results")
    parser.add_argument("paths", nargs="+", help="Path(s) to experiment group directories")
    parser.add_argument("--output", "-o", default=None, help="Output path for plot")
    parser.add_argument("--no-show", action="store_true", help="Don't display plot")
    parser.add_argument("--compare", action="store_true", help="Compare multiple groups")
    args = parser.parse_args()

    # Expand globs
    all_paths = []
    for p in args.paths:
        expanded = glob(p)
        all_paths.extend(expanded if expanded else [p])

    if args.compare or len(all_paths) > 1:
        plot_comparison(all_paths, args.output, show=not args.no_show)
    else:
        data = load_seed_metrics(all_paths[0])
        if data:
            plot_learning_curves(data, args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
