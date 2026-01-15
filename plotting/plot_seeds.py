#!/usr/bin/env python3
"""
Simple plotting script for multi-seed Stoix experiments.

Plots mean +/- std over seeds for comparing algorithms/configurations.

Usage:
    # Plot from local JSON results
    python plotting/plot_seeds.py --path results/json/my_experiment

    # Specify metrics to plot
    python plotting/plot_seeds.py --path results/json/my_experiment --metrics episode_return

    # Save figure
    python plotting/plot_seeds.py --path results/json/my_experiment --save my_plot.png
"""

import argparse
import collections
import json
from os import walk
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_json_files(folder: str) -> Dict[str, Any]:
    """Load all JSON files from a folder and merge them."""
    files = []
    for dirpath, _, filenames in walk(folder):
        for filename in filenames:
            if filename.endswith(".json"):
                files.append(Path(dirpath) / filename)

    if not files:
        raise FileNotFoundError(f"No JSON files found in {folder}")

    print(f"Found {len(files)} JSON files")

    # Load and merge
    merged: Dict[str, Any] = {}
    for filepath in sorted(files):
        with open(filepath) as f:
            data = json.load(f)
            _deep_update(merged, data)

    return merged


def _deep_update(d: Dict, u: Dict) -> Dict:
    """Recursively update dict d with dict u."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def extract_learning_curves(
    data: Dict[str, Any],
    metric: str = "mean_episode_return",
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Extract learning curves from nested marl-eval format.

    Args:
        data: Loaded JSON data in marl-eval format
        metric: Metric name to extract

    Returns:
        Dict[env_name][algo_name] -> (timesteps, values) arrays for each seed
    """
    results: Dict[str, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]] = {}

    # Navigate marl-eval nested structure:
    # env_name -> task_name -> algo_name -> seed -> {step: {metric: value}}
    for env_name, env_data in data.items():
        if not isinstance(env_data, dict):
            continue

        results[env_name] = {}

        for task_name, task_data in env_data.items():
            if not isinstance(task_data, dict):
                continue

            for algo_name, algo_data in task_data.items():
                if not isinstance(algo_data, dict):
                    continue

                # Create key combining task and algo if multiple tasks
                key = f"{algo_name}" if len(env_data) == 1 else f"{task_name}/{algo_name}"

                if key not in results[env_name]:
                    results[env_name][key] = []

                for seed_name, seed_data in algo_data.items():
                    if not isinstance(seed_data, dict):
                        continue

                    # Extract timesteps and values
                    timesteps = []
                    values = []

                    # Filter to only step_N keys, skip 'absolute_metrics' etc
                    step_items = [(k, v) for k, v in seed_data.items() if k.startswith('step_')]
                    for step_str, step_data in sorted(step_items, key=lambda x: int(x[0].split('_')[1])):
                        if not isinstance(step_data, dict):
                            continue
                        if metric in step_data:
                            # Use step_count for actual timestep
                            timesteps.append(step_data.get('step_count', 0))
                            # metric value may be a list, take first element
                            val = step_data[metric]
                            values.append(val[0] if isinstance(val, list) else val)

                    if timesteps:
                        results[env_name][key].append(
                            (np.array(timesteps), np.array(values))
                        )

    return results


def compute_mean_std(
    curves: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and std over multiple seed runs.

    Args:
        curves: List of (timesteps, values) tuples from different seeds

    Returns:
        (timesteps, mean, std) arrays
    """
    if not curves:
        return np.array([]), np.array([]), np.array([])

    # Find common timesteps (use first seed's timesteps as reference)
    timesteps = curves[0][0]

    # Interpolate all curves to common timesteps
    all_values = []
    for ts, vals in curves:
        if len(ts) == len(timesteps) and np.allclose(ts, timesteps):
            all_values.append(vals)
        else:
            # Interpolate to common timesteps
            interp_vals = np.interp(timesteps, ts, vals)
            all_values.append(interp_vals)

    all_values = np.array(all_values)
    mean = np.mean(all_values, axis=0)
    std = np.std(all_values, axis=0)

    return timesteps, mean, std


def plot_comparison(
    data: Dict[str, Any],
    metric: str = "mean_episode_return",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    smoothing: int = 1,
) -> plt.Figure:
    """Plot comparison of algorithms with mean +/- std.

    Args:
        data: Loaded JSON data
        metric: Metric to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        smoothing: Window size for smoothing (1 = no smoothing)

    Returns:
        matplotlib Figure
    """
    curves_by_env = extract_learning_curves(data, metric)

    if not curves_by_env:
        raise ValueError(f"No data found for metric '{metric}'")

    # Create subplots for each environment
    n_envs = len(curves_by_env)
    fig, axes = plt.subplots(1, n_envs, figsize=(figsize[0] * n_envs, figsize[1]), squeeze=False)

    colors = plt.cm.tab10.colors

    for env_idx, (env_name, algo_curves) in enumerate(curves_by_env.items()):
        ax = axes[0, env_idx]

        for algo_idx, (algo_name, curves) in enumerate(algo_curves.items()):
            n_seeds = len(curves)
            timesteps, mean, std = compute_mean_std(curves)

            if len(timesteps) == 0:
                continue

            # Apply smoothing if requested
            if smoothing > 1:
                kernel = np.ones(smoothing) / smoothing
                mean = np.convolve(mean, kernel, mode='valid')
                std = np.convolve(std, kernel, mode='valid')
                timesteps = timesteps[:len(mean)]

            color = colors[algo_idx % len(colors)]

            # Plot mean line
            ax.plot(
                timesteps,
                mean,
                label=f"{algo_name} (n={n_seeds})",
                color=color,
                linewidth=2,
            )

            # Plot std band
            ax.fill_between(
                timesteps,
                mean - std,
                mean + std,
                alpha=0.2,
                color=color,
            )

        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(env_name.replace("_", " ").title(), fontsize=14)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def print_summary(data: Dict[str, Any], metric: str = "mean_episode_return") -> None:
    """Print summary statistics."""
    curves_by_env = extract_learning_curves(data, metric)

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for env_name, algo_curves in curves_by_env.items():
        print(f"\n{env_name}:")
        print("-" * 40)

        for algo_name, curves in algo_curves.items():
            n_seeds = len(curves)
            if n_seeds == 0:
                continue

            # Get final values
            final_values = [c[1][-1] for c in curves if len(c[1]) > 0]
            if final_values:
                mean_final = np.mean(final_values)
                std_final = np.std(final_values)
                print(f"  {algo_name:30s} | seeds={n_seeds} | final={mean_final:.2f} +/- {std_final:.2f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Plot multi-seed Stoix experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--path", "-p",
        required=True,
        help="Path to results directory containing JSON files",
    )
    parser.add_argument(
        "--metric", "-m",
        default="mean_episode_return",
        help="Metric to plot (default: mean_episode_return)",
    )
    parser.add_argument(
        "--save", "-s",
        help="Path to save the figure (e.g., plot.png)",
    )
    parser.add_argument(
        "--title", "-t",
        help="Plot title",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=1,
        help="Smoothing window size (default: 1 = no smoothing)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (useful for scripts)",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.path}")
    data = load_json_files(args.path)

    # Print summary
    print_summary(data, args.metric)

    # Plot
    fig = plot_comparison(
        data,
        metric=args.metric,
        title=args.title,
        save_path=args.save,
        smoothing=args.smoothing,
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
