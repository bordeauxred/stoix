#!/usr/bin/env python3
"""Plot learning curves for depth studies.

Creates learning curve plots showing training progression:
- X-axis: training epochs/steps
- Y-axis: episode return
- One plot per environment
- Each line: mean ± std over seeds for each depth

Usage:
    python experiments/plot_learning_curves.py
"""

import json
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 11
except ImportError:
    print("Please install matplotlib: pip install matplotlib")
    exit(1)

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results" / "json"
PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Color palette for depths (using a colormap)
DEPTH_COLORS = {
    2: '#1f77b4',   # blue
    4: '#ff7f0e',   # orange
    8: '#2ca02c',   # green
    16: '#d62728',  # red
    32: '#9467bd',  # purple
    64: '#8c564b',  # brown
}

ACTIVATION_LINESTYLES = {
    'silu': '-',
    'relu': '--',
}


def extract_learning_curve(metrics: dict, window: int = 500) -> tuple:
    """Extract learning curve from metrics.

    Returns (steps, mean_returns) where mean_returns is computed
    over a rolling window of episodes.
    """
    returns = np.array(metrics['episode_return'])
    terminals = np.array(metrics['is_terminal'])

    # Flatten across batch dimensions
    flat_returns = returns.reshape(returns.shape[0], -1)
    flat_terminals = terminals.reshape(terminals.shape[0], -1)

    # Compute rolling mean of episode returns
    steps = []
    mean_returns = []

    # Sample at regular intervals
    sample_interval = max(1, returns.shape[0] // 200)  # ~200 points per curve

    for step in range(sample_interval, returns.shape[0], sample_interval):
        # Look at a window of steps
        start = max(0, step - window)
        ret_window = flat_returns[start:step]
        term_window = flat_terminals[start:step]

        # Get returns where episodes ended
        episode_returns = ret_window[term_window > 0]

        if len(episode_returns) > 0:
            steps.append(step)
            mean_returns.append(np.mean(episode_returns))

    return np.array(steps), np.array(mean_returns)


def parse_experiment_name(name: str) -> dict:
    """Parse experiment folder name into components."""
    pattern = r"^(dqn|td3)_(.+?)_d(\d+)_(silu|relu)(?:_utd(\d+))?_\d{8}_\d{6}$"
    match = re.match(pattern, name)
    if match:
        return {
            'algo': match.group(1),
            'env': match.group(2),
            'depth': int(match.group(3)),
            'activation': match.group(4),
            'utd': int(match.group(5)) if match.group(5) else None,
        }
    return None


def load_learning_curves(exp_dir: Path, window: int = 500) -> dict:
    """Load learning curves from all seeds in an experiment."""
    seed_dirs = sorted(glob(os.path.join(exp_dir, "seed_*")))

    if not seed_dirs:
        return None

    all_curves = []

    for seed_dir in seed_dirs:
        metrics_file = os.path.join(seed_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            steps, returns = extract_learning_curve(metrics, window)
            if len(steps) > 0:
                all_curves.append((steps, returns))

    if not all_curves:
        return None

    # Interpolate all curves to common steps
    # Use the steps from the longest curve as reference
    max_steps = max(curve[0][-1] for curve in all_curves)
    common_steps = np.linspace(0, max_steps, 200)

    interpolated = []
    for steps, returns in all_curves:
        interp_returns = np.interp(common_steps, steps, returns)
        interpolated.append(interp_returns)

    interpolated = np.array(interpolated)
    mean = np.mean(interpolated, axis=0)
    std = np.std(interpolated, axis=0)

    return {
        'steps': common_steps,
        'mean': mean,
        'std': std,
        'n_seeds': len(all_curves),
    }


def collect_dqn_learning_curves() -> dict:
    """Collect learning curves for all DQN experiments."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    exp_dirs = list(RESULTS_DIR.glob("dqn_*"))
    print(f"Loading learning curves from {len(exp_dirs)} DQN experiments...")

    for i, exp_dir in enumerate(exp_dirs):
        if not exp_dir.is_dir():
            continue

        parsed = parse_experiment_name(exp_dir.name)
        if not parsed:
            continue

        print(f"  [{i+1}/{len(exp_dirs)}] {exp_dir.name}")

        curves = load_learning_curves(exp_dir)
        if curves is None:
            continue

        env = parsed['env']
        depth = parsed['depth']
        activation = parsed['activation']

        results[env][activation][depth] = curves

    return dict(results)


def collect_td3_learning_curves() -> dict:
    """Collect learning curves for TD3 UTD experiments (focusing on depth comparison)."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    exp_dirs = list(RESULTS_DIR.glob("td3_*"))
    print(f"Loading learning curves from {len(exp_dirs)} TD3 experiments...")

    for i, exp_dir in enumerate(exp_dirs):
        if not exp_dir.is_dir():
            continue

        parsed = parse_experiment_name(exp_dir.name)
        if not parsed:
            continue

        print(f"  [{i+1}/{len(exp_dirs)}] {exp_dir.name}")

        curves = load_learning_curves(exp_dir)
        if curves is None:
            continue

        env = parsed['env']
        depth = parsed['depth']
        activation = parsed['activation']
        utd = parsed['utd']

        # Group by depth (for depth comparison) or by UTD (for UTD comparison)
        if utd is not None:
            # For UTD study, store with UTD as key
            key = f"utd{utd}"
            results[env][activation][key] = curves
            results[env][activation][key]['depth'] = depth
            results[env][activation][key]['utd'] = utd
        else:
            results[env][activation][depth] = curves

    return dict(results)


def plot_dqn_learning_curves(results: dict):
    """Plot DQN learning curves: one plot per env, lines for each depth/activation."""
    if not results:
        print("No DQN learning curve results")
        return

    envs = sorted(results.keys())
    n_envs = len(envs)

    # Create figure with subplots
    n_cols = min(3, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for idx, env in enumerate(envs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        env_data = results[env]

        # Plot each activation/depth combination
        for activation in sorted(env_data.keys()):
            linestyle = ACTIVATION_LINESTYLES.get(activation, '-')

            for depth in sorted(env_data[activation].keys()):
                curves = env_data[activation][depth]
                color = DEPTH_COLORS.get(depth, '#333333')

                steps = curves['steps']
                mean = curves['mean']
                std = curves['std']
                n_seeds = curves['n_seeds']

                label = f'd{depth} {activation.upper()} (n={n_seeds})'

                ax.plot(steps, mean, color=color, linestyle=linestyle,
                       linewidth=2, label=label)
                ax.fill_between(steps, mean - std, mean + std,
                               color=color, alpha=0.15)

        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Episode Return', fontsize=12)
        ax.set_title(env.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_envs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle('DQN Depth Study: Learning Curves', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = PLOTS_DIR / "dqn_learning_curves.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    fig.savefig(PLOTS_DIR / "dqn_learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_td3_learning_curves(results: dict):
    """Plot TD3 learning curves for UTD study."""
    if not results:
        print("No TD3 learning curve results")
        return

    envs = sorted(results.keys())
    n_envs = len(envs)

    # Create figure with subplots
    n_cols = min(2, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    # UTD colors
    utd_colors = {
        1: '#1f77b4',
        2: '#ff7f0e',
        4: '#2ca02c',
        8: '#d62728',
        16: '#9467bd',
        20: '#8c564b',
        32: '#e377c2',
    }

    for idx, env in enumerate(envs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        env_data = results[env]

        # Collect all UTD configs
        utd_configs = []
        for activation in env_data.keys():
            for key, curves in env_data[activation].items():
                if key.startswith('utd'):
                    utd = curves.get('utd', int(key.replace('utd', '')))
                    utd_configs.append((utd, activation, curves))

        # Sort by UTD
        utd_configs.sort(key=lambda x: x[0])

        for utd, activation, curves in utd_configs:
            color = utd_colors.get(utd, '#333333')
            linestyle = ACTIVATION_LINESTYLES.get(activation, '-')

            steps = curves['steps']
            mean = curves['mean']
            std = curves['std']
            n_seeds = curves['n_seeds']

            label = f'UTD={utd} (n={n_seeds})'

            ax.plot(steps, mean, color=color, linestyle=linestyle,
                   linewidth=2, label=label)
            ax.fill_between(steps, mean - std, mean + std,
                           color=color, alpha=0.15)

        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Episode Return', fontsize=12)
        ax.set_title(env.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_envs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle('TD3 UTD Study: Learning Curves', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = PLOTS_DIR / "td3_utd_learning_curves.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    fig.savefig(PLOTS_DIR / "td3_utd_learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    print("=" * 60)
    print("Generating Learning Curve Plots")
    print("=" * 60)

    # Collect and plot DQN learning curves
    print("\n--- DQN Learning Curves ---")
    dqn_curves = collect_dqn_learning_curves()
    plot_dqn_learning_curves(dqn_curves)

    # Collect and plot TD3 learning curves
    print("\n--- TD3 Learning Curves ---")
    td3_curves = collect_td3_learning_curves()
    plot_td3_learning_curves(td3_curves)

    print(f"\nAll learning curve plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
