#!/usr/bin/env python3
"""Plot TD3 depth study learning curves (UTD=32, ReLU)."""

import json
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 11

RESULTS_DIR = Path(__file__).parent.parent / "results" / "json"
PLOTS_DIR = Path(__file__).parent.parent / "plots"

DEPTH_COLORS = {
    2: '#1f77b4',
    4: '#ff7f0e',
    8: '#2ca02c',
    16: '#d62728',
    32: '#9467bd',
    64: '#8c564b',
}


def extract_learning_curve(metrics, window=500):
    returns = np.array(metrics['episode_return'])
    terminals = np.array(metrics['is_terminal'])
    flat_returns = returns.reshape(returns.shape[0], -1)
    flat_terminals = terminals.reshape(terminals.shape[0], -1)

    steps = []
    mean_returns = []
    sample_interval = max(1, returns.shape[0] // 200)

    for step in range(sample_interval, returns.shape[0], sample_interval):
        start = max(0, step - window)
        ret_window = flat_returns[start:step]
        term_window = flat_terminals[start:step]
        episode_returns = ret_window[term_window > 0]
        if len(episode_returns) > 0:
            steps.append(step)
            mean_returns.append(np.mean(episode_returns))

    return np.array(steps), np.array(mean_returns)


def parse_experiment_name(name):
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


def load_learning_curves(exp_dir, window=500):
    seed_dirs = sorted(glob(os.path.join(exp_dir, 'seed_*')))
    if not seed_dirs:
        return None

    all_curves = []
    for seed_dir in seed_dirs:
        metrics_file = os.path.join(seed_dir, 'metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            steps, returns = extract_learning_curve(metrics, window)
            if len(steps) > 0:
                all_curves.append((steps, returns))

    if not all_curves:
        return None

    max_steps = max(curve[0][-1] for curve in all_curves)
    common_steps = np.linspace(0, max_steps, 200)

    interpolated = []
    for steps, returns in all_curves:
        interp_returns = np.interp(common_steps, steps, returns)
        interpolated.append(interp_returns)

    interpolated = np.array(interpolated)
    return {
        'steps': common_steps,
        'mean': np.mean(interpolated, axis=0),
        'std': np.std(interpolated, axis=0),
        'n_seeds': len(all_curves),
    }


def main():
    print("Collecting TD3 depth study (UTD=32) learning curves...")
    results = defaultdict(lambda: defaultdict(dict))

    for exp_dir in RESULTS_DIR.glob("td3_*_utd32_*"):
        if not exp_dir.is_dir():
            continue

        parsed = parse_experiment_name(exp_dir.name)
        if not parsed or parsed['utd'] != 32:
            continue

        # Focus on relu for depth comparison (more complete data)
        if parsed['activation'] != 'relu':
            continue

        print(f"  Loading {exp_dir.name}")
        curves = load_learning_curves(str(exp_dir))
        if curves is None:
            continue

        env = parsed['env']
        depth = parsed['depth']
        results[env][depth] = curves

    # Plot learning curves
    envs = sorted(results.keys())
    n_envs = len(envs)
    print(f"Found data for {n_envs} environments")

    n_cols = min(2, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    for idx, env in enumerate(envs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        env_data = results[env]

        for depth in sorted(env_data.keys()):
            curves = env_data[depth]
            color = DEPTH_COLORS.get(depth, '#333333')

            steps = curves['steps']
            mean = curves['mean']
            std = curves['std']
            n_seeds = curves['n_seeds']

            label = f'd{depth} (n={n_seeds})'
            ax.plot(steps, mean, color=color, linewidth=2, label=label)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)

        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Episode Return', fontsize=12)
        ax.set_title(env.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_envs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle('TD3 Depth Study (UTD=32, ReLU): Learning Curves', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(PLOTS_DIR / 'td3_depth_learning_curves.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(PLOTS_DIR / 'td3_depth_learning_curves.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR}/td3_depth_learning_curves.pdf")
    plt.close(fig)

    # Create final performance bar plot
    print("\nCreating final performance plot...")
    fig, axes = plt.subplots(1, n_envs, figsize=(5 * n_envs, 4), squeeze=False)

    for idx, env in enumerate(envs):
        ax = axes[0, idx]
        env_data = results[env]

        depths = sorted(env_data.keys())
        means = [env_data[d]['mean'][-20:].mean() for d in depths]
        stds = [env_data[d]['std'][-20:].mean() for d in depths]
        n_seeds = [env_data[d]['n_seeds'] for d in depths]

        colors = [DEPTH_COLORS.get(d, '#333333') for d in depths]
        x = np.arange(len(depths))

        ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1)

        ax.set_xlabel('Network Depth', fontsize=12)
        ax.set_ylabel('Episode Return', fontsize=12)
        ax.set_title(env.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'd{d}\n(n={n})' for d, n in zip(depths, n_seeds)])
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('TD3 Depth Study (UTD=32, ReLU): Final Performance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(PLOTS_DIR / 'td3_depth_final_performance.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(PLOTS_DIR / 'td3_depth_final_performance.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR}/td3_depth_final_performance.pdf")

    # Print summary table
    print("\n" + "=" * 60)
    print("TD3 DEPTH STUDY SUMMARY (UTD=32, ReLU)")
    print("=" * 60)
    for env in envs:
        print(f"\n{env.upper()}:")
        print("-" * 40)
        print(f"{'Depth':<8} {'Mean':>12} {'Std':>10} {'N':>6}")
        print("-" * 40)
        for depth in sorted(results[env].keys()):
            data = results[env][depth]
            final_mean = data['mean'][-20:].mean()
            final_std = data['std'][-20:].mean()
            print(f"d{depth:<7} {final_mean:>12.2f} {final_std:>10.2f} {data['n_seeds']:>6}")
    print("=" * 60)


if __name__ == "__main__":
    main()
