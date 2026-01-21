#!/usr/bin/env python3
"""Plot all depth and UTD study results.

Creates publication-quality plots for:
1. DQN depth study (swish vs relu activation)
2. TD3 UTD study
3. TD3 depth study (partial)

Usage:
    python experiments/plot_all_studies.py
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

# Color palettes
ENV_COLORS = {
    'asterix': '#e41a1c',
    'breakout': '#377eb8',
    'freeway': '#4daf4a',
    'mountain_car': '#984ea3',
    'space_invaders': '#ff7f00',
    'cartpole': '#a65628',
    'ant': '#e41a1c',
    'halfcheetah': '#377eb8',
    'hopper': '#4daf4a',
    'pendulum': '#984ea3',
}

ACTIVATION_COLORS = {
    'silu': '#2ecc71',
    'relu': '#e74c3c',
}

DEPTH_COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, 6))


def extract_episode_returns(metrics: dict) -> np.ndarray:
    """Extract per-episode returns from metrics (when is_terminal=True)."""
    returns = np.array(metrics['episode_return'])
    terminals = np.array(metrics['is_terminal'])

    # Flatten and filter by terminal flag
    flat_returns = returns.flatten()
    flat_terminals = terminals.flatten()

    # Get returns at terminal states
    episode_returns = flat_returns[flat_terminals > 0]
    return episode_returns


def compute_rolling_mean(returns: np.ndarray, window: int = 100) -> tuple:
    """Compute rolling mean of episode returns."""
    if len(returns) < window:
        window = max(1, len(returns) // 10)

    rolling_mean = np.convolve(returns, np.ones(window)/window, mode='valid')
    return rolling_mean


def load_multiseed_experiment(group_path: str, last_n: int = 100) -> dict:
    """Load metrics from all seeds and compute final performance only."""
    seed_dirs = sorted(glob(os.path.join(group_path, "seed_*")))

    if not seed_dirs:
        return {}

    final_returns = []
    seeds = []

    for seed_dir in seed_dirs:
        seed = int(os.path.basename(seed_dir).replace("seed_", ""))
        seeds.append(seed)

        metrics_file = os.path.join(seed_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            episode_returns = extract_episode_returns(metrics)
            # Only keep final performance
            if len(episode_returns) >= last_n:
                final_returns.append(np.mean(episode_returns[-last_n:]))
            elif len(episode_returns) > 0:
                final_returns.append(np.mean(episode_returns))

    if not final_returns:
        return {}

    return {
        "mean": np.mean(final_returns),
        "std": np.std(final_returns),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "group": os.path.basename(group_path)
    }


def get_final_performance(returns_list: list, last_n: int = 100) -> tuple:
    """Get mean and std of final performance across seeds."""
    final_returns = []
    for returns in returns_list:
        if len(returns) >= last_n:
            final_returns.append(np.mean(returns[-last_n:]))
        elif len(returns) > 0:
            final_returns.append(np.mean(returns))

    if not final_returns:
        return np.nan, np.nan

    return np.mean(final_returns), np.std(final_returns)


def parse_experiment_name(name: str) -> dict:
    """Parse experiment folder name into components."""
    # Format: algo_env_d{depth}_{activation}[_utd{utd}]_YYYYMMDD_HHMMSS
    # Activation is silu or relu
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


def collect_dqn_results() -> dict:
    """Collect all DQN depth study results."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    exp_dirs = list(RESULTS_DIR.glob("dqn_*"))
    print(f"Found {len(exp_dirs)} DQN experiment directories")

    for i, exp_dir in enumerate(exp_dirs):
        if not exp_dir.is_dir():
            continue

        parsed = parse_experiment_name(exp_dir.name)
        if not parsed:
            continue

        print(f"  [{i+1}/{len(exp_dirs)}] Loading {exp_dir.name}...")
        data = load_multiseed_experiment(str(exp_dir))
        if not data:
            continue

        env = parsed['env']
        depth = parsed['depth']
        activation = parsed['activation']

        results[env][activation][depth] = {
            'mean': data['mean'],
            'std': data['std'],
            'n_seeds': data['n_seeds'],
        }

    return dict(results)


def collect_td3_utd_results() -> dict:
    """Collect all TD3 UTD study results."""
    results = defaultdict(lambda: defaultdict(dict))

    exp_dirs = list(RESULTS_DIR.glob("td3_*"))
    print(f"Found {len(exp_dirs)} TD3 experiment directories")

    for i, exp_dir in enumerate(exp_dirs):
        if not exp_dir.is_dir():
            continue

        parsed = parse_experiment_name(exp_dir.name)
        if not parsed or parsed['utd'] is None:
            continue

        print(f"  [{i+1}/{len(exp_dirs)}] Loading {exp_dir.name}...")
        data = load_multiseed_experiment(str(exp_dir))
        if not data:
            continue

        env = parsed['env']
        utd = parsed['utd']

        results[env][utd] = {
            'mean': data['mean'],
            'std': data['std'],
            'n_seeds': data['n_seeds'],
            'depth': parsed['depth'],
            'activation': parsed['activation'],
        }

    return dict(results)


def collect_td3_depth_results() -> dict:
    """Collect TD3 depth study results from baseline folder."""
    results = defaultdict(lambda: defaultdict(dict))

    baseline_dir = RESULTS_DIR / "depth_baseline_td3_10M"
    if not baseline_dir.exists():
        return {}

    for depth_env_dir in baseline_dir.glob("d*_*"):
        if not depth_env_dir.is_dir():
            continue

        # Parse d{depth}_{env}
        match = re.match(r"d(\d+)_(.+)", depth_env_dir.name)
        if not match:
            continue

        depth = int(match.group(1))
        env = match.group(2)

        metrics_file = depth_env_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        with open(metrics_file, 'r') as f:
            data = json.load(f)

        # Extract final returns from marl-eval format
        final_returns = []
        try:
            env_data = data.get('brax', {}).get(env, {})
            for algo_name, algo_data in env_data.items():
                for seed_name, seed_data in algo_data.items():
                    if 'absolute_metrics' in seed_data:
                        abs_metrics = seed_data['absolute_metrics']
                        for key, val in abs_metrics.items():
                            if 'episode_return' in key and isinstance(val, list):
                                final_returns.extend(val)
        except Exception as e:
            print(f"Error processing {depth_env_dir}: {e}")
            continue

        if final_returns:
            results[env][depth] = {
                'mean': np.mean(final_returns),
                'std': np.std(final_returns),
                'n_seeds': len(final_returns),
            }

    return dict(results)


def plot_dqn_depth_study(results: dict):
    """Create DQN depth study plots comparing activations per environment."""
    if not results:
        print("No DQN results to plot")
        return

    envs = sorted(results.keys())
    n_envs = len(envs)

    if n_envs == 0:
        return

    # Create multi-panel figure
    n_cols = min(3, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, env in enumerate(envs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        env_data = results[env]

        for activation, act_data in sorted(env_data.items()):
            depths = sorted(act_data.keys())
            means = [act_data[d]['mean'] for d in depths]
            stds = [act_data[d]['std'] for d in depths]
            n_seeds = [act_data[d]['n_seeds'] for d in depths]

            color = ACTIVATION_COLORS.get(activation, '#333333')

            ax.errorbar(depths, means, yerr=stds,
                       marker='o', linewidth=2, markersize=6,
                       color=color, capsize=4,
                       label=f'{activation.upper()} (n={n_seeds[0]})')

        ax.set_xscale('log', base=2)
        ax.set_xlabel('Network Depth', fontsize=12)
        ax.set_ylabel('Episode Return', fontsize=12)
        ax.set_title(env.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(depths)
        ax.set_xticklabels([str(d) for d in depths])

    # Hide unused subplots
    for idx in range(n_envs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle('DQN Depth Study: Swish vs ReLU Activation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = PLOTS_DIR / "dqn_depth_study.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save PNG
    fig.savefig(PLOTS_DIR / "dqn_depth_study.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Create combined normalized plot
    plot_dqn_normalized(results)


def plot_dqn_normalized(results: dict):
    """Create a combined normalized plot for DQN depth study."""
    fig, ax = plt.subplots(figsize=(8, 5))

    all_depths = set()
    for env_data in results.values():
        for act_data in env_data.values():
            all_depths.update(act_data.keys())

    depths = sorted(all_depths)

    for activation in ['silu', 'relu']:
        # Compute normalized performance per env then average
        normalized_per_env = []

        for env, env_data in results.items():
            if activation not in env_data:
                continue

            act_data = env_data[activation]
            env_depths = sorted(act_data.keys())
            means = [act_data[d]['mean'] for d in env_depths]

            if not means or np.isnan(means).all():
                continue

            # Normalize by max performance in this env
            max_perf = max(max(means), 1e-6)
            min_perf = min(means)
            range_perf = max_perf - min_perf if max_perf != min_perf else 1

            normalized = [(m - min_perf) / range_perf for m in means]
            normalized_per_env.append((env_depths, normalized))

        if not normalized_per_env:
            continue

        # Aggregate across environments
        depth_vals = defaultdict(list)
        for env_depths, normalized in normalized_per_env:
            for d, n in zip(env_depths, normalized):
                depth_vals[d].append(n)

        plot_depths = sorted(depth_vals.keys())
        plot_means = [np.mean(depth_vals[d]) for d in plot_depths]
        plot_stds = [np.std(depth_vals[d]) for d in plot_depths]

        color = ACTIVATION_COLORS.get(activation, '#333333')
        ax.errorbar(plot_depths, plot_means, yerr=plot_stds,
                   marker='o', linewidth=2, markersize=8,
                   color=color, capsize=4,
                   label=f'{activation.upper()}')

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Network Depth', fontsize=12)
    ax.set_ylabel('Normalized Performance', fontsize=12)
    ax.set_title('DQN Depth Study: Normalized Across Environments', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])

    plt.tight_layout()

    output_path = PLOTS_DIR / "dqn_depth_study_normalized.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    fig.savefig(PLOTS_DIR / "dqn_depth_study_normalized.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_td3_utd_study(results: dict):
    """Create TD3 UTD study plots."""
    if not results:
        print("No TD3 UTD results to plot")
        return

    envs = sorted(results.keys())
    n_envs = len(envs)

    if n_envs == 0:
        return

    # Create multi-panel figure
    n_cols = min(2, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    for idx, env in enumerate(envs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        env_data = results[env]
        utds = sorted(env_data.keys())
        means = [env_data[u]['mean'] for u in utds]
        stds = [env_data[u]['std'] for u in utds]
        n_seeds = [env_data[u]['n_seeds'] for u in utds]

        color = ENV_COLORS.get(env, '#333333')

        ax.errorbar(utds, means, yerr=stds,
                   marker='o', linewidth=2, markersize=8,
                   color=color, capsize=4,
                   label=f'n={n_seeds[0]}')

        # Mark best UTD
        best_idx = np.nanargmax(means)
        ax.scatter([utds[best_idx]], [means[best_idx]],
                  marker='*', s=200, color='gold', edgecolor='black',
                  zorder=5, label=f'Best: UTD={utds[best_idx]}')

        ax.set_xscale('log', base=2)
        ax.set_xlabel('Update-to-Data Ratio (UTD)', fontsize=12)
        ax.set_ylabel('Episode Return', fontsize=12)
        ax.set_title(env.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(utds)
        ax.set_xticklabels([str(u) for u in utds])

    # Hide unused subplots
    for idx in range(n_envs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle('TD3 Update-to-Data Ratio Study', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = PLOTS_DIR / "td3_utd_study.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    fig.savefig(PLOTS_DIR / "td3_utd_study.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_td3_depth_study(results: dict):
    """Create TD3 depth study plots."""
    if not results:
        print("No TD3 depth results to plot")
        return

    envs = sorted(results.keys())
    n_envs = len(envs)

    if n_envs == 0:
        return

    # Create multi-panel figure
    n_cols = min(3, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, env in enumerate(envs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        env_data = results[env]
        depths = sorted(env_data.keys())
        means = [env_data[d]['mean'] for d in depths]
        stds = [env_data[d]['std'] for d in depths]
        n_seeds = [env_data[d].get('n_seeds', 1) for d in depths]

        color = ENV_COLORS.get(env, '#333333')

        ax.errorbar(depths, means, yerr=stds,
                   marker='o', linewidth=2, markersize=8,
                   color=color, capsize=4,
                   label=f'TD3 (n={n_seeds[0]})')

        ax.set_xscale('log', base=2)
        ax.set_xlabel('Network Depth', fontsize=12)
        ax.set_ylabel('Episode Return', fontsize=12)
        ax.set_title(env.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(depths)
        ax.set_xticklabels([str(d) for d in depths])

    # Hide unused subplots
    for idx in range(n_envs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle('TD3 Depth Study (10M Steps)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = PLOTS_DIR / "td3_depth_study.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    fig.savefig(PLOTS_DIR / "td3_depth_study.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def print_summary_table(dqn_results: dict, td3_utd_results: dict, td3_depth_results: dict):
    """Print summary tables of results."""
    print("\n" + "=" * 80)
    print("DQN DEPTH STUDY SUMMARY")
    print("=" * 80)

    for env in sorted(dqn_results.keys()):
        print(f"\n{env.upper()}:")
        print("-" * 60)
        print(f"{'Activation':<12} {'Depth':<8} {'Mean':>12} {'Std':>10} {'N':>6}")
        print("-" * 60)

        for activation in sorted(dqn_results[env].keys()):
            for depth in sorted(dqn_results[env][activation].keys()):
                data = dqn_results[env][activation][depth]
                print(f"{activation:<12} {depth:<8} {data['mean']:>12.2f} {data['std']:>10.2f} {data['n_seeds']:>6}")

    print("\n" + "=" * 80)
    print("TD3 UTD STUDY SUMMARY")
    print("=" * 80)

    for env in sorted(td3_utd_results.keys()):
        print(f"\n{env.upper()}:")
        print("-" * 60)
        print(f"{'UTD':<8} {'Mean':>12} {'Std':>10} {'N':>6}")
        print("-" * 60)

        for utd in sorted(td3_utd_results[env].keys()):
            data = td3_utd_results[env][utd]
            print(f"{utd:<8} {data['mean']:>12.2f} {data['std']:>10.2f} {data['n_seeds']:>6}")

    print("\n" + "=" * 80)
    print("TD3 DEPTH STUDY SUMMARY (10M Steps)")
    print("=" * 80)

    for env in sorted(td3_depth_results.keys()):
        print(f"\n{env.upper()}:")
        print("-" * 60)
        print(f"{'Depth':<8} {'Mean':>12} {'Std':>10} {'N':>6}")
        print("-" * 60)

        for depth in sorted(td3_depth_results[env].keys()):
            data = td3_depth_results[env][depth]
            print(f"{depth:<8} {data['mean']:>12.2f} {data['std']:>10.2f} {data.get('n_seeds', 'N/A'):>6}")

    print("=" * 80)


def main():
    print("Collecting results...")

    # Collect all results
    dqn_results = collect_dqn_results()
    td3_utd_results = collect_td3_utd_results()
    td3_depth_results = collect_td3_depth_results()

    # Print summary
    print_summary_table(dqn_results, td3_utd_results, td3_depth_results)

    # Create plots
    print("\nGenerating plots...")

    plot_dqn_depth_study(dqn_results)
    plot_td3_utd_study(td3_utd_results)
    plot_td3_depth_study(td3_depth_results)

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
