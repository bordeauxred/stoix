#!/usr/bin/env python3
"""Depth scaling baseline experiments for isometric NN study.

Runs TD3 with relu activation at depths 2, 4, 8, 16, 32 (width 256)
across HalfCheetah, Ant, and Hopper environments.

Estimated runtime (10M steps per run, 45 total runs):
- Sequential: ~8-12 hours (depends on hardware)
- With reduced num_envs for deep networks to fit in memory

Usage:
    python experiments/depth_baseline.py              # Run all
    python experiments/depth_baseline.py --depth 2   # Run only depth 2
    python experiments/depth_baseline.py --dry-run   # Show what would run
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# === Configuration ===
ENVS = ["brax/halfcheetah", "brax/ant", "brax/hopper"]
DEPTHS = [2, 4, 8, 16, 32]
SEEDS = [42, 43, 44]
WIDTH = 256
ACTIVATION = "relu"
TOTAL_TIMESTEPS = 10_000_000
EXPERIMENT_NAME = "depth_baseline_td3"

# WandB configuration
WANDB_PROJECT = "isometric-nn-study"
WANDB_ENABLED = True

# Memory management: reduce envs for deeper networks to avoid OOM
# Also affects throughput (fewer envs = slower but fits in memory)
DEPTH_CONFIG = {
    2:  {"num_envs": 1024, "est_steps_per_sec": 25000},
    4:  {"num_envs": 1024, "est_steps_per_sec": 20000},
    8:  {"num_envs": 512,  "est_steps_per_sec": 12000},
    16: {"num_envs": 256,  "est_steps_per_sec": 6000},
    32: {"num_envs": 128,  "est_steps_per_sec": 3000},
}


def estimate_runtime() -> str:
    """Estimate total runtime for all experiments."""
    total_seconds = 0
    for depth in DEPTHS:
        config = DEPTH_CONFIG[depth]
        runs_at_depth = len(ENVS) * len(SEEDS)
        seconds_per_run = TOTAL_TIMESTEPS / config["est_steps_per_sec"]
        total_seconds += runs_at_depth * seconds_per_run

    hours = total_seconds / 3600
    return f"~{hours:.1f} hours"


def build_layer_sizes(depth: int, width: int = WIDTH) -> str:
    """Build Hydra list override for layer_sizes."""
    return "[" + ",".join([str(width)] * depth) + "]"


def run_experiment(depth: int, env: str, seed: int, dry_run: bool = False) -> tuple:
    """Run a single TD3 experiment.

    Returns: (depth, env, seed, success, duration_seconds)
    """
    config = DEPTH_CONFIG[depth]
    layer_sizes = build_layer_sizes(depth)
    env_name = env.split("/")[1]

    cmd = [
        "uv", "run", "--frozen", "python",
        "stoix/systems/ddpg/ff_td3.py",
        f"env={env}",
        f"arch.seed={seed}",
        f"arch.total_timesteps={TOTAL_TIMESTEPS}",
        f"arch.total_num_envs={config['num_envs']}",
        # Actor network
        f"network.actor_network.pre_torso.layer_sizes={layer_sizes}",
        f"network.actor_network.pre_torso.activation={ACTIVATION}",
        # Q network (same architecture)
        f"network.q_network.pre_torso.layer_sizes={layer_sizes}",
        f"network.q_network.pre_torso.activation={ACTIVATION}",
        # JSON logging
        "logger.loggers.json.enabled=True",
        f"logger.loggers.json.path={EXPERIMENT_NAME}",
    ]

    # WandB logging
    if WANDB_ENABLED:
        cmd.extend([
            "logger.loggers.wandb.enabled=True",
            f"logger.loggers.wandb.project={WANDB_PROJECT}",
            f"logger.loggers.wandb.group_tag={EXPERIMENT_NAME}",
            f"logger.loggers.wandb.tag=[depth_{depth},{env_name},seed_{seed},{ACTIVATION}]",
        ])

    print(f"\n{'='*60}")
    print(f"depth={depth} | env={env_name} | seed={seed} | envs={config['num_envs']}")
    print(f"{'='*60}")

    if dry_run:
        print(f"[DRY RUN] Would execute:")
        print(f"  {' '.join(cmd)}")
        return (depth, env, seed, True, 0)

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        print(f"Completed in {duration/60:.1f} minutes")
        return (depth, env, seed, True, duration)
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"FAILED after {duration/60:.1f} minutes: {e}")
        return (depth, env, seed, False, duration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, choices=DEPTHS,
                        help="Run only specific depth")
    parser.add_argument("--env", type=str, choices=ENVS,
                        help="Run only specific environment")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    args = parser.parse_args()

    # Filter based on args
    depths = [args.depth] if args.depth else DEPTHS
    envs = [args.env] if args.env else ENVS

    total_runs = len(depths) * len(envs) * len(SEEDS)

    print(f"{'='*60}")
    print(f"Depth Baseline Study - TD3 with ReLU")
    print(f"{'='*60}")
    print(f"  Depths: {depths}")
    print(f"  Envs: {[e.split('/')[1] for e in envs]}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Total runs: {total_runs}")
    print(f"  Estimated time: {estimate_runtime()}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No experiments will be executed]\n")

    results = []
    start_time = time.time()
    current = 0

    for depth in depths:
        for env in envs:
            for seed in SEEDS:
                current += 1
                remaining = total_runs - current
                elapsed = time.time() - start_time

                if current > 1 and not args.dry_run:
                    avg_time = elapsed / (current - 1)
                    eta = timedelta(seconds=avg_time * remaining)
                    print(f"\n[{current}/{total_runs}] ETA: {eta}")
                else:
                    print(f"\n[{current}/{total_runs}]")

                result = run_experiment(depth, env, seed, args.dry_run)
                results.append(result)

    # Summary
    total_time = time.time() - start_time
    failed = [r for r in results if not r[3]]

    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(results) - len(failed)}/{total_runs} succeeded")
    print(f"Total time: {total_time/3600:.2f} hours")

    if failed:
        print(f"\nFailed experiments:")
        for d, e, s, _, _ in failed:
            print(f"  depth={d}, env={e.split('/')[1]}, seed={s}")

    print(f"{'='*60}")
    print(f"\nResults saved to: results/json/{EXPERIMENT_NAME}/")
    print(f"To plot: python plotting/plot_seeds.py --path results/json/{EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
