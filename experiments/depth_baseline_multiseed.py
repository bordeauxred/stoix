#!/usr/bin/env python3
"""SLURM-optimized depth baseline experiments using vmapped multi-seed training.

Runs 3 seeds in parallel per job using vmap, reducing from 45 jobs to 15 jobs.
Each seed trains independently but shares the GPU efficiently.

Usage:
    # Generate all commands (for review)
    python experiments/depth_baseline_multiseed.py --list

    # Run specific task ID (called by SLURM)
    python experiments/depth_baseline_multiseed.py --task-id 0

    # Generate sbatch script
    python experiments/depth_baseline_multiseed.py --generate-sbatch > submit_multiseed.sh

    # Dry run locally
    python experiments/depth_baseline_multiseed.py --task-id 0 --dry-run
"""

import argparse
import subprocess
import sys
import os
from itertools import product

# === Configuration ===
ENVS = ["brax/halfcheetah", "brax/ant", "brax/hopper"]
DEPTHS = [2, 4, 8, 16, 32]
SEEDS = [42, 43, 44]  # All seeds run in parallel per job
WIDTH = 256
ACTIVATION = "relu"
TOTAL_TIMESTEPS = 10_000_000
EXPERIMENT_NAME = "depth_baseline_td3_multiseed"

# WandB configuration
WANDB_PROJECT = "isometric-nn-study"
WANDB_ENABLED = True

# A100 can handle more envs than consumer GPUs
# With 3 seeds vmapped, we use slightly fewer envs per seed
# Buffer size must be >= warmup_steps * num_envs per seed
DEPTH_CONFIG = {
    2:  {"num_envs": 2048, "buffer_size": 3_000_000},  # ~680 per seed
    4:  {"num_envs": 2048, "buffer_size": 3_000_000},  # ~680 per seed
    8:  {"num_envs": 1024, "buffer_size": 2_000_000},  # ~340 per seed
    16: {"num_envs": 512,  "buffer_size": 1_500_000},  # ~170 per seed
    32: {"num_envs": 256,  "buffer_size": 1_000_000},  # ~85 per seed
}


def get_all_experiments():
    """Generate all experiment configurations as a list.

    With multi-seed training, we only iterate over (depth, env).
    Seeds are handled within each job via vmap.
    """
    experiments = []
    for depth, env in product(DEPTHS, ENVS):
        experiments.append({
            "depth": depth,
            "env": env,
            "seeds": SEEDS,
            "num_envs": DEPTH_CONFIG[depth]["num_envs"],
            "buffer_size": DEPTH_CONFIG[depth]["buffer_size"],
        })
    return experiments


def build_command(exp: dict) -> list:
    """Build the command for a single experiment (runs all seeds)."""
    depth = exp["depth"]
    env_name = exp["env"].split("/")[1]
    layer_sizes = "[" + ",".join(["256"] * depth) + "]"
    seeds_str = "[" + ",".join(str(s) for s in exp["seeds"]) + "]"
    num_seeds = len(exp["seeds"])

    cmd = [
        "uv", "run", "--frozen", "python",
        "stoix/systems/ddpg/ff_td3_multiseed.py",
        f"env={exp['env']}",
        f"arch.seed={exp['seeds'][0]}",  # Base seed for RNG
        f"+arch.seeds={seeds_str}",  # All seeds for logging (+ adds new key)
        f"arch.update_batch_size={num_seeds}",  # Vmap dimension
        f"arch.total_timesteps={TOTAL_TIMESTEPS}",
        f"arch.total_num_envs={exp['num_envs']}",
        f"system.total_buffer_size={exp['buffer_size']}",  # Ensure buffer fits warmup
        f"network.actor_network.pre_torso.layer_sizes={layer_sizes}",
        f"network.actor_network.pre_torso.activation={ACTIVATION}",
        f"network.q_network.pre_torso.layer_sizes={layer_sizes}",
        f"network.q_network.pre_torso.activation={ACTIVATION}",
        # JSON logging for local aggregation
        "logger.loggers.json.enabled=True",
        f"logger.loggers.json.path={EXPERIMENT_NAME}",
    ]

    # WandB logging for remote monitoring
    if WANDB_ENABLED:
        cmd.extend([
            "logger.loggers.wandb.enabled=True",
            f"logger.loggers.wandb.project={WANDB_PROJECT}",
            f"logger.loggers.wandb.group_tag={EXPERIMENT_NAME}",
            f"logger.loggers.wandb.tag=[depth_{depth},{env_name},{ACTIVATION},multiseed]",
        ])

    return cmd


def generate_sbatch_script():
    """Generate SLURM sbatch script for multi-seed experiments."""
    num_tasks = len(get_all_experiments())
    return f'''#!/bin/bash
#SBATCH --job-name=depth_multiseed
#SBATCH --partition=kisski
#SBATCH --array=0-{num_tasks - 1}%4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/depth_multiseed_%A_%a.out
#SBATCH --error=logs/depth_multiseed_%A_%a.err
#SBATCH -C inet

set -e

# Add uv to PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

mkdir -p logs

# Load secrets from .env file
if [ -f experiments/.env ]; then
    export $(grep -v '^#' experiments/.env | xargs)
fi

# Run the experiment for this array task
# Each task runs 3 seeds in parallel via vmap
uv run --frozen python experiments/depth_baseline_multiseed.py --task-id $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID completed (3 seeds)"
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="List all experiments with task IDs")
    parser.add_argument("--task-id", type=int,
                        help="Run specific task ID (for SLURM array)")
    parser.add_argument("--generate-sbatch", action="store_true",
                        help="Generate sbatch script")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command without executing")
    args = parser.parse_args()

    experiments = get_all_experiments()

    if args.generate_sbatch:
        print(generate_sbatch_script())
        return

    if args.list:
        print(f"Total experiments: {len(experiments)} (each runs {len(SEEDS)} seeds in parallel)")
        print(f"Total seed-runs: {len(experiments) * len(SEEDS)}")
        print(f"\n{'ID':<4} {'Depth':<6} {'Env':<15} {'Seeds':<15} {'Envs':<6}")
        print("-" * 50)
        for i, exp in enumerate(experiments):
            env_name = exp["env"].split("/")[1]
            seeds_str = ",".join(str(s) for s in exp["seeds"])
            print(f"{i:<4} {exp['depth']:<6} {env_name:<15} {seeds_str:<15} {exp['num_envs']:<6}")
        return

    if args.task_id is not None:
        task_id = args.task_id
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        parser.print_help()
        sys.exit(1)

    if task_id < 0 or task_id >= len(experiments):
        print(f"Error: task_id {task_id} out of range [0, {len(experiments)-1}]")
        sys.exit(1)

    exp = experiments[task_id]
    cmd = build_command(exp)

    print(f"Running task {task_id}: depth={exp['depth']}, env={exp['env']}, seeds={exp['seeds']}")
    print(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        print("\n[DRY RUN] Command not executed")
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
