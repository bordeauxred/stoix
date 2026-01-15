#!/usr/bin/env python3
"""SLURM-optimized depth baseline experiments.

Generates commands for SLURM array jobs or runs a single experiment
based on SLURM_ARRAY_TASK_ID.

Usage:
    # Generate all commands (for review)
    python experiments/depth_baseline_slurm.py --list

    # Run specific task ID (called by SLURM)
    python experiments/depth_baseline_slurm.py --task-id 0

    # Generate sbatch script
    python experiments/depth_baseline_slurm.py --generate-sbatch > submit.sh
"""

import argparse
import subprocess
import sys
import os
from itertools import product

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

# A100 can handle more envs than consumer GPUs
DEPTH_CONFIG = {
    2:  {"num_envs": 2048},
    4:  {"num_envs": 2048},
    8:  {"num_envs": 1024},
    16: {"num_envs": 512},
    32: {"num_envs": 256},
}


def get_all_experiments():
    """Generate all experiment configurations as a list."""
    experiments = []
    for depth, env, seed in product(DEPTHS, ENVS, SEEDS):
        experiments.append({
            "depth": depth,
            "env": env,
            "seed": seed,
            "num_envs": DEPTH_CONFIG[depth]["num_envs"],
        })
    return experiments


def build_command(exp: dict) -> list:
    """Build the command for a single experiment."""
    depth = exp["depth"]
    env_name = exp["env"].split("/")[1]
    layer_sizes = "[" + ",".join(["256"] * depth) + "]"

    cmd = [
        "uv", "run", "--frozen", "python",
        "stoix/systems/ddpg/ff_td3.py",
        f"env={exp['env']}",
        f"arch.seed={exp['seed']}",
        f"arch.total_timesteps={TOTAL_TIMESTEPS}",
        f"arch.total_num_envs={exp['num_envs']}",
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
            f"logger.loggers.wandb.tag=[depth_{depth},{env_name},seed_{exp['seed']},{ACTIVATION}]",
        ])

    return cmd


def generate_sbatch_script():
    """Generate SLURM sbatch script."""
    num_tasks = len(get_all_experiments())
    return f'''#!/bin/bash
#SBATCH --job-name=depth_baseline
#SBATCH --array=0-{num_tasks - 1}%4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/depth_baseline_%A_%a.out
#SBATCH --error=logs/depth_baseline_%A_%a.err

# Create logs directory
mkdir -p logs

# Activate environment (adjust as needed)
# source /path/to/venv/bin/activate
# or: module load python/3.10

# WandB API key (set this or run `wandb login` before submitting)
# export WANDB_API_KEY="your-api-key-here"

# Offline mode if no internet on compute nodes (syncs after job)
# export WANDB_MODE="offline"

# Run the experiment for this array task
python experiments/depth_baseline_slurm.py --task-id $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID completed"
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
    args = parser.parse_args()

    experiments = get_all_experiments()

    if args.generate_sbatch:
        print(generate_sbatch_script())
        return

    if args.list:
        print(f"Total experiments: {len(experiments)}")
        print(f"{'ID':<4} {'Depth':<6} {'Env':<15} {'Seed':<6} {'Envs':<6}")
        print("-" * 45)
        for i, exp in enumerate(experiments):
            env_name = exp["env"].split("/")[1]
            print(f"{i:<4} {exp['depth']:<6} {env_name:<15} {exp['seed']:<6} {exp['num_envs']:<6}")
        return

    if args.task_id is not None:
        # Check for SLURM environment variable as fallback
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

    print(f"Running task {task_id}: depth={exp['depth']}, env={exp['env']}, seed={exp['seed']}")
    print(f"Command: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
