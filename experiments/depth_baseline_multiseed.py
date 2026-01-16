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

    # Force rerun even if completed
    python experiments/depth_baseline_multiseed.py --task-id 0 --force
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from itertools import product

# === Configuration ===
ENVS = ["brax/halfcheetah", "brax/ant", "brax/hopper"]
DEPTHS = [2, 4, 8, 16, 32, 64]
SEEDS = [42, 43, 44]  # All seeds run in parallel per job
WIDTH = 256
ACTIVATION = "relu"
TOTAL_TIMESTEPS = 10_000_000  # Full 10M run
EXPERIMENT_NAME = "depth_baseline_td3_10M"

# Hyperparameter improvements based on literature (FastTD3, Brax paper, SB3)
LEARNING_RATE = 3e-4  # Higher than default 1e-4 (FastTD3/SB3 use 3e-4 to 1e-3)
DECAY_LR = False  # Disable LR decay - harmful for learning

# WandB configuration
WANDB_PROJECT = "isometric-nn-study"
WANDB_ENABLED = True

# A100 can handle more envs than consumer GPUs
# With 3 seeds vmapped, we use slightly fewer envs per seed
# Buffer size must be >= warmup_steps * num_envs per seed
DEPTH_CONFIG = {
    2:  {"num_envs": 2040, "buffer_size": 3_000_000},  # 680 * 3
    4:  {"num_envs": 2040, "buffer_size": 3_000_000},  # 680 * 3
    8:  {"num_envs": 1020, "buffer_size": 2_000_000},  # 340 * 3
    16: {"num_envs": 510,  "buffer_size": 1_500_000},  # 170 * 3
    32: {"num_envs": 255,  "buffer_size": 1_000_000},  # 85 * 3
    64: {"num_envs": 126,  "buffer_size": 500_000},    # 42 * 3
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


# Directory for completion markers
COMPLETION_DIR = Path("logs/completed")


def get_completion_marker(exp: dict) -> Path:
    """Get the path to the completion marker for an experiment."""
    env_name = exp["env"].split("/")[1]
    marker_name = f"{EXPERIMENT_NAME}_d{exp['depth']}_{env_name}.done"
    return COMPLETION_DIR / marker_name


def is_completed(exp: dict) -> bool:
    """Check if an experiment has already completed successfully."""
    return get_completion_marker(exp).exists()


def mark_completed(exp: dict) -> None:
    """Mark an experiment as completed."""
    COMPLETION_DIR.mkdir(parents=True, exist_ok=True)
    marker = get_completion_marker(exp)
    marker.write_text(f"Completed: depth={exp['depth']}, env={exp['env']}, seeds={exp['seeds']}\n")


def build_command(exp: dict) -> list:
    """Build the command for a single experiment (runs all seeds)."""
    depth = exp["depth"]
    env_name = exp["env"].split("/")[1]
    layer_sizes = "[" + ",".join(["256"] * depth) + "]"
    seeds_str = "[" + ",".join(str(s) for s in exp["seeds"]) + "]"
    num_seeds = len(exp["seeds"])

    cmd = [
        "uv", "run", "python",
        "stoix/systems/ddpg/ff_td3_multiseed.py",
        f"env={exp['env']}",
        f"arch.seed={exp['seeds'][0]}",  # Base seed for RNG
        f"+arch.seeds={seeds_str}",  # All seeds for logging (+ adds new key)
        f"arch.update_batch_size={num_seeds}",  # Vmap dimension
        f"arch.total_timesteps={TOTAL_TIMESTEPS}",
        f"arch.total_num_envs={exp['num_envs']}",
        f"system.total_buffer_size={exp['buffer_size']}",  # Ensure buffer fits warmup
        # Hyperparameter improvements
        f"system.actor_lr={LEARNING_RATE}",
        f"system.q_lr={LEARNING_RATE}",
        f"system.decay_learning_rates={str(DECAY_LR).lower()}",
        # Network architecture
        f"network.actor_network.pre_torso.layer_sizes={layer_sizes}",
        f"network.actor_network.pre_torso.activation={ACTIVATION}",
        f"network.q_network.pre_torso.layer_sizes={layer_sizes}",
        f"network.q_network.pre_torso.activation={ACTIVATION}",
        # JSON logging with unique path per experiment (avoids race conditions)
        "logger.loggers.json.enabled=True",
        f"logger.loggers.json.path={EXPERIMENT_NAME}/d{depth}_{env_name}",
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
#SBATCH --job-name=depth_10M
#SBATCH --partition=kisski
#SBATCH --array=0-{num_tasks - 1}%6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/depth_10M_%A_%a.out
#SBATCH --error=logs/depth_10M_%A_%a.err
#SBATCH -C inet
# Exclude nodes with known GPU issues
#SBATCH --exclude=ggpu188

set -e

# Load required modules
module purge
module load git
module load gcc/13.2.0
module load cuda/12.6.2
module load cudnn/9.8.0.87-12

# Add uv to PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

mkdir -p logs

# Load secrets from .env file
if [ -f experiments/.env ]; then
    export $(grep -v '^#' experiments/.env | xargs)
fi

# Run the experiment for this array task
# Each task runs 3 seeds in parallel via vmap (10M timesteps)
uv run python experiments/depth_baseline_multiseed.py --task-id $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID completed (3 seeds, 10M steps)"
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
    parser.add_argument("--force", action="store_true",
                        help="Force rerun even if already completed")
    parser.add_argument("--reset-completed", action="store_true",
                        help="Clear all completion markers")
    args = parser.parse_args()

    experiments = get_all_experiments()

    if args.reset_completed:
        import shutil
        if COMPLETION_DIR.exists():
            shutil.rmtree(COMPLETION_DIR)
            print(f"Cleared completion markers from {COMPLETION_DIR}")
        else:
            print("No completion markers to clear")
        return

    if args.generate_sbatch:
        print(generate_sbatch_script())
        return

    if args.list:
        print(f"Total experiments: {len(experiments)} (each runs {len(SEEDS)} seeds in parallel)")
        print(f"Total seed-runs: {len(experiments) * len(SEEDS)}")
        completed_count = sum(1 for exp in experiments if is_completed(exp))
        print(f"Completed: {completed_count}/{len(experiments)}")
        print(f"\n{'ID':<4} {'Depth':<6} {'Env':<15} {'Seeds':<15} {'Envs':<6} {'Status':<10}")
        print("-" * 65)
        for i, exp in enumerate(experiments):
            env_name = exp["env"].split("/")[1]
            seeds_str = ",".join(str(s) for s in exp["seeds"])
            status = "✓ done" if is_completed(exp) else "pending"
            print(f"{i:<4} {exp['depth']:<6} {env_name:<15} {seeds_str:<15} {exp['num_envs']:<6} {status:<10}")
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

    # Check if already completed
    if is_completed(exp) and not args.force:
        print(f"Task {task_id} already completed (depth={exp['depth']}, env={exp['env']})")
        print(f"Use --force to rerun. Skipping.")
        return

    cmd = build_command(exp)

    print(f"Running task {task_id}: depth={exp['depth']}, env={exp['env']}, seeds={exp['seeds']}")
    print(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        print("\n[DRY RUN] Command not executed")
        return

    subprocess.run(cmd, check=True)

    # Mark as completed only if subprocess succeeded (check=True raises on failure)
    mark_completed(exp)
    print(f"Task {task_id} completed successfully. Marker: {get_completion_marker(exp)}")


if __name__ == "__main__":
    main()
