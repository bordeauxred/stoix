#!/bin/bash
#SBATCH --job-name=depth_multiseed
#SBATCH --partition=kisski
#SBATCH --array=0-14%4
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

# Add uv to PATH (installed via: curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

mkdir -p logs

# Load secrets from .env file
if [ -f experiments/.env ]; then
    export $(grep -v '^#' experiments/.env | xargs)
fi

# Run the experiment for this array task
# Each task runs 3 seeds in parallel via vmap (45 total seed-runs in 15 jobs)
uv run --frozen python experiments/depth_baseline_multiseed.py --task-id $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID completed (3 seeds via vmap)"
