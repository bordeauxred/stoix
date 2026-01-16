#!/bin/bash
#SBATCH --job-name=depth_10M
#SBATCH --partition=kisski
#SBATCH --array=0-17%6
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
