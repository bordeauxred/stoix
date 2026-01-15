#!/bin/bash
#SBATCH --job-name=depth_baseline
#SBATCH --partition=kisski
#SBATCH --array=0-44%4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/depth_baseline_%A_%a.out
#SBATCH --error=logs/depth_baseline_%A_%a.err
#SBATCH -C inet

mkdir -p logs

# Load secrets from .env file
if [ -f experiments/.env ]; then
    export $(grep -v '^#' experiments/.env | xargs)
fi

# Run the experiment for this array task
uv run python experiments/depth_baseline_slurm.py --task-id $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID completed"
