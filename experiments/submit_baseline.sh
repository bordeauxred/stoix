#!/bin/bash
#SBATCH --job-name=depth_baseline
#SBATCH --array=0-44%4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/depth_baseline_%A_%a.out
#SBATCH --error=logs/depth_baseline_%A_%a.err

mkdir -p logs

# Run the experiment for this array task
uv run python experiments/depth_baseline_slurm.py --task-id $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID completed"
