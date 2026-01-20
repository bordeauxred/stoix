#!/bin/bash
#SBATCH --job-name=dqn_adamo_tune
#SBATCH --partition=kisski
#SBATCH --array=0-15%8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/dqn_adamo_tune_%A_%a.out
#SBATCH --error=logs/dqn_adamo_tune_%A_%a.err
#SBATCH -C inet

set -e

module purge
module load git gcc/13.2.0 cuda/12.6.2 cudnn/9.8.0.87-12

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p logs

TASK_ID=${SLURM_ARRAY_TASK_ID:-$1}  # Support both SLURM and manual run

# ============================================================================
# DQN AdamO Tuning - Phase 1
#
# Goal: Find optimal ortho_coeff for AdamO
#
# Design:
#   - 4 environments × 4 configs = 16 jobs
#   - Configs: loss (baseline), adamo_1e-4, adamo_1e-3, adamo_1e-2
#   - Network: 4 layers × 256 (fast)
#   - 5 seeds vmapped
#   - 10M steps
#
# Task mapping:
#   Task ID = env_idx * 4 + config_idx
# ============================================================================

SEEDS="[42,43,44,45,46]"
NUM_ENVS=64
STEPS=10000000
NUM_EVAL=100  # Log every ~100k steps
SYSTEM="stoix/systems/q_learning/ff_dqn.py"
LAYERS="[256,256,256,256]"

# Parse task indices
ENV_IDX=$((TASK_ID / 4))
CONFIG_IDX=$((TASK_ID % 4))

# Environments
ENVS=("gymnax/breakout" "gymnax/asterix" "gymnax/freeway" "gymnax/space_invaders")
ENV_SHORTS=("breakout" "asterix" "freeway" "spaceinv")
ENV=${ENVS[$ENV_IDX]}
ENV_SHORT=${ENV_SHORTS[$ENV_IDX]}

# Configs: mode and coefficient
case $CONFIG_IDX in
    0) MODE="loss"; COEFF_ARG="system.ortho_lambda=0.2"; TAG="loss" ;;
    1) MODE="optimizer"; COEFF_ARG="system.ortho_coeff=0.0001"; TAG="adamo_1e-4" ;;
    2) MODE="optimizer"; COEFF_ARG="system.ortho_coeff=0.001"; TAG="adamo_1e-3" ;;
    3) MODE="optimizer"; COEFF_ARG="system.ortho_coeff=0.01"; TAG="adamo_1e-2" ;;
esac

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="dqn_${TAG}_${ENV_SHORT}_${TIMESTAMP}"

echo "============================================================"
echo "DQN AdamO Tuning - Phase 1"
echo "============================================================"
echo "Task ID:       $TASK_ID"
echo "Environment:   $ENV ($ENV_SHORT)"
echo "Mode:          $MODE"
echo "Config:        $TAG"
echo "Depth:         4 layers"
echo "Seeds:         $SEEDS"
echo "============================================================"

uv run python $SYSTEM \
    env=$ENV \
    arch.seed=42 \
    arch.total_timesteps=$STEPS \
    arch.total_num_envs=$NUM_ENVS \
    arch.num_evaluation=$NUM_EVAL \
    +multiseed=$SEEDS \
    network.actor_network.pre_torso.layer_sizes="$LAYERS" \
    network.actor_network.pre_torso.activation=groupsort \
    system.ortho_mode=$MODE \
    $COEFF_ARG \
    system.log_spectral_freq=1000000 \
    logger.loggers.wandb.enabled=True \
    logger.loggers.wandb.project=stoix_adamo_tuning \
    "logger.loggers.wandb.tag=[adamo_tuning,${TAG},${ENV_SHORT}]" \
    +logger.loggers.file.enabled=True

echo "============================================================"
echo "Task $TASK_ID completed: $TAG on $ENV_SHORT"
echo "============================================================"
