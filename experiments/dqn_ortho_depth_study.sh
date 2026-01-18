#!/bin/bash
#SBATCH --job-name=dqn_ortho_depth
#SBATCH --partition=kisski
#SBATCH --array=0-31%16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/dqn_ortho_depth_%A_%a.out
#SBATCH --error=logs/dqn_ortho_depth_%A_%a.err
#SBATCH -C inet
#SBATCH --exclude=ggpu188

set -e

module purge
module load git
module load gcc/13.2.0
module load cuda/12.6.2
module load cudnn/9.8.0.87-12

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
mkdir -p logs

TASK_ID=$SLURM_ARRAY_TASK_ID

# ============================================================================
# DQN Orthogonalization Depth Study - GroupSort + Gram Regularization
#
# Study design:
#   - 4 environments: MinAtar games
#   - 8 network depths: 2, 4, 8, 16, 32, 64, 128, 256 layers
#   - Configuration: GroupSort activation + Gram regularization (lambda=0.2)
#   - 5 seeds per configuration (vmapped parallel training)
#   - Total: 8 × 4 = 32 SLURM tasks
#
# Naming convention: dqn_gs_gram_depth{D}_{env}_{timestamp}
#   where D = number of hidden layers
#
# WandB project: stoix_dqn_ortho_depth_study
# ============================================================================

SEEDS="[42,43,44,45,46]"
NUM_ENVS=64
STEPS=10000000  # 10M steps for MinAtar
SYSTEM="stoix/systems/q_learning/ff_dqn_ortho.py"

# Parse task indices
# Task ID = env_idx * 8 + depth_idx
# Where: env_idx in [0-3], depth_idx in [0-7]
ENV_IDX=$((TASK_ID / 8))
DEPTH_IDX=$((TASK_ID % 8))

# Environments
case $ENV_IDX in
    0) ENV="gymnax/breakout"; ENV_SHORT="minatar_breakout" ;;
    1) ENV="gymnax/asterix"; ENV_SHORT="minatar_asterix" ;;
    2) ENV="gymnax/freeway"; ENV_SHORT="minatar_freeway" ;;
    3) ENV="gymnax/space_invaders"; ENV_SHORT="minatar_spaceinv" ;;
esac

# Network depths (number of hidden layers)
# Layer sizes are 256 repeated DEPTH times
case $DEPTH_IDX in
    0) DEPTH=2; LAYERS="[256,256]" ;;
    1) DEPTH=4; LAYERS="[256,256,256,256]" ;;
    2) DEPTH=8; LAYERS="[256,256,256,256,256,256,256,256]" ;;
    3) DEPTH=16; LAYERS="[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]" ;;
    4) DEPTH=32; LAYERS="[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]" ;;
    5) DEPTH=64; LAYERS="[$(printf '256,%.0s' {1..64} | sed 's/,$//')]" ;;
    6) DEPTH=128; LAYERS="[$(printf '256,%.0s' {1..128} | sed 's/,$//')]" ;;
    7) DEPTH=256; LAYERS="[$(printf '256,%.0s' {1..256} | sed 's/,$//')]" ;;
esac

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="dqn_gs_gram_depth${DEPTH}_${ENV_SHORT}_${TIMESTAMP}"

echo "============================================================"
echo "DQN Orthogonalization Depth Study - GroupSort + Gram"
echo "============================================================"
echo "Task ID:       $TASK_ID"
echo "Environment:   $ENV ($ENV_SHORT)"
echo "Activation:    groupsort"
echo "Ortho Lambda:  0.2"
echo "Depth:         $DEPTH layers"
echo "Seeds:         $SEEDS"
echo "Run name:      $RUN_NAME"
echo "============================================================"

# Run the experiment
uv run python $SYSTEM \
    env=$ENV \
    arch.seed=42 \
    arch.total_timesteps=$STEPS \
    arch.total_num_envs=$NUM_ENVS \
    +multiseed=$SEEDS \
    network.actor_network.pre_torso.layer_sizes="$LAYERS" \
    network.actor_network.pre_torso.activation=groupsort \
    system.ortho_lambda=0.2 \
    logger.loggers.wandb.enabled=True \
    logger.loggers.wandb.project=stoix_dqn_ortho_depth_study \
    "logger.loggers.wandb.tag=[ortho_depth_study,gs_gram,depth_$DEPTH]"

echo "============================================================"
echo "Task $TASK_ID completed: gs_gram depth=$DEPTH on $ENV_SHORT"
echo "============================================================"
