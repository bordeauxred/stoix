#!/bin/bash
#SBATCH --job-name=val_multiseed
#SBATCH --partition=kisski
#SBATCH --array=0-5%6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/val_multiseed_%A_%a.out
#SBATCH --error=logs/val_multiseed_%A_%a.err
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
# Multi-Seed Validation with PureJaxRL-Style vmap
#
# This script validates the new Option 3 (PureJaxRL-style) multi-seed training.
# Each algorithm runs 5 seeds in parallel on a single GPU using vmap.
#
# Usage:
#   sbatch experiments/validate_multiseed.sh
#
# Key feature: +multiseed=[42,43,44,45,46] triggers vmapped multi-seed training
# ============================================================================

STEPS=500000  # 500K steps for quick validation
SEEDS="[42,43,44,45,46]"

# UTD-fixed settings for off-policy algorithms
NUM_ENVS=64
EPOCHS=16
WARMUP=200

COMMON="arch.total_num_envs=$NUM_ENVS system.total_buffer_size=1000000"
UTD_FIX="system.epochs=$EPOCHS system.warmup_steps=$WARMUP"
HP_FIX="system.actor_lr=3e-4 system.q_lr=3e-4 system.decay_learning_rates=false"
LAYERS="network.actor_network.pre_torso.layer_sizes=[256,256] network.q_network.pre_torso.layer_sizes=[256,256]"

# DQN settings
DQN_COMMON="arch.total_num_envs=64"

case $TASK_ID in
    # ========================================
    # DQN Multi-seed on MinAtar
    # ========================================
    0)  ALGO="dqn"; ENV="gymnax/breakout"; SYSTEM="stoix/systems/q_learning/ff_dqn.py"
        EXTRA="$DQN_COMMON" ;;
    1)  ALGO="dqn"; ENV="gymnax/asterix"; SYSTEM="stoix/systems/q_learning/ff_dqn.py"
        EXTRA="$DQN_COMMON" ;;

    # ========================================
    # TD3 Multi-seed with UTD fix
    # ========================================
    2)  ALGO="td3"; ENV="gymnax/pendulum"; SYSTEM="stoix/systems/ddpg/ff_td3.py"
        EXTRA="$COMMON $UTD_FIX $HP_FIX $LAYERS" ;;
    3)  ALGO="td3"; ENV="brax/halfcheetah"; SYSTEM="stoix/systems/ddpg/ff_td3.py"
        EXTRA="$COMMON $UTD_FIX $HP_FIX $LAYERS env.kwargs.backend=generalized" ;;

    # ========================================
    # Continuous control baselines
    # ========================================
    4)  ALGO="td3"; ENV="mjc_playground/dm_control/cartpole_balance"; SYSTEM="stoix/systems/ddpg/ff_td3.py"
        EXTRA="$COMMON $UTD_FIX $HP_FIX $LAYERS" ;;
    5)  ALGO="td3"; ENV="brax/ant"; SYSTEM="stoix/systems/ddpg/ff_td3.py"
        EXTRA="$COMMON $UTD_FIX $HP_FIX $LAYERS env.kwargs.backend=generalized" ;;
esac

ENV_SHORT=$(echo $ENV | sed 's/\//_/g')
RUN_NAME="multiseed_${ALGO}_${ENV_SHORT}_$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "Task $TASK_ID: $ALGO on $ENV (5 seeds in parallel)"
echo "Seeds: $SEEDS"
echo "Run name: $RUN_NAME"
echo "============================================================"

uv run python $SYSTEM \
    env=$ENV \
    arch.seed=42 \
    arch.total_timesteps=$STEPS \
    +multiseed=$SEEDS \
    logger.loggers.wandb.enabled=True \
    logger.loggers.wandb.project=stoix_multiseed_validation \
    logger.loggers.wandb.name=$RUN_NAME \
    logger.loggers.json.enabled=True \
    logger.loggers.json.path=multiseed_validation/${ALGO}_${ENV_SHORT} \
    $EXTRA

echo "============================================================"
echo "Task $TASK_ID ($ALGO on $ENV) completed"
echo "============================================================"
