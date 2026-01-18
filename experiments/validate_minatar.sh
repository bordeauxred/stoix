#!/bin/bash
#SBATCH --job-name=val_minatar
#SBATCH --partition=kisski
#SBATCH --array=0-11%6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/val_minatar_%A_%a.out
#SBATCH --error=logs/val_minatar_%A_%a.err
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
# MinAtar Validation Suite
#
# Testing DQN, Rainbow, and PPO on all 4 MinAtar games (via Gymnax)
#
# Literature baselines (approximate):
# - Breakout: DQN ~15-20, PPO (tuned) ~28-58+
# - Asterix: DQN ~10-20
# - Freeway: DQN ~50-60
# - Space Invaders: DQN ~50-100
#
# Note: Stoix DQN has epochs=16, Rainbow has epochs=128
# This gives much better UTD ratio than TD3 (epochs=1)
# ============================================================================

STEPS=5000000  # 5M steps (standard MinAtar benchmark)

# Common settings
DQN_COMMON="arch.total_num_envs=64"
RAINBOW_COMMON="arch.total_num_envs=64"
PPO_COMMON="arch.total_num_envs=2048"

case $TASK_ID in
    # ========================================
    # DQN on all MinAtar games
    # ========================================
    0)  ALGO="dqn"; ENV="gymnax/breakout"; SYSTEM="stoix/systems/q_learning/ff_dqn.py"
        EXTRA="$DQN_COMMON" ;;
    1)  ALGO="dqn"; ENV="gymnax/asterix"; SYSTEM="stoix/systems/q_learning/ff_dqn.py"
        EXTRA="$DQN_COMMON" ;;
    2)  ALGO="dqn"; ENV="gymnax/freeway"; SYSTEM="stoix/systems/q_learning/ff_dqn.py"
        EXTRA="$DQN_COMMON" ;;
    3)  ALGO="dqn"; ENV="gymnax/space_invaders"; SYSTEM="stoix/systems/q_learning/ff_dqn.py"
        EXTRA="$DQN_COMMON" ;;

    # ========================================
    # Rainbow on all MinAtar games
    # ========================================
    4)  ALGO="rainbow"; ENV="gymnax/breakout"; SYSTEM="stoix/systems/q_learning/ff_rainbow.py"
        EXTRA="$RAINBOW_COMMON" ;;
    5)  ALGO="rainbow"; ENV="gymnax/asterix"; SYSTEM="stoix/systems/q_learning/ff_rainbow.py"
        EXTRA="$RAINBOW_COMMON" ;;
    6)  ALGO="rainbow"; ENV="gymnax/freeway"; SYSTEM="stoix/systems/q_learning/ff_rainbow.py"
        EXTRA="$RAINBOW_COMMON" ;;
    7)  ALGO="rainbow"; ENV="gymnax/space_invaders"; SYSTEM="stoix/systems/q_learning/ff_rainbow.py"
        EXTRA="$RAINBOW_COMMON" ;;

    # ========================================
    # PPO on all MinAtar games
    # ========================================
    8)  ALGO="ppo"; ENV="gymnax/breakout"; SYSTEM="stoix/systems/ppo/anakin/ff_ppo.py"
        EXTRA="$PPO_COMMON" ;;
    9)  ALGO="ppo"; ENV="gymnax/asterix"; SYSTEM="stoix/systems/ppo/anakin/ff_ppo.py"
        EXTRA="$PPO_COMMON" ;;
    10) ALGO="ppo"; ENV="gymnax/freeway"; SYSTEM="stoix/systems/ppo/anakin/ff_ppo.py"
        EXTRA="$PPO_COMMON" ;;
    11) ALGO="ppo"; ENV="gymnax/space_invaders"; SYSTEM="stoix/systems/ppo/anakin/ff_ppo.py"
        EXTRA="$PPO_COMMON" ;;
esac

ENV_SHORT=$(echo $ENV | sed 's/\//_/g')

echo "============================================================"
echo "Task $TASK_ID: $ALGO on $ENV (5M steps)"
echo "============================================================"

uv run python $SYSTEM \
    env=$ENV \
    arch.seed=42 \
    arch.total_timesteps=$STEPS \
    logger.loggers.wandb.enabled=False \
    logger.loggers.json.enabled=True \
    logger.loggers.json.path=minatar_validation/${ALGO}_${ENV_SHORT} \
    $EXTRA

echo "============================================================"
echo "Task $TASK_ID ($ALGO on $ENV) completed"
echo "============================================================"
