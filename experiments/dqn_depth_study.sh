#!/bin/bash
#SBATCH --job-name=dqn_depth
#SBATCH --partition=kisski
#SBATCH --array=0-29%15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/dqn_depth_%A_%a.out
#SBATCH --error=logs/dqn_depth_%A_%a.err
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
# DQN Network Depth Study - Baseline for Isometric Network Comparison
#
# Study design:
#   - 6 environments: 4 MinAtar + Pendulum + MountainCar
#   - 5 network depths: 2, 4, 8, 16, 32 layers
#   - 5 seeds per configuration (vmapped parallel training)
#   - Total: 6 × 5 = 30 SLURM tasks
#
# Naming convention: dqn_depth{D}_{env}_{timestamp}
#   where D = number of hidden layers
#
# WandB project: stoix_dqn_depth_study
# ============================================================================

# Literature-based step counts:
#   MinAtar (Young & Tian 2019): 5M frames, many papers use 10M
#   Simple control: 1-2M sufficient
#
# We use 10M for MinAtar (thorough baseline), 2M for simple control
SEEDS="[42,43,44,45,46]"
NUM_ENVS=64
SYSTEM="stoix/systems/q_learning/ff_dqn.py"

# Environment index (0-5)
ENV_IDX=$((TASK_ID / 5))
# Depth index (0-4)
DEPTH_IDX=$((TASK_ID % 5))

# Environments with appropriate step counts
case $ENV_IDX in
    0) ENV="gymnax/breakout"; ENV_SHORT="minatar_breakout"; STEPS=10000000 ;;
    1) ENV="gymnax/asterix"; ENV_SHORT="minatar_asterix"; STEPS=10000000 ;;
    2) ENV="gymnax/freeway"; ENV_SHORT="minatar_freeway"; STEPS=10000000 ;;
    3) ENV="gymnax/space_invaders"; ENV_SHORT="minatar_spaceinv"; STEPS=10000000 ;;
    4) ENV="gymnax/pendulum"; ENV_SHORT="pendulum"; STEPS=2000000 ;;
    5) ENV="gymnax/mountain_car"; ENV_SHORT="mountaincar"; STEPS=2000000 ;;
esac

# Network depths (number of hidden layers)
case $DEPTH_IDX in
    0) DEPTH=2; LAYERS="[256,256]" ;;
    1) DEPTH=4; LAYERS="[256,256,256,256]" ;;
    2) DEPTH=8; LAYERS="[256,256,256,256,256,256,256,256]" ;;
    3) DEPTH=16; LAYERS="[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]" ;;
    4) DEPTH=32; LAYERS="[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]" ;;
esac

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="dqn_depth${DEPTH}_${ENV_SHORT}_${TIMESTAMP}"

echo "============================================================"
echo "DQN Depth Study - Baseline Experiment"
echo "============================================================"
echo "Task ID:     $TASK_ID"
echo "Environment: $ENV ($ENV_SHORT)"
echo "Depth:       $DEPTH layers"
echo "Seeds:       $SEEDS"
echo "Run name:    $RUN_NAME"
echo "============================================================"

# Common settings
COMMON="arch.total_num_envs=$NUM_ENVS"

# Network architecture override
NETWORK="network.actor_network.pre_torso.layer_sizes=$LAYERS"

# Note: Run naming is handled automatically by the code
# Each seed gets its own WandB run with shared group for aggregation
# JSON logs are saved automatically to results/json/{group}/seed_{seed}/metrics.json
uv run python $SYSTEM \
    env=$ENV \
    arch.seed=42 \
    arch.total_timesteps=$STEPS \
    +multiseed=$SEEDS \
    $COMMON \
    $NETWORK \
    logger.loggers.wandb.enabled=True \
    logger.loggers.wandb.project=stoix_dqn_depth_study \
    "logger.loggers.wandb.tag=[depth_study,baseline]"

echo "============================================================"
echo "Task $TASK_ID completed: depth=$DEPTH on $ENV_SHORT"
echo "============================================================"
