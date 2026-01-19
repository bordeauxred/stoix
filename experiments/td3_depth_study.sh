#!/bin/bash
#SBATCH --job-name=td3_depth
#SBATCH --partition=kisski
#SBATCH --array=0-29%6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/td3_depth_%A_%a.out
#SBATCH --error=logs/td3_depth_%A_%a.err
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
# TD3 Network Depth Study - Baseline for Isometric Network Comparison
#
# Study design:
#   - 6 environments: 4 dense + 2 sparse-ish reward
#   - 5 network depths: 2, 4, 8, 16, 32 layers
#   - 5 seeds per configuration (vmapped parallel training)
#   - Total: 6 × 5 = 30 SLURM tasks
#
# Environments:
#   Dense reward:
#     - brax/halfcheetah (MJX) - standard locomotion benchmark
#     - brax/ant (MJX) - harder locomotion
#     - brax/hopper (MJX) - locomotion
#     - gymnax/pendulum - simple continuous control
#   Sparse-ish reward:
#     - brax/reacher - goal reaching (sparse at target)
#     - brax/pusher - object manipulation (sparse at goal)
#
# Naming convention: td3_depth{D}_{env}_{timestamp}
# WandB project: stoix_td3_depth_study
# ============================================================================

SEEDS="[42,43,44,45,46]"
NUM_ENVS=64
SYSTEM="stoix/systems/ddpg/ff_td3.py"

# Environment index (0-5)
ENV_IDX=$((TASK_ID / 5))
# Depth index (0-4)
DEPTH_IDX=$((TASK_ID % 5))

# Environments with appropriate step counts
# 10M steps for Brax locomotion, 4M for simpler envs
case $ENV_IDX in
    0) ENV="brax/halfcheetah"; ENV_SHORT="halfcheetah"; STEPS=10000000; BACKEND="env.kwargs.backend=mjx" ;;
    1) ENV="brax/ant"; ENV_SHORT="ant"; STEPS=10000000; BACKEND="env.kwargs.backend=mjx" ;;
    2) ENV="brax/hopper"; ENV_SHORT="hopper"; STEPS=10000000; BACKEND="env.kwargs.backend=mjx" ;;
    3) ENV="gymnax/pendulum"; ENV_SHORT="pendulum"; STEPS=4000000; BACKEND="" ;;
    4) ENV="brax/reacher"; ENV_SHORT="reacher_sparse"; STEPS=4000000; BACKEND="env.kwargs.backend=mjx" ;;
    5) ENV="brax/pusher"; ENV_SHORT="pusher_sparse"; STEPS=4000000; BACKEND="env.kwargs.backend=mjx" ;;
esac

# Network depths (number of hidden layers)
case $DEPTH_IDX in
    0) DEPTH=2; ACTOR_LAYERS="[256,256]"; Q_LAYERS="[256,256]" ;;
    1) DEPTH=4; ACTOR_LAYERS="[256,256,256,256]"; Q_LAYERS="[256,256,256,256]" ;;
    2) DEPTH=8; ACTOR_LAYERS="[256,256,256,256,256,256,256,256]"; Q_LAYERS="[256,256,256,256,256,256,256,256]" ;;
    3) DEPTH=16; ACTOR_LAYERS="[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]"; Q_LAYERS="[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]" ;;
    4) DEPTH=32; ACTOR_LAYERS="[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]"; Q_LAYERS="[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]" ;;
esac

# Hyperparameters
# UTD=32 based on UTD ratio study: best for ant, halfcheetah, pendulum (3/4 envs)
# Hopper preferred UTD=20 but difference was marginal (2017 vs 1975)
UTD=32
ACTIVATION="relu"
SEED_START=42
SEED_END=46

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Naming: {algo}_{env}_d{depth}_{activation}_utd{utd}_s{seeds}_{timestamp}
RUN_NAME="td3_${ENV_SHORT}_d${DEPTH}_${ACTIVATION}_utd${UTD}_s${SEED_START}-${SEED_END}_${TIMESTAMP}"

echo "============================================================"
echo "TD3 Depth Study - Baseline Experiment"
echo "============================================================"
echo "Task ID:     $TASK_ID"
echo "Environment: $ENV ($ENV_SHORT)"
echo "Depth:       $DEPTH layers"
echo "Activation:  $ACTIVATION"
echo "UTD:         $UTD"
echo "Seeds:       $SEEDS"
echo "Steps:       $STEPS"
echo "Run name:    $RUN_NAME"
echo "============================================================"

# Common settings
COMMON="arch.total_num_envs=$NUM_ENVS"
UTD_FIX="system.epochs=$UTD system.warmup_steps=200"
HP_FIX="system.actor_lr=3e-4 system.q_lr=3e-4 system.decay_learning_rates=false"

# Network architecture override (both actor and critic)
NETWORK="network.actor_network.pre_torso.layer_sizes=$ACTOR_LAYERS network.actor_network.pre_torso.activation=$ACTIVATION network.q_network.pre_torso.layer_sizes=$Q_LAYERS network.q_network.pre_torso.activation=$ACTIVATION"

# Note: Run naming is handled automatically by the code
# Each seed gets its own WandB run with shared group for aggregation
# JSON logs are saved automatically to results/json/{group}/seed_{seed}/metrics.json
uv run python $SYSTEM \
    env=$ENV \
    arch.seed=42 \
    arch.total_timesteps=$STEPS \
    +multiseed=$SEEDS \
    $COMMON \
    $UTD_FIX \
    $HP_FIX \
    $NETWORK \
    $BACKEND \
    logger.loggers.wandb.enabled=True \
    logger.loggers.wandb.project=stoix_td3_depth_study \
    "logger.loggers.wandb.tag=[depth_study,utd32,relu,baseline]"

echo "============================================================"
echo "Task $TASK_ID completed: depth=$DEPTH on $ENV_SHORT"
echo "============================================================"
