#!/bin/bash
#SBATCH --job-name=td3_utd
#SBATCH --partition=kisski
#SBATCH --array=0-41%12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/td3_utd_%A_%a.out
#SBATCH --error=logs/td3_utd_%A_%a.err
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
# TD3 UTD (Update-to-Data) Ratio Study
#
# Similar to GCJaxRL's UTD study - find optimal UTD per environment
# before doing depth study with fixed UTD.
#
# Study design:
#   - 6 environments: 4 dense + 2 sparse-ish reward (same as depth study)
#   - 7 UTD ratios: 1, 2, 4, 8, 16, 20, 32
#   - 5 seeds per configuration (vmapped parallel training)
#   - Total: 6 × 7 = 42 SLURM tasks
#
# UTD ratio = epochs (number of gradient updates per env step)
# In Stoix: system.epochs controls this directly
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
# Naming convention: td3_utd{U}_{env}_{timestamp}
# WandB project: stoix_td3_utd_study
# ============================================================================

SEEDS="[42,43,44,45,46]"
NUM_ENVS=64
SYSTEM="stoix/systems/ddpg/ff_td3.py"

# Environment index (0-5)
ENV_IDX=$((TASK_ID / 7))
# UTD index (0-6)
UTD_IDX=$((TASK_ID % 7))

# Environments with appropriate step counts
# Using 1M steps as UTD study is about finding the ratio, not final performance
case $ENV_IDX in
    0) ENV="brax/halfcheetah"; ENV_SHORT="halfcheetah"; STEPS=1000000; BACKEND="env.kwargs.backend=mjx" ;;
    1) ENV="brax/ant"; ENV_SHORT="ant"; STEPS=1000000; BACKEND="env.kwargs.backend=mjx" ;;
    2) ENV="brax/hopper"; ENV_SHORT="hopper"; STEPS=1000000; BACKEND="env.kwargs.backend=mjx" ;;
    3) ENV="gymnax/pendulum"; ENV_SHORT="pendulum"; STEPS=500000; BACKEND="" ;;
    4) ENV="brax/reacher"; ENV_SHORT="reacher"; STEPS=500000; BACKEND="env.kwargs.backend=mjx" ;;
    5) ENV="brax/pusher"; ENV_SHORT="pusher"; STEPS=500000; BACKEND="env.kwargs.backend=mjx" ;;
esac

# UTD ratios (epochs parameter)
# Common values from literature: 1 (original TD3), 20 (REDQ), 16-32 (modern off-policy)
case $UTD_IDX in
    0) UTD=1 ;;
    1) UTD=2 ;;
    2) UTD=4 ;;
    3) UTD=8 ;;
    4) UTD=16 ;;
    5) UTD=20 ;;
    6) UTD=32 ;;
esac

# Fixed hyperparameters for UTD study
DEPTH=2
ACTIVATION="silu"
SEED_START=42
SEED_END=46

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Naming: {algo}_{env}_d{depth}_{activation}_utd{utd}_s{seeds}_{timestamp}
RUN_NAME="td3_${ENV_SHORT}_d${DEPTH}_${ACTIVATION}_utd${UTD}_s${SEED_START}-${SEED_END}_${TIMESTAMP}"

echo "============================================================"
echo "TD3 UTD Ratio Study"
echo "============================================================"
echo "Task ID:     $TASK_ID"
echo "Environment: $ENV ($ENV_SHORT)"
echo "Depth:       $DEPTH layers"
echo "Activation:  $ACTIVATION"
echo "UTD ratio:   $UTD (epochs)"
echo "Seeds:       $SEEDS"
echo "Steps:       $STEPS"
echo "Run name:    $RUN_NAME"
echo "============================================================"

# Common settings
COMMON="arch.total_num_envs=$NUM_ENVS"

# Warmup steps scale with UTD to ensure buffer has enough samples
# Higher UTD needs more warmup to fill buffer adequately
WARMUP=$((100 + UTD * 10))

# Standard hyperparameters (from TD3 paper)
HP_FIX="system.actor_lr=3e-4 system.q_lr=3e-4 system.decay_learning_rates=false"

# Standard 2-layer network (baseline architecture)
NETWORK="network.actor_network.pre_torso.layer_sizes=[256,256] network.q_network.pre_torso.layer_sizes=[256,256]"

# Note: Run naming is handled automatically by the code
# Each seed gets its own WandB run with shared group for aggregation
# JSON logs are saved automatically to results/json/{group}/seed_{seed}/metrics.json
uv run python $SYSTEM \
    env=$ENV \
    arch.seed=42 \
    arch.total_timesteps=$STEPS \
    +multiseed=$SEEDS \
    $COMMON \
    system.epochs=$UTD \
    system.warmup_steps=$WARMUP \
    $HP_FIX \
    $NETWORK \
    $BACKEND \
    logger.loggers.wandb.enabled=True \
    logger.loggers.wandb.project=stoix_td3_utd_study \
    "logger.loggers.wandb.tag=[utd_study]"

echo "============================================================"
echo "Task $TASK_ID completed: UTD=$UTD on $ENV_SHORT"
echo "============================================================"
