#!/bin/bash
# ============================================================================
# PPO Ortho Depth Study - High Throughput Config
#
# Tests PPO with orthonormalization at various depths using Brax-style
# parallelization (2048 envs) for fast iteration.
#
# Modes:
#   - baseline: PPO vanilla (no ortho)
#   - loss: PPO + GroupSort + ortho loss mode
#   - optimizer: PPO + GroupSort + ortho AdamO mode
#
# Depths: 4, 8, 16, 32 layers
# Envs: gymnax/cartpole, gymnax/breakout
# Seeds: Run sequentially (2 seeds for memory, can increase if fits)
#
# Usage:
#   Single GPU:  bash experiments/ppo_ortho_depth_study.sh
#   With range:  bash experiments/ppo_ortho_depth_study.sh 0 23  # configs 0-23
# ============================================================================

export JAX_LOG_COMPILES=0
export PYTHONUNBUFFERED=1

START_IDX=${1:-0}
END_IDX=${2:-47}  # 2 envs x 4 depths x 3 modes x 2 seeds = 48 configs
FAILED=0
PASSED=0

echo "============================================================"
echo "PPO Ortho Depth Study - High Throughput"
echo "============================================================"
echo "Running configs $START_IDX to $END_IDX (of 48 total)"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "No GPU detected"
echo ""

# High-throughput PPO settings (Brax-style)
NUM_ENVS=2048         # High parallelism
STEPS=10000000        # 10M steps
NUM_EVAL=100          # Log 100 times
ROLLOUT=128           # Standard PPO rollout
EPOCHS=4              # Standard PPO epochs
MINIBATCHES=16        # Standard PPO minibatches

# Seeds run sequentially
SEEDS=(42 43)

# Depths
DEPTHS=(4 8 16 32)

# Modes
MODES=("baseline" "loss" "optimizer")

# Environments
ENVS=("gymnax/cartpole" "gymnax/breakout")
ENV_SHORTS=("cartpole" "breakout")

get_layers() {
    local depth=$1
    local layers="["
    for ((i=1; i<=depth; i++)); do
        if [ $i -eq $depth ]; then
            layers="${layers}256"
        else
            layers="${layers}256,"
        fi
    done
    echo "${layers}]"
}

# Config layout: 2 envs x 4 depths x 3 modes x 2 seeds = 48 configs
# Index order: seed -> mode -> depth -> env
for TASK_ID in $(seq $START_IDX $END_IDX); do
    SEED_IDX=$((TASK_ID / 24))      # 24 = 2 envs x 4 depths x 3 modes
    REMAINING=$((TASK_ID % 24))
    MODE_IDX=$((REMAINING / 8))      # 8 = 2 envs x 4 depths
    REMAINING=$((REMAINING % 8))
    DEPTH_IDX=$((REMAINING / 2))     # 2 = 2 envs
    ENV_IDX=$((REMAINING % 2))

    SEED=${SEEDS[$SEED_IDX]}
    MODE=${MODES[$MODE_IDX]}
    DEPTH=${DEPTHS[$DEPTH_IDX]}
    LAYERS=$(get_layers $DEPTH)
    ENV=${ENVS[$ENV_IDX]}
    ENV_SHORT=${ENV_SHORTS[$ENV_IDX]}

    echo ""
    echo "============================================================"
    echo "[$TASK_ID/47] PPO Depth Study"
    echo "============================================================"
    echo "Seed:       $SEED"
    echo "Env:        $ENV"
    echo "Depth:      $DEPTH layers"
    echo "Mode:       $MODE"
    echo ""

    # Build command based on mode
    CMD="uv run python stoix/systems/ppo/anakin/ff_ppo.py"
    CMD="$CMD env=$ENV"
    CMD="$CMD arch.seed=$SEED"
    CMD="$CMD arch.total_timesteps=$STEPS"
    CMD="$CMD arch.total_num_envs=$NUM_ENVS"
    CMD="$CMD arch.num_evaluation=$NUM_EVAL"
    CMD="$CMD system.rollout_length=$ROLLOUT"
    CMD="$CMD system.epochs=$EPOCHS"
    CMD="$CMD system.num_minibatches=$MINIBATCHES"
    CMD="$CMD \"network.actor_network.pre_torso.layer_sizes=$LAYERS\""
    CMD="$CMD \"network.critic_network.pre_torso.layer_sizes=$LAYERS\""

    if [ "$MODE" == "baseline" ]; then
        # Vanilla PPO (no ortho, no groupsort)
        TAG="baseline"
    elif [ "$MODE" == "loss" ]; then
        # PPO + GroupSort + loss mode ortho
        CMD="$CMD network.actor_network.pre_torso.activation=groupsort"
        CMD="$CMD network.critic_network.pre_torso.activation=groupsort"
        CMD="$CMD +system.ortho_mode=loss"
        CMD="$CMD +system.ortho_lambda=0.2"
        CMD="$CMD +system.ortho_exclude_output=true"
        TAG="loss"
    else
        # PPO + GroupSort + optimizer (AdamO) mode ortho
        CMD="$CMD network.actor_network.pre_torso.activation=groupsort"
        CMD="$CMD network.critic_network.pre_torso.activation=groupsort"
        CMD="$CMD +system.ortho_mode=optimizer"
        CMD="$CMD +system.ortho_coeff=0.001"
        CMD="$CMD +system.ortho_exclude_output=true"
        TAG="adamo"
    fi

    # Logging
    CMD="$CMD logger.loggers.wandb.enabled=True"
    CMD="$CMD logger.loggers.wandb.project=stoix_ppo_depth_study"
    CMD="$CMD \"logger.loggers.wandb.tag=[ppo,${TAG},depth_${DEPTH},${ENV_SHORT},seed_${SEED}]\""
    CMD="$CMD logger.loggers.json.enabled=True"

    echo "Running: $CMD"
    echo ""

    if eval "$CMD"; then
        ((PASSED++))
        echo "[$TASK_ID] PASSED: PPO $TAG depth=$DEPTH env=$ENV_SHORT seed=$SEED"
    else
        ((FAILED++))
        echo "[$TASK_ID] FAILED: PPO $TAG depth=$DEPTH env=$ENV_SHORT seed=$SEED"
    fi

    echo ""
done

echo ""
echo "============================================================"
echo "PPO Ortho Depth Study completed!"
echo "RESULTS: $PASSED passed, $FAILED failed out of $((END_IDX - START_IDX + 1)) configs"
echo "============================================================"
echo ""
echo "Check WandB project: stoix_ppo_depth_study"
