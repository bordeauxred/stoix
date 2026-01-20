#!/bin/bash
# ============================================================================
# Overnight Depth Study - H100
#
# Runs DQN and TD3 depth scaling experiments with both ortho modes
# Using ortho_coeff=1e-3 for AdamO
#
# Design:
#   - DQN: 4 depths x 2 modes x 2 envs (breakout, asterix) = 16 configs
#   - TD3: 4 depths x 2 modes x 2 envs (halfcheetah, ant) = 16 configs
#   - Total: 32 configs
#
# To run a subset, use: bash overnight_depth_study_h100.sh [start] [end]
# ============================================================================

set -e

# Enable verbose logging
export JAX_LOG_COMPILES=1
export PYTHONUNBUFFERED=1

START_IDX=${1:-0}
END_IDX=${2:-31}

echo "============================================================"
echo "Overnight Depth Study - H100"
echo "============================================================"
echo "Running configs $START_IDX to $END_IDX"
echo ""

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Common settings
SEEDS="[42,43,44,45,46]"
NUM_ENVS=64
STEPS=10000000
NUM_EVAL=100

# Depths to test (up to 32 layers)
DEPTHS=(4 8 16 32)

# Environment arrays
DQN_ENVS=("gymnax/breakout" "gymnax/asterix")
DQN_ENV_SHORTS=("breakout" "asterix")
TD3_ENVS=("brax/halfcheetah" "brax/ant")
TD3_ENV_SHORTS=("halfcheetah" "ant")

# Generate layer string for a given depth
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

# Config mapping:
# 0-15:  DQN (4 depths x 2 modes x 2 envs)
# 16-31: TD3 (4 depths x 2 modes x 2 envs)
#
# Within each algo block of 16:
#   env_idx = config / 8
#   depth_idx = (config % 8) / 2
#   mode_idx = config % 2

for TASK_ID in $(seq $START_IDX $END_IDX); do
    ALGO_IDX=$((TASK_ID / 16))
    CONFIG_IDX=$((TASK_ID % 16))
    ENV_IDX=$((CONFIG_IDX / 8))
    REMAINING=$((CONFIG_IDX % 8))
    DEPTH_IDX=$((REMAINING / 2))
    MODE_IDX=$((REMAINING % 2))

    DEPTH=${DEPTHS[$DEPTH_IDX]}
    LAYERS=$(get_layers $DEPTH)

    if [ $MODE_IDX -eq 0 ]; then
        MODE="loss"
        TAG="loss"
    else
        MODE="optimizer"
        TAG="adamo"
    fi

    echo ""
    echo "============================================================"
    echo "[$TASK_ID/$END_IDX] Starting..."
    echo "============================================================"

    if [ $ALGO_IDX -eq 0 ]; then
        # DQN
        ALGO="DQN"
        ENV=${DQN_ENVS[$ENV_IDX]}
        ENV_SHORT=${DQN_ENV_SHORTS[$ENV_IDX]}

        echo "Algorithm:  $ALGO"
        echo "Env:        $ENV"
        echo "Depth:      $DEPTH layers"
        echo "Mode:       $MODE ($TAG)"
        echo ""

        if [ "$MODE" == "loss" ]; then
            COEFF_ARG="system.ortho_lambda=0.2"
        else
            COEFF_ARG="system.ortho_coeff=0.001"
        fi

        uv run python stoix/systems/q_learning/ff_dqn_ortho.py env=$ENV arch.seed=42 arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL "+multiseed=$SEEDS" "network.actor_network.pre_torso.layer_sizes=$LAYERS" network.actor_network.pre_torso.activation=groupsort system.ortho_mode=$MODE $COEFF_ARG system.log_spectral_freq=1000000 logger.loggers.wandb.enabled=True logger.loggers.wandb.project=stoix_depth_study_h100 "logger.loggers.wandb.tag=[depth_study,dqn,${TAG},depth_${DEPTH},${ENV_SHORT}]" +logger.loggers.file.enabled=True

    else
        # TD3
        ALGO="TD3"
        ENV=${TD3_ENVS[$ENV_IDX]}
        ENV_SHORT=${TD3_ENV_SHORTS[$ENV_IDX]}

        echo "Algorithm:  $ALGO"
        echo "Env:        $ENV"
        echo "Depth:      $DEPTH layers"
        echo "Mode:       $MODE ($TAG)"
        echo ""

        if [ "$MODE" == "loss" ]; then
            COEFF_ARG="+system.ortho_lambda=0.2"
        else
            COEFF_ARG="+system.ortho_coeff=0.001"
        fi

        uv run python stoix/systems/ddpg/ff_td3.py env=$ENV arch.seed=42 arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL "+multiseed=$SEEDS" "network.actor_network.pre_torso.layer_sizes=$LAYERS" network.actor_network.pre_torso.activation=groupsort "network.q_network.pre_torso.layer_sizes=$LAYERS" network.q_network.pre_torso.activation=groupsort +system.ortho_mode=$MODE $COEFF_ARG +system.ortho_exclude_output=true +system.log_spectral_freq=1000000 system.epochs=32 system.warmup_steps=200 system.actor_lr=3e-4 system.q_lr=3e-4 system.decay_learning_rates=false logger.loggers.wandb.enabled=True logger.loggers.wandb.project=stoix_depth_study_h100 "logger.loggers.wandb.tag=[depth_study,td3,${TAG},depth_${DEPTH},${ENV_SHORT}]" +logger.loggers.file.enabled=True
    fi

    echo ""
    echo "[$TASK_ID] Completed: $ALGO $TAG depth=$DEPTH env=$ENV_SHORT"
done

echo ""
echo "============================================================"
echo "Overnight depth study completed!"
echo "============================================================"
