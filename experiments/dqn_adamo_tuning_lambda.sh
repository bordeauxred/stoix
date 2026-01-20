#!/bin/bash
# ============================================================================
# DQN AdamO Tuning - Lambda Labs / Raw GPU Version
#
# Run with: bash experiments/dqn_adamo_tuning_lambda.sh [start_idx] [end_idx]
# Example:  bash experiments/dqn_adamo_tuning_lambda.sh 0 15   # All 16 configs
#           bash experiments/dqn_adamo_tuning_lambda.sh 0 3    # First 4 (breakout)
#
# For parallel on multiple GPUs, run in separate terminals:
#   Terminal 1: CUDA_VISIBLE_DEVICES=0 bash ... 0 7
#   Terminal 2: CUDA_VISIBLE_DEVICES=1 bash ... 8 15
# ============================================================================

set -e

START_IDX=${1:-0}
END_IDX=${2:-15}

echo "Running tasks $START_IDX to $END_IDX"

# Configuration
SEEDS="[42,43,44,45,46]"
NUM_ENVS=64
STEPS=10000000
NUM_EVAL=100
SYSTEM="stoix/systems/q_learning/ff_dqn.py"
LAYERS="[256,256,256,256]"

# Environment arrays
ENVS=("gymnax/breakout" "gymnax/asterix" "gymnax/freeway" "gymnax/space_invaders")
ENV_SHORTS=("breakout" "asterix" "freeway" "spaceinv")

for TASK_ID in $(seq $START_IDX $END_IDX); do
    ENV_IDX=$((TASK_ID / 4))
    CONFIG_IDX=$((TASK_ID % 4))

    ENV=${ENVS[$ENV_IDX]}
    ENV_SHORT=${ENV_SHORTS[$ENV_IDX]}

    case $CONFIG_IDX in
        0) MODE="loss"; COEFF_ARG="system.ortho_lambda=0.2"; TAG="loss" ;;
        1) MODE="optimizer"; COEFF_ARG="system.ortho_coeff=0.0001"; TAG="adamo_1e-4" ;;
        2) MODE="optimizer"; COEFF_ARG="system.ortho_coeff=0.001"; TAG="adamo_1e-3" ;;
        3) MODE="optimizer"; COEFF_ARG="system.ortho_coeff=0.01"; TAG="adamo_1e-2" ;;
    esac

    echo ""
    echo "============================================================"
    echo "[$TASK_ID/15] $TAG on $ENV_SHORT"
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

    echo "[$TASK_ID/15] Completed: $TAG on $ENV_SHORT"
done

echo ""
echo "============================================================"
echo "All tasks completed!"
echo "============================================================"
