#!/bin/bash
# ============================================================================
# Smoke Test: DQN AdamO vs Loss-based Ortho
#
# Quick validation of infrastructure before full experiments.
# Tests 2 depths × 2 modes × 1 env × 3 seeds = 4 configurations
#
# Expected runtime: ~10-15 minutes total (sequential)
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}DQN AdamO Smoke Test${NC}"
echo -e "${YELLOW}============================================================${NC}"

# Configuration
SYSTEM="stoix/systems/q_learning/ff_dqn.py"
ENV="gymnax/breakout"
SEEDS="[42,43,44]"  # 3 seeds for quick test
NUM_ENVS=32
STEPS=100000  # 100k steps for smoke test (fast)
NUM_EVAL=10   # Log every ~10k steps

# Test configurations
DEPTHS=(4 8)
MODES=("loss" "optimizer")

# Hyperparameters
ORTHO_LAMBDA=0.2
ORTHO_COEFF=1e-3

echo ""
echo "Configuration:"
echo "  Environment: $ENV"
echo "  Seeds: $SEEDS"
echo "  Steps: $STEPS"
echo "  Depths: ${DEPTHS[*]}"
echo "  Modes: ${MODES[*]}"
echo "  ortho_lambda (loss mode): $ORTHO_LAMBDA"
echo "  ortho_coeff (optimizer mode): $ORTHO_COEFF"
echo ""

PASSED=0
FAILED=0

for DEPTH in "${DEPTHS[@]}"; do
    # Generate layer sizes
    LAYERS="["
    for ((i=1; i<=DEPTH; i++)); do
        if [ $i -eq $DEPTH ]; then
            LAYERS="${LAYERS}256"
        else
            LAYERS="${LAYERS}256,"
        fi
    done
    LAYERS="${LAYERS}]"

    for MODE in "${MODES[@]}"; do
        echo -e "${YELLOW}------------------------------------------------------------${NC}"
        echo -e "${YELLOW}Testing: depth=$DEPTH, mode=$MODE${NC}"
        echo -e "${YELLOW}------------------------------------------------------------${NC}"

        # Build command
        CMD="uv run python $SYSTEM \
            env=$ENV \
            arch.seed=42 \
            arch.total_timesteps=$STEPS \
            arch.total_num_envs=$NUM_ENVS \
            arch.num_evaluation=$NUM_EVAL \
            +multiseed=$SEEDS \
            network.actor_network.pre_torso.layer_sizes=\"$LAYERS\" \
            network.actor_network.pre_torso.activation=groupsort \
            system.ortho_mode=$MODE"

        # Add mode-specific hyperparameters
        if [ "$MODE" == "loss" ]; then
            CMD="$CMD system.ortho_lambda=$ORTHO_LAMBDA"
        else
            CMD="$CMD system.ortho_coeff=$ORTHO_COEFF"
        fi

        # Enable logging (disable expensive SVD spectral diagnostics)
        CMD="$CMD system.log_spectral_freq=1000000"
        CMD="$CMD logger.loggers.wandb.enabled=True"
        CMD="$CMD logger.loggers.wandb.project=stoix_adamo_smoke_test"
        CMD="$CMD \"logger.loggers.wandb.tag=[smoke_test,dqn,depth_${DEPTH},${MODE}]\""
        CMD="$CMD +logger.loggers.file.enabled=True"

        echo "Command: $CMD"
        echo ""

        # Run and capture result
        START_TIME=$(date +%s)
        if eval $CMD; then
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            echo -e "${GREEN}✓ PASSED: depth=$DEPTH, mode=$MODE (${DURATION}s)${NC}"
            ((PASSED++))
        else
            echo -e "${RED}✗ FAILED: depth=$DEPTH, mode=$MODE${NC}"
            ((FAILED++))
        fi
        echo ""
    done
done

echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Smoke Test Results${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed! Infrastructure is ready.${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Check logs above.${NC}"
    exit 1
fi
