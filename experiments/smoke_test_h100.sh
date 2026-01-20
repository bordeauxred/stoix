#!/bin/bash
# ============================================================================
# Quick H100 Smoke Test - Validate GPU is working
#
# Tests DQN and TD3 with both ortho modes
# Runtime: ~5-10 minutes total
# ============================================================================

set -e

# Enable verbose logging
export JAX_LOG_COMPILES=1
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "H100 Smoke Test - Quick Validation"
echo "============================================================"
echo ""

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Quick settings
SEEDS="[42,43]"
NUM_ENVS=64
STEPS=15000
NUM_EVAL=2
LAYERS="[256,256,256,256]"

PASSED=0
FAILED=0

# Test 1: DQN loss mode
echo "------------------------------------------------------------"
echo "[1/4] DQN + Loss-based ortho"
echo "------------------------------------------------------------"
if uv run python stoix/systems/q_learning/ff_dqn_ortho.py env=gymnax/breakout arch.seed=42 arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL "+multiseed=$SEEDS" "network.actor_network.pre_torso.layer_sizes=$LAYERS" network.actor_network.pre_torso.activation=groupsort system.ortho_mode=loss system.ortho_lambda=0.2 system.log_spectral_freq=1000000 logger.loggers.wandb.enabled=False; then
    echo "PASSED"
    ((PASSED++))
else
    echo "FAILED"
    ((FAILED++))
fi

# Test 2: DQN AdamO mode
echo ""
echo "------------------------------------------------------------"
echo "[2/4] DQN + AdamO"
echo "------------------------------------------------------------"
if uv run python stoix/systems/q_learning/ff_dqn_ortho.py env=gymnax/breakout arch.seed=42 arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL "+multiseed=$SEEDS" "network.actor_network.pre_torso.layer_sizes=$LAYERS" network.actor_network.pre_torso.activation=groupsort system.ortho_mode=optimizer system.ortho_coeff=0.001 system.log_spectral_freq=1000000 logger.loggers.wandb.enabled=False; then
    echo "PASSED"
    ((PASSED++))
else
    echo "FAILED"
    ((FAILED++))
fi

# Test 3: TD3 loss mode
echo ""
echo "------------------------------------------------------------"
echo "[3/4] TD3 + Loss-based ortho"
echo "------------------------------------------------------------"
if uv run python stoix/systems/ddpg/ff_td3.py env=brax/halfcheetah arch.seed=42 arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL "+multiseed=$SEEDS" "network.actor_network.pre_torso.layer_sizes=$LAYERS" network.actor_network.pre_torso.activation=groupsort "network.q_network.pre_torso.layer_sizes=$LAYERS" network.q_network.pre_torso.activation=groupsort +system.ortho_mode=loss +system.ortho_lambda=0.2 +system.ortho_exclude_output=true +system.log_spectral_freq=1000000 system.epochs=32 system.warmup_steps=200 system.actor_lr=3e-4 system.q_lr=3e-4 system.decay_learning_rates=false logger.loggers.wandb.enabled=False; then
    echo "PASSED"
    ((PASSED++))
else
    echo "FAILED"
    ((FAILED++))
fi

# Test 4: TD3 AdamO mode
echo ""
echo "------------------------------------------------------------"
echo "[4/4] TD3 + AdamO"
echo "------------------------------------------------------------"
if uv run python stoix/systems/ddpg/ff_td3.py env=brax/halfcheetah arch.seed=42 arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL "+multiseed=$SEEDS" "network.actor_network.pre_torso.layer_sizes=$LAYERS" network.actor_network.pre_torso.activation=groupsort "network.q_network.pre_torso.layer_sizes=$LAYERS" network.q_network.pre_torso.activation=groupsort +system.ortho_mode=optimizer +system.ortho_coeff=0.001 +system.ortho_exclude_output=true +system.log_spectral_freq=1000000 system.epochs=32 system.warmup_steps=200 system.actor_lr=3e-4 system.q_lr=3e-4 system.decay_learning_rates=false logger.loggers.wandb.enabled=False; then
    echo "PASSED"
    ((PASSED++))
else
    echo "FAILED"
    ((FAILED++))
fi

echo ""
echo "============================================================"
echo "Smoke Test Results: $PASSED/4 passed"
echo "============================================================"

if [ $FAILED -eq 0 ]; then
    echo "All tests passed! H100 is ready for overnight runs."
    exit 0
else
    echo "Some tests failed. Check errors above."
    exit 1
fi
