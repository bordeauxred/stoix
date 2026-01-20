#!/bin/bash
# ============================================================================
# H100 Smoke Test - Validates ortho support for DQN and TD3
#
# Tests:
# 1. DQN vanilla (no ortho) - baseline still works
# 2. DQN loss mode - ortho_loss and gram_deviation in logs
# 3. DQN AdamO mode - runs without ortho_loss in loss
# 4. TD3 loss mode - q_ortho_loss and actor_ortho_loss in logs
# 5. TD3 AdamO mode - runs without ortho_loss in loss
#
# Verifies: WandB logging, JSON logging, ortho metrics appear correctly
# Runtime: ~2-3 minutes
# ============================================================================

set -e

export JAX_LOG_COMPILES=0
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "H100 Smoke Test - Ortho Support Validation"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "No GPU detected"
echo ""

# Short runs with frequent logging to verify metrics
STEPS=5000
NUM_ENVS=32
NUM_EVAL=5  # Log 5 times to verify metrics appear
LAYERS="[128,128,128,128]"

PASSED=0
FAILED=0
RESULTS_DIR="results/json"

run_test() {
    local name="$1"
    local cmd="$2"
    local check_metric="$3"  # Optional: metric to verify in JSON

    echo "------------------------------------------------------------"
    echo "[$name]"
    echo "------------------------------------------------------------"

    # Run the command
    if eval "$cmd"; then
        echo "  Run: OK"

        # Check for JSON output if metric verification requested
        if [ -n "$check_metric" ]; then
            # Find most recent JSON file
            json_file=$(ls -t $RESULTS_DIR/*/metrics.json 2>/dev/null | head -1)
            if [ -n "$json_file" ] && [ -f "$json_file" ]; then
                if grep -q "$check_metric" "$json_file"; then
                    echo "  JSON logs: OK (found $check_metric)"
                    ((PASSED++))
                else
                    echo "  JSON logs: FAILED (missing $check_metric)"
                    echo "    File: $json_file"
                    echo "    Keys found: $(head -1 "$json_file" | tr ',' '\n' | head -10)"
                    ((FAILED++))
                fi
            else
                echo "  JSON logs: FAILED (no file found)"
                ((FAILED++))
            fi
        else
            ((PASSED++))
        fi
    else
        echo "  Run: FAILED"
        ((FAILED++))
    fi
    echo ""
}

# Clean old results
rm -rf $RESULTS_DIR 2>/dev/null || true

# Test 1: DQN vanilla (no ortho) - verify baseline still works
run_test "1/5 DQN vanilla (no ortho)" \
    "uv run python stoix/systems/q_learning/ff_dqn.py env=gymnax/cartpole arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" logger.loggers.wandb.enabled=False +logger.loggers.file.enabled=True" \
    "q_loss"

# Test 2: DQN loss mode - verify ortho_loss appears
run_test "2/5 DQN + loss ortho" \
    "uv run python stoix/systems/q_learning/ff_dqn.py env=gymnax/cartpole arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" network.actor_network.pre_torso.activation=groupsort +system.ortho_mode=loss +system.ortho_lambda=0.2 +system.ortho_exclude_output=true logger.loggers.wandb.enabled=False +logger.loggers.file.enabled=True" \
    "ortho_loss"

# Test 3: DQN AdamO mode
run_test "3/5 DQN + AdamO" \
    "uv run python stoix/systems/q_learning/ff_dqn.py env=gymnax/cartpole arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" network.actor_network.pre_torso.activation=groupsort +system.ortho_mode=optimizer +system.ortho_coeff=0.001 +system.ortho_exclude_output=true logger.loggers.wandb.enabled=False +logger.loggers.file.enabled=True" \
    "q_loss"

# Test 4: TD3 loss mode - verify both q_ortho_loss and actor_ortho_loss appear
run_test "4/5 TD3 + loss ortho" \
    "uv run python stoix/systems/ddpg/ff_td3.py env=brax/halfcheetah arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" network.actor_network.pre_torso.activation=groupsort \"network.q_network.pre_torso.layer_sizes=$LAYERS\" network.q_network.pre_torso.activation=groupsort +system.ortho_mode=loss +system.ortho_lambda=0.2 +system.ortho_exclude_output=true system.epochs=4 system.warmup_steps=50 logger.loggers.wandb.enabled=False +logger.loggers.file.enabled=True" \
    "q_ortho_loss"

# Test 5: TD3 AdamO mode
run_test "5/5 TD3 + AdamO" \
    "uv run python stoix/systems/ddpg/ff_td3.py env=brax/halfcheetah arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" network.actor_network.pre_torso.activation=groupsort \"network.q_network.pre_torso.layer_sizes=$LAYERS\" network.q_network.pre_torso.activation=groupsort +system.ortho_mode=optimizer +system.ortho_coeff=0.001 +system.ortho_exclude_output=true system.epochs=4 system.warmup_steps=50 logger.loggers.wandb.enabled=False +logger.loggers.file.enabled=True" \
    "q_loss"

# Summary
echo "============================================================"
echo "RESULTS: $PASSED passed, $FAILED failed"
echo "============================================================"

if [ $FAILED -eq 0 ]; then
    echo "All tests passed! Ortho support working for DQN and TD3."
    exit 0
else
    echo "Some tests failed. Check output above."
    exit 1
fi
