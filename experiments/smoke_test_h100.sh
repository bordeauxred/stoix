#!/bin/bash
# ============================================================================
# H100 Smoke Test - Validates ortho support for DQN, TD3, and PPO
#
# Tests (AdamO first to verify the fix):
# 1. DQN AdamO mode - decoupled ortho
# 2. TD3 AdamO mode - decoupled ortho
# 3. DQN loss mode - ortho_loss in logs
# 4. TD3 loss mode - q_ortho_loss in logs
# 5. DQN vanilla - baseline still works
# 6. PPO AdamO mode - decoupled ortho
# 7. PPO loss mode - ortho_loss in logs
# 8. PPO vanilla - baseline still works
#
# Verifies: WandB logging, JSON logging, ortho metrics appear correctly
# Runtime: ~5 minutes
# ============================================================================

# Don't use set -e - we want all tests to run even if some fail

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
# JSON path: results/json/{system_name}/{unique_token}/*.json
RESULTS_DIR="results/json"

run_test() {
    local name="$1"
    local cmd="$2"
    local check_metric="$3"  # metric to verify in JSON

    echo "------------------------------------------------------------"
    echo "[$name]"
    echo "------------------------------------------------------------"

    # Run the command
    if eval "$cmd"; then
        echo "  Run: OK"

        # Check for JSON output - path is results/json/{system_name}/{token}/*.json
        json_file=$(find $RESULTS_DIR -name "*.json" -type f -newer /tmp/smoke_test_marker 2>/dev/null | head -1)
        if [ -n "$json_file" ] && [ -f "$json_file" ]; then
            if grep -q "$check_metric" "$json_file"; then
                echo "  JSON: OK (found $check_metric in $json_file)"
                ((PASSED++))
            else
                echo "  JSON: FAILED (missing $check_metric)"
                echo "    File: $json_file"
                echo "    Content sample: $(head -c 300 "$json_file")"
                ((FAILED++))
            fi
        else
            echo "  JSON: FAILED (no json file found in $RESULTS_DIR)"
            ls -la $RESULTS_DIR 2>/dev/null || echo "  Directory doesn't exist"
            find $RESULTS_DIR -name "*.json" 2>/dev/null | head -5 || true
            ((FAILED++))
        fi
    else
        echo "  Run: FAILED (exit code $?)"
        ((FAILED++))
    fi
    echo ""
}

# Create marker file to find new json files
touch /tmp/smoke_test_marker
sleep 1

# Test 1: DQN AdamO mode - TEST THE FIX FIRST
run_test "1/8 DQN + AdamO (decoupled ortho)" \
    "uv run python stoix/systems/q_learning/ff_dqn.py env=gymnax/cartpole arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" network.actor_network.pre_torso.activation=groupsort +system.ortho_mode=optimizer +system.ortho_coeff=0.001 +system.ortho_exclude_output=true logger.loggers.wandb.enabled=True logger.loggers.wandb.project=stoix_smoke_test logger.loggers.json.enabled=True" \
    "gram_deviation"

touch /tmp/smoke_test_marker; sleep 1

# Test 2: TD3 AdamO mode - TEST THE FIX
run_test "2/8 TD3 + AdamO (decoupled ortho)" \
    "uv run python stoix/systems/ddpg/ff_td3.py env=brax/halfcheetah arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" network.actor_network.pre_torso.activation=groupsort \"network.q_network.pre_torso.layer_sizes=$LAYERS\" network.q_network.pre_torso.activation=groupsort +system.ortho_mode=optimizer +system.ortho_coeff=0.001 +system.ortho_exclude_output=true system.epochs=4 system.warmup_steps=50 logger.loggers.wandb.enabled=True logger.loggers.wandb.project=stoix_smoke_test logger.loggers.json.enabled=True" \
    "gram_deviation"

touch /tmp/smoke_test_marker; sleep 1

# Test 3: DQN loss mode - verify ortho_loss appears
run_test "3/8 DQN + loss ortho" \
    "uv run python stoix/systems/q_learning/ff_dqn.py env=gymnax/cartpole arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" network.actor_network.pre_torso.activation=groupsort +system.ortho_mode=loss +system.ortho_lambda=0.2 +system.ortho_exclude_output=true logger.loggers.wandb.enabled=True logger.loggers.wandb.project=stoix_smoke_test logger.loggers.json.enabled=True" \
    "ortho_loss"

touch /tmp/smoke_test_marker; sleep 1

# Test 4: TD3 loss mode - verify q_ortho_loss appears
run_test "4/8 TD3 + loss ortho" \
    "uv run python stoix/systems/ddpg/ff_td3.py env=brax/halfcheetah arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" network.actor_network.pre_torso.activation=groupsort \"network.q_network.pre_torso.layer_sizes=$LAYERS\" network.q_network.pre_torso.activation=groupsort +system.ortho_mode=loss +system.ortho_lambda=0.2 +system.ortho_exclude_output=true system.epochs=4 system.warmup_steps=50 logger.loggers.wandb.enabled=True logger.loggers.wandb.project=stoix_smoke_test logger.loggers.json.enabled=True" \
    "q_ortho_loss"

touch /tmp/smoke_test_marker; sleep 1

# Test 5: DQN vanilla (no ortho) - baseline still works
run_test "5/8 DQN vanilla (baseline)" \
    "uv run python stoix/systems/q_learning/ff_dqn.py env=gymnax/cartpole arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" logger.loggers.wandb.enabled=True logger.loggers.wandb.project=stoix_smoke_test logger.loggers.json.enabled=True" \
    "q_loss"

touch /tmp/smoke_test_marker; sleep 1

# Test 6: PPO + AdamO mode
run_test "6/8 PPO + AdamO (decoupled ortho)" \
    "uv run python stoix/systems/ppo/anakin/ff_ppo.py env=gymnax/cartpole arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" network.actor_network.pre_torso.activation=groupsort \"network.critic_network.pre_torso.layer_sizes=$LAYERS\" network.critic_network.pre_torso.activation=groupsort +system.ortho_mode=optimizer +system.ortho_coeff=0.001 +system.ortho_exclude_output=true logger.loggers.wandb.enabled=True logger.loggers.wandb.project=stoix_smoke_test logger.loggers.json.enabled=True" \
    "gram_deviation"

touch /tmp/smoke_test_marker; sleep 1

# Test 7: PPO + loss ortho
run_test "7/8 PPO + loss ortho" \
    "uv run python stoix/systems/ppo/anakin/ff_ppo.py env=gymnax/cartpole arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" network.actor_network.pre_torso.activation=groupsort \"network.critic_network.pre_torso.layer_sizes=$LAYERS\" network.critic_network.pre_torso.activation=groupsort +system.ortho_mode=loss +system.ortho_lambda=0.2 +system.ortho_exclude_output=true logger.loggers.wandb.enabled=True logger.loggers.wandb.project=stoix_smoke_test logger.loggers.json.enabled=True" \
    "ortho_loss"

touch /tmp/smoke_test_marker; sleep 1

# Test 8: PPO vanilla (baseline)
run_test "8/8 PPO vanilla (baseline)" \
    "uv run python stoix/systems/ppo/anakin/ff_ppo.py env=gymnax/cartpole arch.total_timesteps=$STEPS arch.total_num_envs=$NUM_ENVS arch.num_evaluation=$NUM_EVAL \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" logger.loggers.wandb.enabled=True logger.loggers.wandb.project=stoix_smoke_test logger.loggers.json.enabled=True" \
    "actor_loss"

# Summary
echo "============================================================"
echo "RESULTS: $PASSED passed, $FAILED failed"
echo "============================================================"

if [ $FAILED -eq 0 ]; then
    echo "All tests passed! Ortho support working for DQN, TD3, and PPO."
    echo "Check WandB project: stoix_smoke_test"
    exit 0
else
    echo "Some tests failed. Check output above."
    exit 1
fi
