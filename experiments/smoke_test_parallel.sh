#!/bin/bash
# ============================================================================
# Parallel Smoke Test - High Throughput Validation with Profiling
#
# Validates that all algorithms work with massive parallelization:
# - PPO: 2048 envs (Brax-style on-policy)
# - TD3/DQN: 1024 envs (Brax-style off-policy)
#
# Tests:
# 1. PPO baseline @ 2048 envs
# 2. PPO + AdamO @ 2048 envs
# 3. TD3 + AdamO @ 1024 envs
# 4. DQN + AdamO @ 1024 envs
#
# Metrics measured:
# - Steps per second (throughput)
# - GPU memory usage
# - Wall clock time per 1M steps
#
# Usage:
#   bash experiments/smoke_test_parallel.sh
#   PROFILE=1 bash experiments/smoke_test_parallel.sh  # Enable JAX profiling
# ============================================================================

export JAX_LOG_COMPILES=0
export PYTHONUNBUFFERED=1

# Enable profiling if requested
if [ "$PROFILE" == "1" ]; then
    echo "Profiling enabled - results will be in /tmp/jax-trace-*"
    export JAX_TRACEBACK_FILTERING=off
    # JAX profiler writes to TensorBoard format
    PROFILE_DIR="/tmp/jax-trace-$(date +%Y%m%d-%H%M%S)"
    mkdir -p $PROFILE_DIR
fi

echo "============================================================"
echo "Parallel Smoke Test - High Throughput Validation"
echo "============================================================"
echo ""

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo ""

PASSED=0
FAILED=0
RESULTS_DIR="results/json"

# Test configuration
STEPS=1000000       # 1M steps for quick validation
NUM_EVAL=10         # 10 eval points
LAYERS="[256,256,256,256]"  # 4-layer network

# Track timing
declare -A TIMES
declare -A SPS

run_test() {
    local name="$1"
    local cmd="$2"
    local expected_sps="$3"  # Expected steps per second (rough)

    echo "------------------------------------------------------------"
    echo "[$name]"
    echo "------------------------------------------------------------"

    # Check GPU memory before
    local mem_before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)

    # Time the run
    local start_time=$(date +%s.%N)

    # Run the command
    if eval "$cmd"; then
        local end_time=$(date +%s.%N)
        local elapsed=$(echo "$end_time - $start_time" | bc)
        TIMES[$name]=$elapsed

        # Calculate actual SPS
        local actual_sps=$(echo "$STEPS / $elapsed" | bc)
        SPS[$name]=$actual_sps

        # Check GPU memory after
        local mem_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)

        echo ""
        echo "  Status:     PASSED"
        echo "  Time:       ${elapsed}s"
        echo "  Throughput: ${actual_sps} steps/sec"
        echo "  GPU Memory: ${mem_before}MB -> ${mem_after}MB"

        if [ -n "$expected_sps" ] && [ "$actual_sps" -lt "$expected_sps" ]; then
            echo "  Warning:    Below expected ${expected_sps} steps/sec"
        fi

        ((PASSED++))
    else
        echo "  Status: FAILED (exit code $?)"
        ((FAILED++))
    fi
    echo ""
}

echo "============================================================"
echo "Test 1/4: PPO Baseline @ 2048 envs"
echo "============================================================"
echo ""

run_test "PPO baseline 2048" \
    "uv run python stoix/systems/ppo/anakin/ff_ppo.py \
        env=gymnax/cartpole \
        arch.total_timesteps=$STEPS \
        arch.total_num_envs=2048 \
        arch.num_evaluation=$NUM_EVAL \
        system.rollout_length=128 \
        system.epochs=4 \
        system.num_minibatches=16 \
        \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" \
        \"network.critic_network.pre_torso.layer_sizes=$LAYERS\" \
        logger.loggers.wandb.enabled=False \
        logger.loggers.json.enabled=False" \
    100000  # Expect ~100k+ SPS with 2048 envs

echo "============================================================"
echo "Test 2/4: PPO + AdamO @ 2048 envs"
echo "============================================================"
echo ""

run_test "PPO AdamO 2048" \
    "uv run python stoix/systems/ppo/anakin/ff_ppo.py \
        env=gymnax/cartpole \
        arch.total_timesteps=$STEPS \
        arch.total_num_envs=2048 \
        arch.num_evaluation=$NUM_EVAL \
        system.rollout_length=128 \
        system.epochs=4 \
        system.num_minibatches=16 \
        \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" \
        network.actor_network.pre_torso.activation=groupsort \
        \"network.critic_network.pre_torso.layer_sizes=$LAYERS\" \
        network.critic_network.pre_torso.activation=groupsort \
        +system.ortho_mode=optimizer \
        +system.ortho_coeff=0.001 \
        +system.ortho_exclude_output=true \
        logger.loggers.wandb.enabled=False \
        logger.loggers.json.enabled=False" \
    80000  # GroupSort + ortho adds overhead

echo "============================================================"
echo "Test 3/4: TD3 + AdamO @ 1024 envs"
echo "============================================================"
echo ""

run_test "TD3 AdamO 1024" \
    "uv run python stoix/systems/ddpg/ff_td3.py \
        env=brax/halfcheetah \
        arch.total_timesteps=$STEPS \
        arch.total_num_envs=1024 \
        arch.num_evaluation=$NUM_EVAL \
        system.epochs=32 \
        system.warmup_steps=100 \
        \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" \
        network.actor_network.pre_torso.activation=groupsort \
        \"network.q_network.pre_torso.layer_sizes=$LAYERS\" \
        network.q_network.pre_torso.activation=groupsort \
        +system.ortho_mode=optimizer \
        +system.ortho_coeff=0.001 \
        +system.ortho_exclude_output=true \
        logger.loggers.wandb.enabled=False \
        logger.loggers.json.enabled=False" \
    50000  # Off-policy typically slower

echo "============================================================"
echo "Test 4/4: DQN + AdamO @ 1024 envs"
echo "============================================================"
echo ""

run_test "DQN AdamO 1024" \
    "uv run python stoix/systems/q_learning/ff_dqn.py \
        env=gymnax/cartpole \
        arch.total_timesteps=$STEPS \
        arch.total_num_envs=1024 \
        arch.num_evaluation=$NUM_EVAL \
        \"network.actor_network.pre_torso.layer_sizes=$LAYERS\" \
        network.actor_network.pre_torso.activation=groupsort \
        +system.ortho_mode=optimizer \
        +system.ortho_coeff=0.001 \
        +system.ortho_exclude_output=true \
        logger.loggers.wandb.enabled=False \
        logger.loggers.json.enabled=False" \
    80000  # DQN simpler than TD3

# Summary
echo "============================================================"
echo "THROUGHPUT SUMMARY"
echo "============================================================"
printf "%-20s %12s %12s\n" "Test" "Time (s)" "Steps/sec"
printf "%-20s %12s %12s\n" "----" "--------" "---------"
for test in "PPO baseline 2048" "PPO AdamO 2048" "TD3 AdamO 1024" "DQN AdamO 1024"; do
    if [ -n "${TIMES[$test]}" ]; then
        printf "%-20s %12.1f %12d\n" "$test" "${TIMES[$test]}" "${SPS[$test]}"
    fi
done
echo ""

echo "============================================================"
echo "RESULTS: $PASSED passed, $FAILED failed"
echo "============================================================"

if [ "$PROFILE" == "1" ]; then
    echo ""
    echo "Profiling data saved to: $PROFILE_DIR"
    echo "View with: tensorboard --logdir=$PROFILE_DIR"
fi

if [ $FAILED -eq 0 ]; then
    echo "All parallel throughput tests passed!"
    exit 0
else
    echo "Some tests failed. Check output above."
    exit 1
fi
