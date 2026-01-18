#!/bin/bash
# Quick smoke test for TD3 multi-seed - run on head node to catch config errors
# No GPU needed, just tests Hydra parsing and basic JAX CPU execution
#
# New logging schema: Separate WandB run per seed with shared group
# - Run names: td3_pendulum_d4_silu_utd4_{timestamp}_s42, _s43, etc.
# - Group: td3_pendulum_d4_silu_utd4_{timestamp}
# - WandB aggregates grouped runs automatically (mean ± std)

set -e

echo "=== TD3 Multi-seed Smoke Test (CPU) ==="
echo "This should complete in <3 minutes on CPU"
echo "Testing per-seed WandB logging (separate runs per seed)"
echo ""

# Use offline mode by default - set WANDB_MODE=online to log to cloud
export WANDB_MODE=${WANDB_MODE:-offline}

# Minimal config: 1000 steps, 2 envs, 2 seeds
# The code now handles run naming and creates separate runs per seed
uv run python stoix/systems/ddpg/ff_td3.py \
    env=gymnax/pendulum \
    arch.seed=42 \
    arch.total_timesteps=1000 \
    arch.num_evaluation=2 \
    arch.total_num_envs=2 \
    '+multiseed=[42,43]' \
    system.epochs=4 \
    system.warmup_steps=8 \
    "network.actor_network.pre_torso.layer_sizes=[256,256,256,256]" \
    "network.q_network.pre_torso.layer_sizes=[256,256,256,256]" \
    logger.loggers.wandb.enabled=True \
    logger.loggers.wandb.project=stoix_smoke_test \
    "logger.loggers.wandb.tag=[smoke_test]"

echo ""
echo "=== TD3 Smoke test PASSED ==="
echo "Check WandB for 2 separate runs in the same group"
echo "JSON metrics saved to: results/json/td3_pendulum_d4_silu_utd4_*/seed_*/metrics.json"

# Cleanup offline wandb files
rm -rf wandb/offline-* 2>/dev/null || true
