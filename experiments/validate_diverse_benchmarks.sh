#!/bin/bash
#SBATCH --job-name=val_diverse
#SBATCH --partition=kisski
#SBATCH --array=0-11%6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/val_diverse_%A_%a.out
#SBATCH --error=logs/val_diverse_%A_%a.err
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
# Diverse Benchmark Validation Suite
#
# Three benchmark suites to validate Stoix:
# 1. MuJoCo Playground (MJX) - Gold standard continuous control
# 2. Gymnax Continuous - Simple JAX-native sanity checks
# 3. Brax Generalized - Alternative physics backend (NOT spring!)
#
# All use sensible hyperparameters based on literature.
# ============================================================================

STEPS=1000000  # 1M steps for validation

# Sensible defaults for off-policy (based on TD3/SAC papers)
# NOTE: warmup_steps IS multiplied by num_envs! Standard is ~10K total warmup.
# With 64 envs: warmup_steps=200 gives 12.8K total transitions
OFF_POLICY_COMMON="arch.total_num_envs=64 system.total_buffer_size=1000000"
OFF_POLICY_HP="system.actor_lr=3e-4 system.q_lr=3e-4 system.decay_learning_rates=false"
LAYERS_2="network.actor_network.pre_torso.layer_sizes=[256,256] network.q_network.pre_torso.layer_sizes=[256,256]"
WARMUP="system.warmup_steps=200"  # 200 * 64 envs = 12.8K total warmup transitions

# On-policy settings
ON_POLICY_COMMON="arch.total_num_envs=2048"
PPO_LAYERS="network.actor_network.pre_torso.layer_sizes=[256,256] network.critic_network.pre_torso.layer_sizes=[256,256]"

case $TASK_ID in
    # ========================================
    # MuJoCo Playground (MJX) - TD3
    # ========================================
    0)  ALGO="td3"; ENV="mjc_playground/dm_control/hopper_hop"; SYSTEM="stoix/systems/ddpg/ff_td3.py"
        EXTRA="$OFF_POLICY_COMMON $OFF_POLICY_HP $WARMUP $LAYERS_2" ;;
    1)  ALGO="td3"; ENV="mjc_playground/dm_control/cartpole_balance"; SYSTEM="stoix/systems/ddpg/ff_td3.py"
        EXTRA="$OFF_POLICY_COMMON $OFF_POLICY_HP $WARMUP $LAYERS_2" ;;
    2)  ALGO="td3"; ENV="mjc_playground/locomotion/go_1_joystick_flat_terrain"; SYSTEM="stoix/systems/ddpg/ff_td3.py"
        EXTRA="$OFF_POLICY_COMMON $OFF_POLICY_HP $WARMUP $LAYERS_2" ;;

    # ========================================
    # MuJoCo Playground (MJX) - SAC
    # ========================================
    3)  ALGO="sac"; ENV="mjc_playground/dm_control/hopper_hop"; SYSTEM="stoix/systems/sac/ff_sac.py"
        EXTRA="$OFF_POLICY_COMMON $WARMUP $LAYERS_2" ;;
    4)  ALGO="sac"; ENV="mjc_playground/dm_control/cartpole_balance"; SYSTEM="stoix/systems/sac/ff_sac.py"
        EXTRA="$OFF_POLICY_COMMON $WARMUP $LAYERS_2" ;;

    # ========================================
    # Gymnax Continuous - Simple sanity checks
    # ========================================
    5)  ALGO="td3"; ENV="gymnax/pendulum"; SYSTEM="stoix/systems/ddpg/ff_td3.py"
        EXTRA="$OFF_POLICY_COMMON $OFF_POLICY_HP $WARMUP $LAYERS_2" ;;
    6)  ALGO="td3"; ENV="gymnax/mountain_car_continuous"; SYSTEM="stoix/systems/ddpg/ff_td3.py"
        EXTRA="$OFF_POLICY_COMMON $OFF_POLICY_HP $WARMUP $LAYERS_2" ;;
    7)  ALGO="sac"; ENV="gymnax/pendulum"; SYSTEM="stoix/systems/sac/ff_sac.py"
        EXTRA="$OFF_POLICY_COMMON $WARMUP $LAYERS_2" ;;
    8)  ALGO="ppo"; ENV="gymnax/pendulum"; SYSTEM="stoix/systems/ppo/anakin/ff_ppo_continuous.py"
        EXTRA="$ON_POLICY_COMMON $PPO_LAYERS" ;;

    # ========================================
    # Brax with GENERALIZED backend (not spring!)
    # ========================================
    9)  ALGO="td3"; ENV="brax/halfcheetah"; SYSTEM="stoix/systems/ddpg/ff_td3.py"
        EXTRA="$OFF_POLICY_COMMON $OFF_POLICY_HP $WARMUP $LAYERS_2 env.kwargs.backend=generalized" ;;
    10) ALGO="sac"; ENV="brax/halfcheetah"; SYSTEM="stoix/systems/sac/ff_sac.py"
        EXTRA="$OFF_POLICY_COMMON $WARMUP $LAYERS_2 env.kwargs.backend=generalized" ;;
    11) ALGO="ppo"; ENV="brax/halfcheetah"; SYSTEM="stoix/systems/ppo/anakin/ff_ppo_continuous.py"
        EXTRA="$ON_POLICY_COMMON $PPO_LAYERS env.kwargs.backend=generalized" ;;
esac

ENV_SHORT=$(echo $ENV | sed 's/\//_/g')

echo "============================================================"
echo "Task $TASK_ID: $ALGO on $ENV (1M steps)"
echo "============================================================"

uv run python $SYSTEM \
    env=$ENV \
    arch.seed=42 \
    arch.total_timesteps=$STEPS \
    logger.loggers.wandb.enabled=False \
    logger.loggers.json.enabled=True \
    logger.loggers.json.path=diverse_validation/${ALGO}_${ENV_SHORT} \
    $EXTRA

echo "============================================================"
echo "Task $TASK_ID ($ALGO on $ENV) completed"
echo "============================================================"
