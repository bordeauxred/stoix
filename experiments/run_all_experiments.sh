#!/bin/bash
# Sequential experiment runner for non-SLURM clusters
# Runs experiments one at a time on a single GPU
#
# Currently running: DQN depth study with ReLU activation
#
# Usage:
#   bash experiments/run_all_experiments.sh

set -e

# Configuration
SEEDS="[42,43,44,45,46]"
NUM_ENVS=64
ACTIVATION="relu"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[DONE]${NC} $1"; }
log_section() { echo -e "\n${YELLOW}============================================================${NC}"; echo -e "${YELLOW}$1${NC}"; echo -e "${YELLOW}============================================================${NC}\n"; }

# ============================================================================
# DQN Depth Study (ReLU activation)
# ============================================================================
run_dqn_depth_study() {
    log_section "DQN Depth Study (ReLU) - 6 envs × 5 depths = 30 configurations"

    SYSTEM="stoix/systems/q_learning/ff_dqn.py"
    COMMON="arch.total_num_envs=$NUM_ENVS"

    # Environments with step counts
    declare -A ENVS=(
        ["gymnax/breakout"]="minatar_breakout:10000000"
        ["gymnax/asterix"]="minatar_asterix:10000000"
        ["gymnax/freeway"]="minatar_freeway:10000000"
        ["gymnax/space_invaders"]="minatar_spaceinv:10000000"
        ["gymnax/pendulum"]="pendulum:2000000"
        ["gymnax/mountain_car"]="mountaincar:2000000"
    )

    # Network depths
    declare -A DEPTHS=(
        [2]="[256,256]"
        [4]="[256,256,256,256]"
        [8]="[256,256,256,256,256,256,256,256]"
        [16]="[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]"
        [32]="[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]"
    )

    total=$((${#ENVS[@]} * ${#DEPTHS[@]}))
    current=0

    for env in "${!ENVS[@]}"; do
        IFS=':' read -r env_short steps <<< "${ENVS[$env]}"

        for depth in "${!DEPTHS[@]}"; do
            ((current++))
            layers="${DEPTHS[$depth]}"

            log_info "[$current/$total] DQN depth=$depth activation=$ACTIVATION on $env_short ($steps steps)"

            uv run python $SYSTEM \
                env=$env \
                arch.seed=42 \
                arch.total_timesteps=$steps \
                +multiseed=$SEEDS \
                $COMMON \
                "network.actor_network.pre_torso.layer_sizes=$layers" \
                "network.actor_network.pre_torso.activation=$ACTIVATION" \
                logger.loggers.wandb.enabled=True \
                logger.loggers.wandb.project=stoix_dqn_depth_study \
                "logger.loggers.wandb.tag=[depth_study,$ACTIVATION]"

            log_success "DQN depth=$depth on $env_short completed"
        done
    done

    log_success "DQN Depth Study complete!"
}

# ============================================================================
# TD3 UTD Study (commented out - run after DQN depth study)
# ============================================================================
# run_td3_utd_study() {
#     log_section "TD3 UTD Study - 6 envs × 7 UTD ratios = 42 configurations"
#
#     SYSTEM="stoix/systems/ddpg/ff_td3.py"
#     COMMON="arch.total_num_envs=$NUM_ENVS"
#
#     # Brax environments with step counts
#     declare -A ENVS=(
#         ["brax/halfcheetah"]="halfcheetah:2000000"
#         ["brax/hopper"]="hopper:2000000"
#         ["brax/walker2d"]="walker2d:2000000"
#         ["brax/ant"]="ant:4000000"
#         ["brax/humanoid"]="humanoid:10000000"
#         ["gymnax/pendulum"]="pendulum:500000"
#     )
#
#     # UTD ratios (epochs parameter)
#     UTD_RATIOS=(1 2 4 8 16 32 64)
#
#     # Fixed 4-layer network
#     LAYERS="[256,256,256,256]"
#
#     total=$((${#ENVS[@]} * ${#UTD_RATIOS[@]}))
#     current=0
#
#     for env in "${!ENVS[@]}"; do
#         IFS=':' read -r env_short steps <<< "${ENVS[$env]}"
#
#         for utd in "${UTD_RATIOS[@]}"; do
#             ((current++))
#
#             log_info "[$current/$total] TD3 UTD=$utd on $env_short ($steps steps)"
#
#             uv run python $SYSTEM \
#                 env=$env \
#                 arch.seed=42 \
#                 arch.total_timesteps=$steps \
#                 +multiseed=$SEEDS \
#                 $COMMON \
#                 system.epochs=$utd \
#                 "network.actor_network.pre_torso.layer_sizes=$LAYERS" \
#                 "network.actor_network.pre_torso.activation=$ACTIVATION" \
#                 "network.q_network.pre_torso.layer_sizes=$LAYERS" \
#                 "network.q_network.pre_torso.activation=$ACTIVATION" \
#                 logger.loggers.wandb.enabled=True \
#                 logger.loggers.wandb.project=stoix_td3_utd_study \
#                 "logger.loggers.wandb.tag=[utd_study,$ACTIVATION]"
#
#             log_success "TD3 UTD=$utd on $env_short completed"
#         done
#     done
#
#     log_success "TD3 UTD Study complete!"
# }

# ============================================================================
# TD3 Depth Study (commented out - run after UTD study determines best ratio)
# ============================================================================
# run_td3_depth_study() {
#     log_section "TD3 Depth Study - 6 envs × 6 depths = 36 configurations"
#
#     # NOTE: This should use the optimal UTD found from the UTD study
#     # Default to UTD=4 if not specified
#     BEST_UTD=${TD3_BEST_UTD:-4}
#     log_info "Using UTD=$BEST_UTD (set TD3_BEST_UTD env var to override)"
#
#     SYSTEM="stoix/systems/ddpg/ff_td3.py"
#     COMMON="arch.total_num_envs=$NUM_ENVS"
#
#     # Brax environments with step counts
#     declare -A ENVS=(
#         ["brax/halfcheetah"]="halfcheetah:2000000"
#         ["brax/hopper"]="hopper:2000000"
#         ["brax/walker2d"]="walker2d:2000000"
#         ["brax/ant"]="ant:4000000"
#         ["brax/humanoid"]="humanoid:10000000"
#         ["gymnax/pendulum"]="pendulum:500000"
#     )
#
#     # Network depths
#     declare -A DEPTHS=(
#         [2]="[256,256]"
#         [4]="[256,256,256,256]"
#         [8]="[256,256,256,256,256,256,256,256]"
#         [16]="[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]"
#         [32]="[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]"
#         [64]="[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]"
#     )
#
#     total=$((${#ENVS[@]} * ${#DEPTHS[@]}))
#     current=0
#
#     for env in "${!ENVS[@]}"; do
#         IFS=':' read -r env_short steps <<< "${ENVS[$env]}"
#
#         for depth in "${!DEPTHS[@]}"; do
#             ((current++))
#             layers="${DEPTHS[$depth]}"
#
#             log_info "[$current/$total] TD3 depth=$depth UTD=$BEST_UTD on $env_short ($steps steps)"
#
#             uv run python $SYSTEM \
#                 env=$env \
#                 arch.seed=42 \
#                 arch.total_timesteps=$steps \
#                 +multiseed=$SEEDS \
#                 $COMMON \
#                 system.epochs=$BEST_UTD \
#                 "network.actor_network.pre_torso.layer_sizes=$layers" \
#                 "network.actor_network.pre_torso.activation=$ACTIVATION" \
#                 "network.q_network.pre_torso.layer_sizes=$layers" \
#                 "network.q_network.pre_torso.activation=$ACTIVATION" \
#                 logger.loggers.wandb.enabled=True \
#                 logger.loggers.wandb.project=stoix_td3_depth_study \
#                 "logger.loggers.wandb.tag=[depth_study,utd$BEST_UTD,$ACTIVATION]"
#
#             log_success "TD3 depth=$depth on $env_short completed"
#         done
#     done
#
#     log_success "TD3 Depth Study complete!"
# }

# ============================================================================
# Main
# ============================================================================
main() {
    log_section "Running DQN Depth Study with ReLU activation"
    log_info "6 environments × 5 depths × 5 seeds = 30 configurations"
    log_info "Activation: $ACTIVATION"
    echo ""

    run_dqn_depth_study

    log_section "DQN Depth Study complete!"
}

main "$@"
