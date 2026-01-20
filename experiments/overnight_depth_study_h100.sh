#!/bin/bash
# ============================================================================
# Overnight Depth Study - H100 / 2xA100
#
# Runs DQN and TD3 depth scaling experiments with both ortho modes
# Plus coefficient sweep (1e-2, 1e-3, 1e-4) for shallow networks (4, 8 layers)
#
# Design (48 configs total):
#   PART 1 - Depth Study (0-31):
#     - DQN: 4 depths x 2 modes x 2 envs = 16 configs
#     - TD3: 4 depths x 2 modes x 2 envs = 16 configs
#
#   PART 2 - Coeff Sweep (32-47, AdamO only, depths 4,8):
#     - DQN: 2 depths x 2 coeffs x 2 envs = 8 configs  (1e-2, 1e-4; 1e-3 in part 1)
#     - TD3: 2 depths x 2 coeffs x 2 envs = 8 configs
#
# Split for 2xA100:
#   GPU 1: bash overnight_depth_study_h100.sh 0 23
#   GPU 2: bash overnight_depth_study_h100.sh 24 47
#
# Single H100: bash overnight_depth_study_h100.sh
# ============================================================================

export JAX_LOG_COMPILES=1
export PYTHONUNBUFFERED=1

START_IDX=${1:-0}
END_IDX=${2:-47}
FAILED=0
PASSED=0

echo "============================================================"
echo "Overnight Depth + Coeff Study"
echo "============================================================"
echo "Running configs $START_IDX to $END_IDX (of 48 total)"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Common settings
SEEDS="[42,43,44,45,46]"
NUM_ENVS=64
STEPS=10000000
NUM_EVAL=100

# Depths for main study
DEPTHS=(4 8 16 32)
# Depths for coeff sweep
SHALLOW_DEPTHS=(4 8)
# Coefficients for sweep (excluding 1e-3 which is default)
EXTRA_COEFFS=(0.01 0.0001)
EXTRA_COEFF_TAGS=("coeff_1e-2" "coeff_1e-4")

# Environments
DQN_ENVS=("gymnax/breakout" "gymnax/asterix")
DQN_ENV_SHORTS=("breakout" "asterix")
TD3_ENVS=("brax/halfcheetah" "brax/ant")
TD3_ENV_SHORTS=("halfcheetah" "ant")

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

for TASK_ID in $(seq $START_IDX $END_IDX); do

    if [ $TASK_ID -lt 32 ]; then
        # ================================================================
        # PART 1: Depth Study (configs 0-31)
        # ================================================================
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
            COEFF_ARG="+system.ortho_lambda=0.2"
            COEFF_TAG=""
        else
            MODE="optimizer"
            TAG="adamo"
            COEFF_ARG="+system.ortho_coeff=0.001"
            COEFF_TAG="coeff_1e-3"
        fi

        echo ""
        echo "============================================================"
        echo "[$TASK_ID/47] DEPTH STUDY"
        echo "============================================================"

        if [ $ALGO_IDX -eq 0 ]; then
            ALGO="DQN"
            ENV=${DQN_ENVS[$ENV_IDX]}
            ENV_SHORT=${DQN_ENV_SHORTS[$ENV_IDX]}

            echo "Algorithm:  $ALGO"
            echo "Env:        $ENV"
            echo "Depth:      $DEPTH layers"
            echo "Mode:       $MODE ($TAG)"
            echo ""

            if uv run python stoix/systems/q_learning/ff_dqn.py \
                env=$ENV \
                arch.seed=42 \
                arch.total_timesteps=$STEPS \
                arch.total_num_envs=$NUM_ENVS \
                arch.num_evaluation=$NUM_EVAL \
                "+multiseed=$SEEDS" \
                "network.actor_network.pre_torso.layer_sizes=$LAYERS" \
                network.actor_network.pre_torso.activation=groupsort \
                +system.ortho_mode=$MODE \
                $COEFF_ARG \
                +system.ortho_exclude_output=true \
                logger.loggers.wandb.enabled=True \
                logger.loggers.wandb.project=stoix_depth_study_h100 \
                "logger.loggers.wandb.tag=[depth_study,dqn,${TAG},depth_${DEPTH},${ENV_SHORT}${COEFF_TAG:+,}${COEFF_TAG}]" \
                logger.loggers.json.enabled=True; then
                ((PASSED++))
            else
                ((FAILED++))
                echo "FAILED: $ALGO $TAG depth=$DEPTH env=$ENV_SHORT"
            fi
        else
            ALGO="TD3"
            ENV=${TD3_ENVS[$ENV_IDX]}
            ENV_SHORT=${TD3_ENV_SHORTS[$ENV_IDX]}

            echo "Algorithm:  $ALGO"
            echo "Env:        $ENV"
            echo "Depth:      $DEPTH layers"
            echo "Mode:       $MODE ($TAG)"
            echo ""

            if uv run python stoix/systems/ddpg/ff_td3.py \
                env=$ENV \
                arch.seed=42 \
                arch.total_timesteps=$STEPS \
                arch.total_num_envs=$NUM_ENVS \
                arch.num_evaluation=$NUM_EVAL \
                "+multiseed=$SEEDS" \
                "network.actor_network.pre_torso.layer_sizes=$LAYERS" \
                network.actor_network.pre_torso.activation=groupsort \
                "network.q_network.pre_torso.layer_sizes=$LAYERS" \
                network.q_network.pre_torso.activation=groupsort \
                +system.ortho_mode=$MODE \
                $COEFF_ARG \
                +system.ortho_exclude_output=true \
                system.epochs=32 \
                system.warmup_steps=200 \
                system.actor_lr=3e-4 \
                system.q_lr=3e-4 \
                system.decay_learning_rates=false \
                logger.loggers.wandb.enabled=True \
                logger.loggers.wandb.project=stoix_depth_study_h100 \
                "logger.loggers.wandb.tag=[depth_study,td3,${TAG},depth_${DEPTH},${ENV_SHORT}${COEFF_TAG:+,}${COEFF_TAG}]" \
                logger.loggers.json.enabled=True; then
                ((PASSED++))
            else
                ((FAILED++))
                echo "FAILED: $ALGO $TAG depth=$DEPTH env=$ENV_SHORT"
            fi
        fi

    else
        # ================================================================
        # PART 2: Coefficient Sweep (configs 32-47)
        # AdamO only, depths 4 and 8, coeffs 1e-2 and 1e-4
        # ================================================================
        COEFF_TASK=$((TASK_ID - 32))
        ALGO_IDX=$((COEFF_TASK / 8))
        CONFIG_IDX=$((COEFF_TASK % 8))
        ENV_IDX=$((CONFIG_IDX / 4))
        REMAINING=$((CONFIG_IDX % 4))
        DEPTH_IDX=$((REMAINING / 2))
        COEFF_IDX=$((REMAINING % 2))

        DEPTH=${SHALLOW_DEPTHS[$DEPTH_IDX]}
        LAYERS=$(get_layers $DEPTH)
        COEFF=${EXTRA_COEFFS[$COEFF_IDX]}
        COEFF_TAG=${EXTRA_COEFF_TAGS[$COEFF_IDX]}

        echo ""
        echo "============================================================"
        echo "[$TASK_ID/47] COEFF SWEEP"
        echo "============================================================"

        if [ $ALGO_IDX -eq 0 ]; then
            ALGO="DQN"
            ENV=${DQN_ENVS[$ENV_IDX]}
            ENV_SHORT=${DQN_ENV_SHORTS[$ENV_IDX]}

            echo "Algorithm:  $ALGO"
            echo "Env:        $ENV"
            echo "Depth:      $DEPTH layers"
            echo "Coeff:      $COEFF ($COEFF_TAG)"
            echo ""

            if uv run python stoix/systems/q_learning/ff_dqn.py \
                env=$ENV \
                arch.seed=42 \
                arch.total_timesteps=$STEPS \
                arch.total_num_envs=$NUM_ENVS \
                arch.num_evaluation=$NUM_EVAL \
                "+multiseed=$SEEDS" \
                "network.actor_network.pre_torso.layer_sizes=$LAYERS" \
                network.actor_network.pre_torso.activation=groupsort \
                +system.ortho_mode=optimizer \
                +system.ortho_coeff=$COEFF \
                +system.ortho_exclude_output=true \
                logger.loggers.wandb.enabled=True \
                logger.loggers.wandb.project=stoix_depth_study_h100 \
                "logger.loggers.wandb.tag=[coeff_sweep,dqn,adamo,${COEFF_TAG},depth_${DEPTH},${ENV_SHORT}]" \
                logger.loggers.json.enabled=True; then
                ((PASSED++))
            else
                ((FAILED++))
                echo "FAILED: $ALGO coeff=$COEFF depth=$DEPTH env=$ENV_SHORT"
            fi
        else
            ALGO="TD3"
            ENV=${TD3_ENVS[$ENV_IDX]}
            ENV_SHORT=${TD3_ENV_SHORTS[$ENV_IDX]}

            echo "Algorithm:  $ALGO"
            echo "Env:        $ENV"
            echo "Depth:      $DEPTH layers"
            echo "Coeff:      $COEFF ($COEFF_TAG)"
            echo ""

            if uv run python stoix/systems/ddpg/ff_td3.py \
                env=$ENV \
                arch.seed=42 \
                arch.total_timesteps=$STEPS \
                arch.total_num_envs=$NUM_ENVS \
                arch.num_evaluation=$NUM_EVAL \
                "+multiseed=$SEEDS" \
                "network.actor_network.pre_torso.layer_sizes=$LAYERS" \
                network.actor_network.pre_torso.activation=groupsort \
                "network.q_network.pre_torso.layer_sizes=$LAYERS" \
                network.q_network.pre_torso.activation=groupsort \
                +system.ortho_mode=optimizer \
                +system.ortho_coeff=$COEFF \
                +system.ortho_exclude_output=true \
                system.epochs=32 \
                system.warmup_steps=200 \
                system.actor_lr=3e-4 \
                system.q_lr=3e-4 \
                system.decay_learning_rates=false \
                logger.loggers.wandb.enabled=True \
                logger.loggers.wandb.project=stoix_depth_study_h100 \
                "logger.loggers.wandb.tag=[coeff_sweep,td3,adamo,${COEFF_TAG},depth_${DEPTH},${ENV_SHORT}]" \
                logger.loggers.json.enabled=True; then
                ((PASSED++))
            else
                ((FAILED++))
                echo "FAILED: $ALGO coeff=$COEFF depth=$DEPTH env=$ENV_SHORT"
            fi
        fi
    fi

    echo ""
    echo "[$TASK_ID] Completed"
done

echo ""
echo "============================================================"
echo "Study completed!"
echo "RESULTS: $PASSED passed, $FAILED failed out of $((END_IDX - START_IDX + 1)) configs"
echo "============================================================"
