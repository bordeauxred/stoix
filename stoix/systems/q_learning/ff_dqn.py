import copy
import time
from typing import Any, Callable, Tuple

import chex
import flashbax as fbx
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf
from stoa import Environment, TimeStep, WrapperState, get_final_step_metrics

from stoix.base_types import (
    ActorApply,
    AnakinExperimentOutput,
    LearnerFn,
    OffPolicyLearnerState,
    OnlineAndTarget,
)
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.networks.base import FeedForwardActor as Actor
from stoix.systems.q_learning.dqn_types import Transition
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.loss import q_learning
from stoix.utils.orthogonalization import apply_ortho_update, compute_gram_regularization_loss
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate, make_optimizer


def get_warmup_fn(
    env: Environment,
    q_params: OnlineAndTarget,
    q_apply_fn: ActorApply,
    buffer_add_fn: Callable,
    config: DictConfig,
) -> Callable:
    def warmup(
        env_states: WrapperState,
        timesteps: TimeStep,
        buffer_states: BufferState,
        keys: chex.PRNGKey,
    ) -> Tuple[WrapperState, TimeStep, BufferState, chex.PRNGKey]:
        def _env_step(
            carry: Tuple[WrapperState, TimeStep, chex.PRNGKey], _: Any
        ) -> Tuple[Tuple[WrapperState, TimeStep, chex.PRNGKey], Transition]:
            """Step the environment."""

            env_state, last_timestep, key = carry
            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = q_apply_fn(q_params.online, last_timestep.observation)
            action = actor_policy.sample(seed=policy_key)

            # STEP ENVIRONMENT
            env_state, timestep = env.step(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]
            next_obs = timestep.extras["next_obs"]

            transition = Transition(
                last_timestep.observation, action, timestep.reward, done, next_obs, info
            )

            return (env_state, timestep, key), transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        (env_states, timesteps, keys), traj_batch = jax.lax.scan(
            _env_step, (env_states, timesteps, keys), None, config.system.warmup_steps
        )

        # Add the trajectory to the buffer.
        buffer_states = buffer_add_fn(buffer_states, traj_batch)

        return env_states, timesteps, keys, buffer_states

    batched_warmup_step: Callable = jax.vmap(
        warmup, in_axes=(0, 0, 0, 0), out_axes=(0, 0, 0, 0), axis_name="batch"
    )

    return batched_warmup_step


def get_learner_fn(
    env: Environment,
    q_apply_fn: ActorApply,
    q_update_fn: optax.TransformUpdateFn,
    buffer_fns: Tuple[Callable, Callable],
    config: DictConfig,
) -> LearnerFn[OffPolicyLearnerState]:
    """Get the learner function."""

    buffer_add_fn, buffer_sample_fn = buffer_fns

    # Ortho config with defaults
    ortho_mode = getattr(config.system, "ortho_mode", None)  # None, "loss", or "optimizer"
    ortho_lambda = getattr(config.system, "ortho_lambda", 0.2)  # for loss mode
    ortho_coeff = getattr(config.system, "ortho_coeff", 1e-3)  # for optimizer mode
    ortho_exclude_output = getattr(config.system, "ortho_exclude_output", True)
    # Get learning rate for decoupled ortho
    q_lr_value = getattr(config.system, "q_lr", 1e-3)

    def _update_step(
        learner_state: OffPolicyLearnerState, _: Any
    ) -> Tuple[OffPolicyLearnerState, Tuple]:
        def _env_step(
            learner_state: OffPolicyLearnerState, _: Any
        ) -> Tuple[OffPolicyLearnerState, Transition]:
            """Step the environment."""
            q_params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = q_apply_fn(q_params.online, last_timestep.observation)
            action = actor_policy.sample(seed=policy_key)

            # STEP ENVIRONMENT
            env_state, timestep = env.step(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]
            next_obs = timestep.extras["next_obs"]

            transition = Transition(
                last_timestep.observation, action, timestep.reward, done, next_obs, info
            )

            learner_state = OffPolicyLearnerState(
                q_params, opt_states, buffer_state, key, env_state, timestep
            )
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        # Add the trajectory to the buffer.
        buffer_state = buffer_add_fn(buffer_state, traj_batch)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _q_loss_fn(
                q_params: FrozenDict,
                target_q_params: FrozenDict,
                transitions: Transition,
            ) -> Tuple[jnp.ndarray, dict]:

                q_tm1 = q_apply_fn(q_params, transitions.obs).preferences
                q_t = q_apply_fn(target_q_params, transitions.next_obs).preferences

                # Cast and clip rewards.
                discount = 1.0 - transitions.done.astype(jnp.float32)
                d_t = (discount * config.system.gamma).astype(jnp.float32)
                r_t = jnp.clip(
                    transitions.reward, -config.system.max_abs_reward, config.system.max_abs_reward
                ).astype(jnp.float32)
                a_tm1 = transitions.action

                # Compute Q-learning loss.
                batch_td_loss = q_learning(
                    q_tm1,
                    a_tm1,
                    r_t,
                    d_t,
                    q_t,
                    config.system.huber_loss_parameter,
                )

                # Compute ortho regularization loss when in "loss" mode
                if ortho_mode == "loss":
                    ortho_loss, ortho_info = compute_gram_regularization_loss(
                        q_params, exclude_output=ortho_exclude_output
                    )
                    total_loss = batch_td_loss + ortho_lambda * ortho_loss
                    loss_info = {
                        "q_loss": batch_td_loss,
                        "ortho_loss": ortho_loss,
                        "total_loss": total_loss,
                        "gram_deviation": ortho_info["mean_gram_deviation"],
                    }
                else:
                    total_loss = batch_td_loss
                    loss_info = {
                        "q_loss": batch_td_loss,
                    }

                return total_loss, loss_info

            params, opt_states, buffer_state, key = update_state

            key, sample_key = jax.random.split(key)

            # SAMPLE TRANSITIONS
            transition_sample = buffer_sample_fn(buffer_state, sample_key)
            transitions: Transition = transition_sample.experience

            # CALCULATE Q LOSS
            q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
            q_grads, q_loss_info = q_grad_fn(
                params.online,
                params.target,
                transitions,
            )

            # Compute the parallel mean (pmean) over the batch.
            # This calculation is inspired by the Anakin architecture demo notebook.
            # available at https://tinyurl.com/26tdzs5x
            # This pmean could be a regular mean as the batch axis is on the same device.
            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="batch")
            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="device")

            # UPDATE Q PARAMS AND OPTIMISER STATE
            q_updates, q_new_opt_state = q_update_fn(q_grads, opt_states, params.online)
            q_new_online_params = optax.apply_updates(params.online, q_updates)

            # Apply decoupled ortho regularization if mode="optimizer"
            if ortho_mode == "optimizer":
                q_new_online_params = apply_ortho_update(
                    q_new_online_params, q_lr_value, ortho_coeff, ortho_exclude_output
                )

            # Target network polyak update.
            new_target_q_params = optax.incremental_update(
                q_new_online_params, params.target, config.system.tau
            )
            q_new_params = OnlineAndTarget(q_new_online_params, new_target_q_params)

            # PACK NEW PARAMS AND OPTIMISER STATE
            new_params = q_new_params
            new_opt_state = q_new_opt_state

            # PACK LOSS INFO
            loss_info = {
                **q_loss_info,
            }
            return (new_params, new_opt_state, buffer_state, key), loss_info

        update_state = (params, opt_states, buffer_state, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, buffer_state, key = update_state
        learner_state = OffPolicyLearnerState(
            params, opt_states, buffer_state, key, env_state, last_timestep
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: OffPolicyLearnerState,
    ) -> AnakinExperimentOutput[OffPolicyLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        """

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.arch.num_updates_per_eval
        )
        return AnakinExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[OffPolicyLearnerState], Actor, OffPolicyLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of actions.
    action_dim = int(env.action_space().num_values)
    config.system.action_dim = action_dim

    # PRNG keys.
    key, q_net_key = keys

    # Define networks and optimiser.
    q_network_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    q_network_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=action_dim,
        epsilon=config.system.training_epsilon,
    )

    q_network = Actor(torso=q_network_torso, action_head=q_network_action_head)

    eval_q_network_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=action_dim,
        epsilon=config.system.evaluation_epsilon,
    )
    eval_q_network = Actor(torso=q_network_torso, action_head=eval_q_network_action_head)

    q_lr = make_learning_rate(config.system.q_lr, config, config.system.epochs)
    q_optim = make_optimizer(q_lr, config)

    # Initialise observation
    init_x = env.observation_space().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    # Initialise q params and optimiser state.
    q_online_params = q_network.init(q_net_key, init_x)
    q_target_params = q_online_params
    q_opt_state = q_optim.init(q_online_params)

    params = OnlineAndTarget(q_online_params, q_target_params)
    opt_states = q_opt_state

    q_network_apply_fn = q_network.apply

    # Pack apply and update functions.
    apply_fns = q_network_apply_fn
    update_fns = q_optim.update

    # Create replay buffer
    dummy_transition = Transition(
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        action=jnp.zeros((), dtype=int),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        info={"episode_return": 0.0, "episode_length": 0, "is_terminal_step": False},
    )
    assert config.system.total_buffer_size % n_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total buffer size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )
    assert config.system.total_batch_size % n_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total batch size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )
    config.system.buffer_size = config.system.total_buffer_size // (
        n_devices * config.arch.update_batch_size
    )
    config.system.batch_size = config.system.total_batch_size // (
        n_devices * config.arch.update_batch_size
    )
    buffer_fn = fbx.make_item_buffer(
        max_length=config.system.buffer_size,
        min_length=config.system.batch_size,
        sample_batch_size=config.system.batch_size,
        add_batches=True,
        add_sequences=True,
    )
    buffer_fns = (buffer_fn.add, buffer_fn.sample)
    buffer_states = buffer_fn.init(dummy_transition)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, buffer_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    warmup = get_warmup_fn(env, params, q_network_apply_fn, buffer_fn.add, config)
    warmup = jax.pmap(warmup, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = env.reset(jnp.stack(env_keys))

    def reshape_states(x: chex.Array) -> chex.Array:
        return x.reshape(
            (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:]
        )

    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key, warmup_key = jax.random.split(key, num=3)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    warmup_keys = jax.random.split(warmup_key, n_devices * config.arch.update_batch_size)

    def reshape_keys(x: chex.Array) -> chex.Array:
        return x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])

    step_keys = reshape_keys(jnp.stack(step_keys))
    warmup_keys = reshape_keys(jnp.stack(warmup_keys))

    replicate_learner = (params, opt_states, buffer_states)

    # Duplicate learner for update_batch_size.
    def broadcast(x):
        # Skip non-array types (e.g., strings in optimizer state)
        if not hasattr(x, 'shape'):
            return x
        return jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)

    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    # Note: Now using standard Adam (no strings in state), so flax.jax_utils.replicate works.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, buffer_states = replicate_learner
    # Warmup the buffer.
    env_states, timesteps, keys, buffer_states = warmup(
        env_states, timesteps, buffer_states, warmup_keys
    )
    init_learner_state = OffPolicyLearnerState(
        params, opt_states, buffer_states, step_keys, env_states, timesteps
    )

    return learn, eval_q_network, init_learner_state


# =============================================================================
# PureJaxRL-Style Multi-Seed Training (Option 3)
#
# The make_train function returns a pure train(rng) function that can be vmapped
# for multi-seed training. This follows the PureJaxRL design pattern.
#
# Single seed:  result = jax.jit(train_fn)(jax.random.PRNGKey(42))
# Multi-seed:   results = jax.vmap(train_fn)(rngs)
# =============================================================================


def make_train(config: DictConfig) -> Callable[[chex.PRNGKey], Tuple[OnlineAndTarget, dict]]:
    """Create a pure training function that can be vmapped over seeds.

    This follows the PureJaxRL design pattern where train(rng) is a pure function
    that initializes everything from the RNG and returns final params + metrics.

    Args:
        config: Hydra config (will be deep-copied internally)

    Returns:
        train_fn: A function with signature train(rng) -> (params, metrics)
                  that can be vmapped for multi-seed training.

    Usage:
        # Single seed
        train_fn = make_train(config)
        params, metrics = jax.jit(train_fn)(jax.random.PRNGKey(42))

        # Multi-seed (just vmap!)
        seeds = jnp.array([42, 43, 44, 45, 46])
        rngs = jax.vmap(jax.random.PRNGKey)(seeds)
        all_params, all_metrics = jax.vmap(train_fn)(rngs)
    """
    config = copy.deepcopy(config)

    # Calculate total timesteps and validate config
    config.num_devices = 1  # Pure function runs on single device
    config.arch.update_batch_size = 1  # Single batch for pure function
    config = check_total_timesteps(config)

    # Create environment (can be created once, reused across seeds)
    env, _ = environments.make(config=config)
    action_dim = int(env.action_space().num_values)
    config.system.action_dim = action_dim

    # Create network architecture (shared structure, different params per seed)
    q_network_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    q_network_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=action_dim,
        epsilon=config.system.training_epsilon,
    )
    q_network = Actor(torso=q_network_torso, action_head=q_network_action_head)

    # Create optimizer (supports both standard Adam and AdamO based on ortho_mode)
    q_lr = make_learning_rate(config.system.q_lr, config, config.system.epochs)
    q_optim = make_optimizer(q_lr, config)

    # Ortho config with defaults
    ortho_mode = getattr(config.system, "ortho_mode", None)  # None, "loss", or "optimizer"
    ortho_lambda = getattr(config.system, "ortho_lambda", 0.2)  # for loss mode
    ortho_coeff = getattr(config.system, "ortho_coeff", 1e-3)  # for optimizer mode
    ortho_exclude_output = getattr(config.system, "ortho_exclude_output", True)
    # Get learning rate for decoupled ortho
    q_lr_value = getattr(config.system, "q_lr", 1e-3)

    # Create buffer template
    init_x = env.observation_space().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)
    dummy_transition = Transition(
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        action=jnp.zeros((), dtype=int),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        info={"episode_return": 0.0, "episode_length": 0, "is_terminal_step": False},
    )

    config.system.buffer_size = config.system.total_buffer_size
    config.system.batch_size = config.system.total_batch_size
    buffer_fn = fbx.make_item_buffer(
        max_length=config.system.buffer_size,
        min_length=config.system.batch_size,
        sample_batch_size=config.system.batch_size,
        add_batches=True,
        add_sequences=True,
    )

    def train(rng: chex.PRNGKey) -> Tuple[OnlineAndTarget, dict]:
        """Pure training function - vmappable over seeds.

        Args:
            rng: JAX PRNGKey for this training run

        Returns:
            params: Final trained OnlineAndTarget params
            metrics: Dict of accumulated metrics
        """
        # Split RNG for different uses
        rng, init_rng, env_rng, warmup_rng, train_rng = jax.random.split(rng, 5)

        # Initialize params (different for each seed due to different rng)
        q_online_params = q_network.init(init_rng, init_x)
        q_target_params = q_online_params
        q_opt_state = q_optim.init(q_online_params)
        params = OnlineAndTarget(q_online_params, q_target_params)

        # Initialize buffer
        buffer_state = buffer_fn.init(dummy_transition)

        # Initialize environment
        env_keys = jax.random.split(env_rng, config.arch.num_envs)
        env_states, timesteps = env.reset(env_keys)

        # Warmup: collect initial transitions, then add to buffer
        def _warmup_step(carry, _):
            env_state, timestep, key = carry
            key, policy_key = jax.random.split(key)
            actor_policy = q_network.apply(params.online, timestep.observation)
            action = actor_policy.sample(seed=policy_key)

            new_env_state, new_timestep = env.step(env_state, action)
            done = new_timestep.last().reshape(-1)
            info = new_timestep.extras["episode_metrics"]
            next_obs = new_timestep.extras["next_obs"]

            transition = Transition(
                timestep.observation, action, new_timestep.reward, done, next_obs, info
            )

            return (new_env_state, new_timestep, key), transition

        (env_states, timesteps, warmup_rng), traj_batch = jax.lax.scan(
            _warmup_step,
            (env_states, timesteps, warmup_rng),
            None,
            config.system.warmup_steps,
        )
        # Add collected trajectory to buffer (shape: warmup_steps, num_envs, ...)
        buffer_state = buffer_fn.add(buffer_state, traj_batch)

        # Progress logging: every 100k steps
        num_updates = config.arch.num_updates
        steps_per_update = config.system.rollout_length * config.arch.num_envs
        total_steps = num_updates * steps_per_update
        log_every_steps = 100_000  # Log every 100k environment steps
        log_interval = max(1, log_every_steps // steps_per_update)

        def _progress_callback(step, q_loss, episode_return, num_updates, steps_per_update):
            """Print progress during training."""
            current_steps = (step + 1) * steps_per_update
            total = num_updates * steps_per_update
            pct = 100.0 * current_steps / total
            ret_str = f"return: {episode_return:.1f}" if episode_return is not None else "return: --"
            print(f"\r[Step {current_steps:,}/{total:,}] {ret_str} | q_loss: {q_loss:.4f} | {pct:.1f}%", end="", flush=True)

        # Training step: collect rollout, add to buffer, then do gradient updates
        def _train_step(carry, step_idx):
            params, opt_state, buffer_state, env_state, timestep, key = carry

            # Collect rollout_length transitions
            def _env_step(env_carry, _):
                env_state, timestep, key = env_carry
                key, policy_key = jax.random.split(key)
                actor_policy = q_network.apply(params.online, timestep.observation)
                action = actor_policy.sample(seed=policy_key)

                new_env_state, new_timestep = env.step(env_state, action)
                done = new_timestep.last().reshape(-1)
                info = new_timestep.extras["episode_metrics"]
                next_obs = new_timestep.extras["next_obs"]

                transition = Transition(
                    timestep.observation, action, new_timestep.reward, done, next_obs, info
                )
                return (new_env_state, new_timestep, key), transition

            key, rollout_key = jax.random.split(key)
            (new_env_state, new_timestep, _), traj_batch = jax.lax.scan(
                _env_step,
                (env_state, timestep, rollout_key),
                None,
                config.system.rollout_length,
            )

            # Add collected trajectory to buffer
            new_buffer_state = buffer_fn.add(buffer_state, traj_batch)

            # Multiple gradient updates per rollout (epochs)
            def _update_epoch(update_carry, _):
                params, opt_state, key = update_carry
                key, sample_key = jax.random.split(key)

                # Sample from buffer
                sample = buffer_fn.sample(new_buffer_state, sample_key)
                transitions = sample.experience

                # Q-learning loss with optional ortho regularization
                def _q_loss_fn(q_params, target_q_params, transitions):
                    q_tm1 = q_network.apply(q_params, transitions.obs).preferences
                    q_t = q_network.apply(target_q_params, transitions.next_obs).preferences
                    discount = 1.0 - transitions.done.astype(jnp.float32)
                    d_t = (discount * config.system.gamma).astype(jnp.float32)
                    r_t = jnp.clip(
                        transitions.reward,
                        -config.system.max_abs_reward,
                        config.system.max_abs_reward,
                    ).astype(jnp.float32)
                    batch_td_loss = q_learning(
                        q_tm1, transitions.action, r_t, d_t, q_t,
                        config.system.huber_loss_parameter,
                    )

                    # Compute ortho regularization loss when in "loss" mode
                    if ortho_mode == "loss":
                        ortho_loss, ortho_info = compute_gram_regularization_loss(
                            q_params, exclude_output=ortho_exclude_output
                        )
                        total_loss = batch_td_loss + ortho_lambda * ortho_loss
                        return total_loss, {
                            "q_loss": batch_td_loss,
                            "ortho_loss": ortho_loss,
                            "total_loss": total_loss,
                            "gram_deviation": ortho_info["mean_gram_deviation"],
                        }
                    else:
                        return batch_td_loss, {"q_loss": batch_td_loss}

                # Compute gradients and update
                q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
                q_grads, loss_info = q_grad_fn(params.online, params.target, transitions)

                q_updates, new_opt_state = q_optim.update(q_grads, opt_state, params.online)
                new_online_params = optax.apply_updates(params.online, q_updates)

                # Apply decoupled ortho regularization if mode="optimizer"
                if ortho_mode == "optimizer":
                    new_online_params = apply_ortho_update(
                        new_online_params, q_lr_value, ortho_coeff, ortho_exclude_output
                    )

                new_target_params = optax.incremental_update(
                    new_online_params, params.target, config.system.tau
                )
                new_params = OnlineAndTarget(new_online_params, new_target_params)

                return (new_params, new_opt_state, key), loss_info

            key, epoch_key = jax.random.split(key)
            (new_params, new_opt_state, _), loss_infos = jax.lax.scan(
                _update_epoch,
                (params, opt_state, epoch_key),
                None,
                config.system.epochs,
            )

            # Collect metrics from trajectory batch
            q_loss_mean = jnp.mean(loss_infos["q_loss"])
            metrics = {
                "episode_return": traj_batch.info["episode_return"],
                "episode_length": traj_batch.info["episode_length"],
                "is_terminal": traj_batch.info["is_terminal_step"],
                "q_loss": q_loss_mean,
            }
            # Add ortho metrics if in loss mode
            if ortho_mode == "loss":
                metrics["ortho_loss"] = jnp.mean(loss_infos["ortho_loss"])
                metrics["gram_deviation"] = jnp.mean(loss_infos["gram_deviation"])

            # Compute episode return (mean over terminal steps)
            is_terminal = traj_batch.info["is_terminal_step"]
            episode_returns = traj_batch.info["episode_return"]
            # Mean return where episodes ended, or nan if none
            mean_return = jnp.where(
                jnp.any(is_terminal),
                jnp.sum(jnp.where(is_terminal, episode_returns, 0.0)) / jnp.maximum(jnp.sum(is_terminal), 1),
                jnp.nan,
            )

            # Progress logging (only at intervals to avoid slowdown)
            def _log_progress(step, q_loss, ep_return):
                import math
                # step/q_loss may be batched when vmapped over seeds
                step_val = int(step.flatten()[0]) if hasattr(step, 'flatten') else int(step)
                q_loss_val = float(q_loss.mean()) if hasattr(q_loss, 'mean') else float(q_loss)
                ep_return_val = float(ep_return.mean()) if hasattr(ep_return, 'mean') else float(ep_return)
                ep_return_val = None if math.isnan(ep_return_val) else ep_return_val
                if step_val % log_interval == 0:
                    _progress_callback(step_val, q_loss_val, ep_return_val, num_updates, steps_per_update)

            jax.debug.callback(_log_progress, step_idx, q_loss_mean, mean_return)

            return (new_params, new_opt_state, new_buffer_state, new_env_state, new_timestep, key), metrics

        # Run training
        init_carry = (params, q_opt_state, buffer_state, env_states, timesteps, train_rng)
        step_indices = jnp.arange(num_updates)
        (final_params, _, _, _, _, _), all_metrics = jax.lax.scan(
            _train_step, init_carry, step_indices
        )

        # Print newline after progress bar
        jax.debug.callback(lambda: print())

        return final_params, all_metrics

    return train


def run_multiseed_experiment(
    config: DictConfig,
    seeds: list,
    logger: StoixLogger = None,
) -> Tuple[OnlineAndTarget, dict]:
    """Run multi-seed training using vmap over seeds.

    This is the PureJaxRL-style approach where we vmap over the pure train function.

    Args:
        config: Hydra config
        seeds: List of seed values to train with
        logger: Optional StoixLogger for logging

    Returns:
        all_params: Params for each seed (shape: [num_seeds, ...])
        all_metrics: Metrics for each seed (shape: [num_seeds, num_steps, ...])
    """
    import time

    num_seeds = len(seeds)
    print(f"{Fore.CYAN}{Style.BRIGHT}Running {num_seeds} seeds in parallel: {seeds}{Style.RESET_ALL}")

    # Create the training function
    train_fn = make_train(config)

    # Create RNGs for each seed
    seed_array = jnp.array(seeds)
    rngs = jax.vmap(jax.random.PRNGKey)(seed_array)

    # Vmap and JIT the training function
    vmapped_train = jax.jit(jax.vmap(train_fn))

    # Run training
    start_time = time.time()
    print(f"{Fore.YELLOW}{Style.BRIGHT}Starting JIT compilation...{Style.RESET_ALL}")

    all_params, all_metrics = vmapped_train(rngs)
    jax.block_until_ready(all_params)

    elapsed_time = time.time() - start_time
    print(f"{Fore.GREEN}{Style.BRIGHT}Training completed in {elapsed_time:.1f}s{Style.RESET_ALL}")

    # Log results if logger provided
    if logger is not None:
        # Log per-seed final metrics
        for i, seed in enumerate(seeds):
            # Get final episode return for this seed (where terminal)
            is_terminal = all_metrics["is_terminal"][i]
            if jnp.any(is_terminal):
                terminal_returns = jnp.where(
                    is_terminal,
                    all_metrics["episode_return"][i],
                    jnp.nan,
                )
                final_return = jnp.nanmean(terminal_returns)
            else:
                final_return = all_metrics["episode_return"][i, -1]

            logger.log(
                {
                    f"seed_{seed}/episode_return": float(final_return),
                    f"seed_{seed}/q_loss": float(jnp.mean(all_metrics["q_loss"][i])),
                },
                int(config.arch.total_timesteps),
                0,
                LogEvent.EVAL,
            )

        # Log aggregate metrics
        final_returns = []
        for i in range(num_seeds):
            is_terminal = all_metrics["is_terminal"][i]
            if jnp.any(is_terminal):
                terminal_returns = jnp.where(
                    is_terminal,
                    all_metrics["episode_return"][i],
                    jnp.nan,
                )
                final_returns.append(float(jnp.nanmean(terminal_returns)))
            else:
                final_returns.append(float(all_metrics["episode_return"][i, -1]))

        final_returns = jnp.array(final_returns)
        logger.log(
            {
                "episode_return": float(jnp.mean(final_returns)),
                "episode_return_std": float(jnp.std(final_returns)),
            },
            int(config.arch.total_timesteps),
            0,
            LogEvent.EVAL,
        )

    return all_params, all_metrics


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment (original single-seed with intermediate evaluation)."""
    config = copy.deepcopy(_config)

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates >= config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Create the environments for train and eval.
    env, eval_env = environments.make(config=config)

    # PRNG keys.
    key, key_e, q_net_key = jax.random.split(jax.random.PRNGKey(config.arch.seed), num=3)

    # Setup learner.
    learn, eval_q_network, learner_state = learner_setup(env, (key, q_net_key), config)

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        eval_act_fn=get_distribution_act_fn(config, eval_q_network.apply),
        params=learner_state.params.online,
        config=config,
    )

    # Calculate number of updates per evaluation.
    config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = StoixLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))
    print(f"{Fore.YELLOW}{Style.BRIGHT}JAX Global Devices {jax.devices()}{Style.RESET_ALL}")

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = unreplicate_batch_dim(learner_state.params.online)
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        train_metrics = learner_output.train_metrics
        # Calculate the number of optimiser steps per second. Since gradients are aggregated
        # across the device and batch axis, we don't consider updates per device/batch as part of
        # the SPS for the learner.
        opt_steps_per_eval = config.arch.num_updates_per_eval * (config.system.epochs)
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_params = unreplicate_batch_dim(
            learner_output.learner_state.params.online
        )  # Select only actor params
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        evaluator_output = evaluator(trained_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        # Log the results of the evaluation.
        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])

        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

        if save_checkpoint:
            checkpointer.save(
                timestep=int(steps_per_rollout * (eval_step + 1)),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        evaluator_output = absolute_metric_evaluator(best_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()
    # Record the performance for the final evaluation run. If the absolute metric is not
    # calculated, this will be the final evaluation run.
    eval_performance = float(jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric]))
    return eval_performance


@hydra.main(
    config_path="../../configs/default/anakin",
    config_name="default_ff_dqn.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point.

    Supports multi-seed training via +multiseed=[42,43,44,45,46] argument.
    When multiseed is provided, uses PureJaxRL-style vmap over seeds.
    """
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Check for multi-seed mode
    multiseed = OmegaConf.select(cfg, "multiseed", default=None)

    if multiseed is not None:
        # Multi-seed mode: use PureJaxRL-style vmapped training
        seeds = list(multiseed)
        print(f"{Fore.CYAN}{Style.BRIGHT}Multi-seed mode: {seeds}{Style.RESET_ALL}")

        # Run multi-seed training (no logging during training)
        all_params, all_metrics = run_multiseed_experiment(cfg, seeds, logger=None)

        # Extract config info for naming
        env_name = OmegaConf.select(cfg, "env.scenario.task_name", default="unknown")
        layer_sizes = OmegaConf.select(cfg, "network.actor_network.pre_torso.layer_sizes", default=[256, 256])
        depth = len(layer_sizes) if layer_sizes else 2
        activation = OmegaConf.select(cfg, "network.actor_network.pre_torso.activation", default="silu")

        # Create group name (shared by all seeds)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        group_name = f"dqn_{env_name}_d{depth}_{activation}_{timestamp}"

        # Log to WandB (separate run per seed) and JSON
        if cfg.logger.loggers.wandb.enabled:
            from stoix.utils.logger import log_multiseed_wandb
            steps_per_log = cfg.system.rollout_length * cfg.arch.total_num_envs
            log_multiseed_wandb(
                all_metrics=all_metrics,
                seeds=seeds,
                config=cfg,
                group_name=group_name,
                env_name=env_name,
                depth=depth,
                activation=activation,
                utd=1,  # DQN doesn't have UTD
                steps_per_log=steps_per_log,
            )

        # Return mean performance across seeds
        final_returns = []
        for i in range(len(seeds)):
            is_terminal = all_metrics["is_terminal"][i]
            if jnp.any(is_terminal):
                terminal_returns = jnp.where(
                    is_terminal,
                    all_metrics["episode_return"][i],
                    jnp.nan,
                )
                final_returns.append(float(jnp.nanmean(terminal_returns)))
            else:
                final_returns.append(float(all_metrics["episode_return"][i, -1]))

        eval_performance = float(jnp.mean(jnp.array(final_returns)))
        print(f"{Fore.CYAN}{Style.BRIGHT}DQN multi-seed experiment completed{Style.RESET_ALL}")
        print(f"Mean return: {eval_performance:.2f} ± {float(jnp.std(jnp.array(final_returns))):.2f}")
    else:
        # Single-seed mode: use original implementation
        eval_performance = run_experiment(cfg)
        print(f"{Fore.CYAN}{Style.BRIGHT}DQN experiment completed{Style.RESET_ALL}")

    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
