from typing import Callable, List, Optional, Union

import optax
from omegaconf import DictConfig

from stoix.utils.orthogonalization import adamo


def make_learning_rate_schedule(
    init_lr: float, num_updates: int, num_epochs: int, num_minibatches: int
) -> Callable:
    """Makes a very simple linear learning rate scheduler.

    Args:
        init_lr: initial learning rate.
        num_updates: number of updates.
        num_epochs: number of epochs.
        num_minibatches: number of minibatches.

    Note:
        We use a simple linear learning rate scheduler based on the suggestions from a blog on PPO
        implementation details which can be viewed at http://tinyurl.com/mr3chs4p
        This function can be extended to have more complex learning rate schedules by adding any
        relevant arguments to the system config and then parsing them accordingly here.
    """

    def linear_scedule(count: int) -> float:
        frac: float = 1.0 - (count // (num_epochs * num_minibatches)) / num_updates
        return init_lr * frac

    return linear_scedule


def make_learning_rate(
    init_lr: float, config: DictConfig, num_epochs: int, num_minibatches: Optional[int] = None
) -> Union[float, Callable]:
    """Returns a constant learning rate or a learning rate schedule.

    Args:
        init_lr: initial learning rate.
        config: system configuration.
        num_epochs: number of epochs.
        num_minibatches: number of minibatches.

    Returns:
        A learning rate schedule or fixed learning rate.
    """
    if num_minibatches is None:
        num_minibatches = 1

    if config.system.decay_learning_rates:
        return make_learning_rate_schedule(
            init_lr, config.arch.num_updates, num_epochs, num_minibatches
        )
    else:
        return init_lr


def make_optimizer(
    learning_rate: Union[float, Callable],
    config: DictConfig,
    max_grad_norm: Optional[float] = None,
    eps: float = 1e-5,
    exclude_layers: Optional[List[str]] = None,
) -> optax.GradientTransformation:
    """Create an optimizer based on config settings.

    Supports both standard Adam and AdamO (Adam with decoupled orthonormalization).
    The ortho_mode config determines which optimizer to use:
    - "loss": Standard Adam (ortho applied via loss function)
    - "optimizer": AdamO (ortho decoupled, applied in optimizer)

    Args:
        learning_rate: Learning rate (scalar or schedule)
        config: DictConfig with system settings. Reads:
            - system.ortho_mode: "loss" or "optimizer" (default: "loss")
            - system.ortho_coeff: Ortho coefficient for AdamO (default: 1e-3)
            - system.ortho_exclude_output: Whether to exclude output layer (default: True)
            - system.max_grad_norm: Max gradient norm for clipping (used if max_grad_norm not provided)
        max_grad_norm: Override for gradient clipping norm. If None, uses config.system.max_grad_norm
        eps: Adam epsilon
        exclude_layers: Optional list of layer name patterns to exclude from ortho

    Returns:
        optax.GradientTransformation optimizer

    Example:
        # In learner_setup:
        q_lr = make_learning_rate(config.system.q_lr, config, config.system.epochs)
        q_optim = make_optimizer(q_lr, config)

        # With custom grad norm:
        actor_optim = make_optimizer(actor_lr, config, max_grad_norm=1.0)
    """
    # Get config values with defaults
    ortho_mode = getattr(config.system, "ortho_mode", "loss")
    grad_norm = max_grad_norm if max_grad_norm is not None else getattr(config.system, "max_grad_norm", 0.5)

    if ortho_mode == "optimizer":
        # AdamO: Adam with decoupled orthonormalization
        ortho_coeff = getattr(config.system, "ortho_coeff", 1e-3)
        ortho_exclude_output = getattr(config.system, "ortho_exclude_output", True)
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_norm),
            adamo(
                learning_rate=learning_rate,
                eps=eps,
                ortho_coeff=ortho_coeff,
                exclude_output=ortho_exclude_output,
            ),
        )
    else:
        # Standard Adam (ortho applied via loss function if enabled)
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_norm),
            optax.adam(learning_rate, eps=eps),
        )

    return optimizer


def make_optimizer_with_mask(
    learning_rate: Union[float, Callable],
    config: DictConfig,
    should_transform_fn: Callable[[int], bool],
    max_grad_norm: Optional[float] = None,
    eps: float = 1e-5,
) -> optax.GradientTransformation:
    """Create a conditionally masked optimizer (e.g., for delayed policy updates in TD3).

    Wraps make_optimizer with optax.conditionally_mask for cases where
    updates should only happen on certain steps (e.g., TD3 delayed actor update).

    Args:
        learning_rate: Learning rate (scalar or schedule)
        config: DictConfig with system settings
        should_transform_fn: Function(step_count) -> bool indicating when to apply updates
        max_grad_norm: Override for gradient clipping norm
        eps: Adam epsilon

    Returns:
        optax.GradientTransformation with conditional masking

    Example:
        # TD3 delayed policy update (every 2 steps):
        def delayed_policy_update(step_count):
            return step_count % config.system.policy_frequency == 0

        actor_optim = make_optimizer_with_mask(
            actor_lr, config, delayed_policy_update
        )
    """
    base_optimizer = make_optimizer(learning_rate, config, max_grad_norm, eps)
    return optax.conditionally_mask(base_optimizer, should_transform_fn=should_transform_fn)
