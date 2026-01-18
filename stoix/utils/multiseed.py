"""Multi-seed training utilities following PureJaxRL-style functional design.

This module provides utilities for running multiple seeds in parallel using vmap.
The key insight is that `train(rng)` should be a pure function that can be vmapped.

Usage:
    # Single seed (existing pattern)
    train_fn = make_train(config)
    result = jax.jit(train_fn)(jax.random.PRNGKey(42))

    # Multi-seed (new pattern - just vmap!)
    from stoix.utils.multiseed import run_multiseed
    results = run_multiseed(train_fn, seeds=[42, 43, 44])

    # Or manually:
    rngs = jax.vmap(jax.random.PRNGKey)(jnp.array([42, 43, 44]))
    results = jax.vmap(train_fn)(rngs)
"""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


class TrainOutput(NamedTuple):
    """Output from a pure training function.

    This is the return type of `train(rng)` in PureJaxRL-style design.
    All metrics are accumulated during training and returned at the end.
    """
    params: FrozenDict  # Final trained parameters
    metrics: Dict[str, chex.Array]  # All accumulated metrics (shape: [num_steps, ...])
    final_state: Optional[Any] = None  # Optional: full learner state for checkpointing


class MultiseedOutput(NamedTuple):
    """Output from multi-seed training.

    Contains per-seed results with an additional seed dimension.
    """
    params: FrozenDict  # Shape: [num_seeds, ...]
    metrics: Dict[str, chex.Array]  # Shape: [num_seeds, num_steps, ...]
    seed_values: List[int]  # The actual seed values used


def run_multiseed(
    train_fn: Callable[[chex.PRNGKey], TrainOutput],
    seeds: Union[List[int], chex.Array],
    jit: bool = True,
) -> MultiseedOutput:
    """Run training with multiple seeds in parallel using vmap.

    Args:
        train_fn: A pure training function with signature `train(rng) -> TrainOutput`
        seeds: List of seed values or array of seeds
        jit: Whether to JIT-compile the vmapped training (default: True)

    Returns:
        MultiseedOutput with per-seed params and metrics

    Example:
        train_fn = make_train(config)
        results = run_multiseed(train_fn, seeds=[42, 43, 44])

        # Access per-seed results:
        for i, seed in enumerate(results.seed_values):
            params_i = jax.tree_util.tree_map(lambda x: x[i], results.params)
            final_return_i = results.metrics['episode_return'][i, -1]
    """
    seed_array = jnp.array(seeds)
    rngs = jax.vmap(jax.random.PRNGKey)(seed_array)

    vmapped_train = jax.vmap(train_fn)
    if jit:
        vmapped_train = jax.jit(vmapped_train)

    results: TrainOutput = vmapped_train(rngs)

    return MultiseedOutput(
        params=results.params,
        metrics=results.metrics,
        seed_values=list(seeds) if isinstance(seeds, list) else seeds.tolist(),
    )


def aggregate_metrics(
    metrics: Dict[str, chex.Array],
    reduction: str = "mean",
) -> Dict[str, chex.Array]:
    """Aggregate metrics across seeds.

    Args:
        metrics: Dict with values of shape [num_seeds, ...]
        reduction: One of "mean", "std", "min", "max"

    Returns:
        Dict with values reduced across first (seed) dimension
    """
    reduce_fn = {
        "mean": jnp.mean,
        "std": jnp.std,
        "min": jnp.min,
        "max": jnp.max,
    }[reduction]

    return {k: reduce_fn(v, axis=0) for k, v in metrics.items()}


def log_multiseed_metrics(
    logger,
    results: MultiseedOutput,
    timestep: int,
    eval_step: int,
    log_event,
) -> None:
    """Log per-seed and aggregate metrics from multi-seed training.

    Args:
        logger: StoixLogger instance
        results: MultiseedOutput from run_multiseed
        timestep: Current timestep for logging
        eval_step: Current evaluation step
        log_event: LogEvent type (ACT, EVAL, TRAIN, etc.)
    """
    num_seeds = len(results.seed_values)

    # Log per-seed metrics
    for i, seed_val in enumerate(results.seed_values):
        seed_metrics = {}
        for k, v in results.metrics.items():
            if v.ndim >= 1:
                # Get final value for this seed
                seed_metrics[f"seed_{seed_val}/{k}"] = float(v[i, -1] if v.ndim > 1 else v[i])
        logger.log(seed_metrics, timestep, eval_step, log_event)

    # Log aggregate metrics (mean ± std)
    aggregate_metrics_dict = {}
    for k, v in results.metrics.items():
        if v.ndim >= 1:
            # Get final values across seeds
            final_vals = v[:, -1] if v.ndim > 1 else v
            aggregate_metrics_dict[k] = float(jnp.mean(final_vals))
            if num_seeds > 1:
                aggregate_metrics_dict[f"{k}_std"] = float(jnp.std(final_vals))
    logger.log(aggregate_metrics_dict, timestep, eval_step, log_event)
