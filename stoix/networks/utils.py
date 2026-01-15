from functools import partial
from typing import Callable, Dict

import chex
import jax.numpy as jnp
from flax import linen as nn


def groupsort(x: chex.Array, group_size: int = 2) -> chex.Array:
    """GroupSort activation function (Anil et al., ICML 2019).

    A 1-Lipschitz activation that preserves gradient norm by sorting features
    within groups. This enables training very deep networks without vanishing
    or exploding gradients.

    Args:
        x: Input tensor of shape (..., features)
        group_size: Number of elements per group to sort together.
            Default is 2 (also called "full sort" or "maxmin").

    Returns:
        Sorted tensor of same shape where each group of `group_size` features
        is sorted in ascending order.

    Raises:
        AssertionError: If features dimension is not divisible by group_size.

    Example:
        >>> x = jnp.array([3.0, 1.0, 4.0, 2.0])
        >>> groupsort(x, group_size=2)
        Array([1., 3., 2., 4.], dtype=float32)  # [min(3,1), max(3,1), min(4,2), max(4,2)]
    """
    *batch_dims, features = x.shape
    assert (
        features % group_size == 0
    ), f"Features {features} must be divisible by group_size {group_size}"

    num_groups = features // group_size
    x_grouped = x.reshape(*batch_dims, num_groups, group_size)
    x_sorted = jnp.sort(x_grouped, axis=-1)
    return x_sorted.reshape(*batch_dims, features)


def parse_activation_fn(activation_fn_name: str) -> Callable[[chex.Array], chex.Array]:
    """Get the activation function."""
    activation_fns: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "relu": nn.relu,
        "tanh": nn.tanh,
        "silu": nn.silu,
        "elu": nn.elu,
        "gelu": nn.gelu,
        "sigmoid": nn.sigmoid,
        "softplus": nn.softplus,
        "swish": nn.swish,
        "identity": lambda x: x,
        "none": lambda x: x,
        "normalise": nn.standardize,
        "softmax": nn.softmax,
        "log_softmax": nn.log_softmax,
        "log_sigmoid": nn.log_sigmoid,
        "groupsort": groupsort,  # group_size=2 (maxmin)
        "groupsort4": partial(groupsort, group_size=4),
        "groupsort8": partial(groupsort, group_size=8),
    }
    return activation_fns[activation_fn_name]


def parse_rnn_cell(rnn_cell_name: str) -> nn.RNNCellBase:
    """Get the rnn cell."""
    rnn_cells: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "lstm": nn.LSTMCell,
        "optimised_lstm": nn.OptimizedLSTMCell,
        "gru": nn.GRUCell,
        "mgu": nn.MGUCell,
        "simple": nn.SimpleCell,
    }
    return rnn_cells[rnn_cell_name]
