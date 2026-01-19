"""Orthogonalization utilities for isometric neural networks.

This module provides functions to maintain approximate orthonormality of
weight matrices, which is critical for:
- Stable gradient flow in very deep networks
- Maintaining bounded Lipschitz constants
- Preventing plasticity collapse

Designed for use with any RL algorithm (DQN, TD3, PPO, etc.).

References:
- Anil et al., "Sorting out Lipschitz function approximation", ICML 2019
- Björck & Bowie, "An iterative algorithm for computing the best estimate
  of an orthogonal matrix", SIAM J. Numer. Anal. 1971
"""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict


def _is_array_like(node) -> bool:
    """Check if node is an array-like object (JAX array, numpy array, etc.)."""
    return hasattr(node, 'ndim') and hasattr(node, 'shape')


def adaptive_gram_loss(W: jnp.ndarray) -> Tuple[jnp.ndarray, Dict]:
    """Compute adaptive Gram regularization loss for a single weight matrix.

    For a weight matrix W of shape (n_in, n_out):
    - Wide (n_in < n_out): Target WW^T = (n_out/n_in) * I
      This ensures row orthogonality with scaled identity
    - Tall/Square (n_in >= n_out): Target W^TW = I
      This ensures column orthonormality

    Args:
        W: Weight matrix of shape (n_in, n_out)

    Returns:
        loss: Squared Frobenius norm of (Gram - Target)
        info: Dict with 'gram_deviation' and 'is_wide' for logging
    """
    n_in, n_out = W.shape
    is_wide = n_in < n_out

    if is_wide:
        # Wide: WW^T should be (n_out/n_in) * I
        gram = W @ W.T  # (n_in, n_in)
        scale = n_out / n_in
        target = scale * jnp.eye(n_in)
    else:
        # Tall/Square: W^TW should be I
        gram = W.T @ W  # (n_out, n_out)
        target = jnp.eye(n_out)

    diff = gram - target
    loss = jnp.sum(diff**2)  # Frobenius norm squared

    return loss, {"gram_deviation": jnp.sqrt(loss), "is_wide": is_wide}


def get_dense_kernels(
    params: FrozenDict,
    layer_filter: Optional[Callable[[str], bool]] = None,
) -> Dict[str, jnp.ndarray]:
    """Extract all Dense layer kernels from parameter tree.

    Traverses the nested parameter tree and finds all 'kernel' arrays
    that are 2D (typical Dense layer weights).

    Args:
        params: Flax parameter tree (FrozenDict)
        layer_filter: Optional function(layer_path) -> bool to include layer.
            If None, includes all Dense layers.

    Returns:
        Dict mapping layer path (e.g., 'torso/Dense_0/kernel') to weight matrix.
    """
    kernels = {}

    def _traverse(node, path: str = ""):
        if isinstance(node, (dict, FrozenDict)):
            for key, value in node.items():
                new_path = f"{path}/{key}" if path else key
                _traverse(value, new_path)
        elif _is_array_like(node) and path.endswith("kernel"):
            # This is likely a kernel (weight matrix)
            # Works with JAX arrays, numpy arrays, and other array-likes
            # Squeeze out leading singleton dimensions (e.g., from unreplicate)
            arr = jnp.asarray(node)
            while arr.ndim > 2 and arr.shape[0] == 1:
                arr = arr.squeeze(axis=0)
            if arr.ndim == 2:
                if layer_filter is None or layer_filter(path):
                    kernels[path] = arr

    _traverse(params)
    return kernels


def _find_output_layer_path(kernel_paths: List[str]) -> Optional[str]:
    """Heuristic to identify the output layer path.

    Looks for common output layer naming patterns.

    Args:
        kernel_paths: List of kernel paths like 'torso/Dense_0/kernel'

    Returns:
        Path of the likely output layer, or None if not found.
    """
    output_patterns = ["output", "final", "head", "q_network", "action"]

    for path in kernel_paths:
        path_lower = path.lower()
        for pattern in output_patterns:
            if pattern in path_lower:
                return path

    # If no pattern found, return the last layer by typical naming (Dense_N with highest N)
    dense_layers = [p for p in kernel_paths if "Dense_" in p or "dense_" in p]
    if dense_layers:
        # Sort by layer number
        def get_layer_num(p):
            for part in p.split("/"):
                if part.lower().startswith("dense_"):
                    try:
                        return int(part.split("_")[-1])
                    except ValueError:
                        pass
            return -1

        sorted_layers = sorted(dense_layers, key=get_layer_num, reverse=True)
        if sorted_layers:
            return sorted_layers[0]

    return None


def compute_gram_regularization_loss(
    params: FrozenDict,
    exclude_output: bool = True,
    layer_filter: Optional[Callable[[str], bool]] = None,
) -> Tuple[jnp.ndarray, Dict]:
    """Compute total Gram regularization loss over all Dense layers.

    Traverses the parameter tree and applies adaptive_gram_loss to
    all Dense layer kernels.

    Args:
        params: Flax parameter tree (FrozenDict)
        exclude_output: If True, skip the final layer (heuristic: layer with
            'output'/'final'/'head' in name, or highest numbered Dense layer)
        layer_filter: Optional function(layer_path) -> bool to include layer.
            Applied in addition to exclude_output logic.

    Returns:
        total_loss: Sum of Gram losses across layers
        info: Dict with per-layer diagnostics and aggregates:
            - 'total_gram_deviation': sqrt of sum of squared deviations
            - 'num_layers': number of layers with Gram loss applied
            - 'mean_gram_deviation': average deviation per layer
            - 'layer_deviations': dict mapping layer path to its deviation
    """
    kernels = get_dense_kernels(params, layer_filter)

    if not kernels:
        return jnp.array(0.0), {
            "total_gram_deviation": jnp.array(0.0),
            "num_layers": 0,
            "mean_gram_deviation": jnp.array(0.0),
            "layer_deviations": {},
        }

    # Find output layer to potentially exclude
    output_layer_path = None
    if exclude_output:
        output_layer_path = _find_output_layer_path(list(kernels.keys()))

    total_loss = jnp.array(0.0)
    layer_deviations = {}
    num_layers = 0

    for path, kernel in kernels.items():
        # Skip output layer if requested
        if exclude_output and path == output_layer_path:
            continue

        loss, info = adaptive_gram_loss(kernel)
        total_loss = total_loss + loss
        layer_deviations[path] = info["gram_deviation"]
        num_layers += 1

    if num_layers > 0:
        mean_deviation = jnp.sqrt(total_loss) / num_layers
    else:
        mean_deviation = jnp.array(0.0)

    return total_loss, {
        "total_gram_deviation": jnp.sqrt(total_loss),
        "num_layers": num_layers,
        "mean_gram_deviation": mean_deviation,
        "layer_deviations": layer_deviations,
    }


def compute_network_lipschitz_bound(
    params: FrozenDict,
    layer_filter: Optional[Callable[[str], bool]] = None,
) -> jnp.ndarray:
    """Compute upper bound on network Lipschitz constant.

    For a network f = W_L σ ... σ W_1, the Lipschitz bound is:
    Lip(f) ≤ ∏_{i=1}^{L} σ_max(W_i)

    With GroupSort (1-Lipschitz activation), this is exact for linear paths.

    References:
    - Anil et al., "Sorting out Lipschitz function approximation", ICML 2019
    - Gouk et al., "Regularisation of Neural Networks by Enforcing Lipschitz Continuity", 2021

    Args:
        params: Flax parameter tree (FrozenDict)
        layer_filter: Optional function(layer_path) -> bool to include layer.

    Returns:
        lipschitz_bound: Product of max singular values across all layers.
    """
    kernels = get_dense_kernels(params, layer_filter)
    lipschitz_bound = jnp.array(1.0)
    for kernel in kernels.values():
        s_max = jnp.linalg.svdvals(kernel)[0]
        lipschitz_bound = lipschitz_bound * s_max
    return lipschitz_bound


def compute_spectral_diagnostics(
    params: FrozenDict,
    compute_full_svd: bool = False,
    layer_filter: Optional[Callable[[str], bool]] = None,
) -> Dict[str, Dict]:
    """Compute spectral properties of weight matrices for monitoring.

    This is expensive (requires SVD) so should only be called periodically
    (e.g., every 1000 updates or at evaluation time).

    Args:
        params: Flax parameter tree
        compute_full_svd: If True, compute all singular values.
            If False, only compute extremal values (faster).
        layer_filter: Optional function to filter which layers to analyze

    Returns:
        Dict mapping layer_path to:
            - 's_max': Largest singular value
            - 's_min': Smallest singular value
            - 'condition_number': s_max / s_min
            - 'weight_norm': Frobenius norm
            - 'effective_rank': (sum of s_i)^2 / (sum of s_i^2) (if full SVD)
    """
    kernels = get_dense_kernels(params, layer_filter)
    diagnostics = {}

    for path, kernel in kernels.items():
        # Compute SVD
        # Note: For very large matrices, we could use randomized SVD
        # but for typical RL network sizes, full SVD is fine
        if compute_full_svd:
            s = jnp.linalg.svdvals(kernel)
            s_max = s[0]
            s_min = s[-1]
            # Effective rank: (sum s_i)^2 / (sum s_i^2)
            effective_rank = (jnp.sum(s) ** 2) / (jnp.sum(s**2) + 1e-10)
        else:
            # Just compute extremal singular values
            s = jnp.linalg.svdvals(kernel)
            s_max = s[0]
            s_min = s[-1]
            effective_rank = None

        # Frobenius norm
        weight_norm = jnp.linalg.norm(kernel, "fro")

        # Condition number (avoid division by zero)
        condition_number = s_max / (s_min + 1e-10)

        layer_info = {
            "s_max": s_max,
            "s_min": s_min,
            "condition_number": condition_number,
            "weight_norm": weight_norm,
        }
        if effective_rank is not None:
            layer_info["effective_rank"] = effective_rank

        diagnostics[path] = layer_info

    # Also compute aggregate statistics
    if diagnostics:
        all_s_max = jnp.array([d["s_max"] for d in diagnostics.values()])
        all_s_min = jnp.array([d["s_min"] for d in diagnostics.values()])
        all_cond = jnp.array([d["condition_number"] for d in diagnostics.values()])

        diagnostics["_aggregate"] = {
            "max_singular_value": jnp.max(all_s_max),
            "min_singular_value": jnp.min(all_s_min),
            "max_condition_number": jnp.max(all_cond),
            "mean_condition_number": jnp.mean(all_cond),
        }

    return diagnostics


def aggregate_spectral_diagnostics(
    diagnostics: Dict[str, Dict],
    params: Optional[FrozenDict] = None,
    include_per_layer: bool = True,
) -> Dict[str, jnp.ndarray]:
    """Flatten spectral diagnostics into a dict suitable for logging.

    Args:
        diagnostics: Output from compute_spectral_diagnostics
        params: Optional Flax parameter tree for computing Lipschitz bound.
            If provided, computes and includes network_lipschitz_bound.
        include_per_layer: If True, include per-layer metrics (s_max, s_min,
            condition_number, weight_norm for each layer).

    Returns:
        Flat dict with keys like 'spectral/max_singular_value', etc.
        Per-layer keys use format: 'spectral/layer_{i}/metric_name'
    """
    if "_aggregate" not in diagnostics:
        return {}

    agg = diagnostics["_aggregate"]
    result = {
        "spectral/max_singular_value": agg["max_singular_value"],
        "spectral/min_singular_value": agg["min_singular_value"],
        "spectral/max_condition_number": agg["max_condition_number"],
        "spectral/mean_condition_number": agg["mean_condition_number"],
    }

    # Compute network Lipschitz bound if params provided
    if params is not None:
        lipschitz_bound = compute_network_lipschitz_bound(params)
        result["spectral/network_lipschitz_bound"] = lipschitz_bound
        # Log scale for deep networks (can be very large)
        result["spectral/log_lipschitz_bound"] = jnp.log(lipschitz_bound + 1e-10)

    # Include per-layer metrics
    if include_per_layer:
        # Sort layer paths to ensure consistent ordering
        layer_paths = sorted([k for k in diagnostics.keys() if k != "_aggregate"])
        for i, path in enumerate(layer_paths):
            layer_data = diagnostics[path]
            prefix = f"spectral/layer_{i}"
            result[f"{prefix}/s_max"] = layer_data["s_max"]
            result[f"{prefix}/s_min"] = layer_data["s_min"]
            result[f"{prefix}/condition_number"] = layer_data["condition_number"]
            result[f"{prefix}/weight_norm"] = layer_data["weight_norm"]
            if "effective_rank" in layer_data:
                result[f"{prefix}/effective_rank"] = layer_data["effective_rank"]

    return result


# =============================================================================
# AdamO: Adam with Decoupled Orthonormalization
# =============================================================================


def _compute_ortho_grad(W: jnp.ndarray) -> jnp.ndarray:
    """Compute gradient of orthonormalization penalty for a single weight matrix.

    For a weight matrix W of shape (n_in, n_out):
    - Wide (n_in < n_out): Gradient of ||WW^T - scale*I||^2
    - Tall/Square (n_in >= n_out): Gradient of ||W^TW - I||^2

    The gradient is: d/dW ||Gram - Target||^2 = 2 * (W @ W.T @ W - scale * W) for wide
                                               = 2 * (W @ W.T @ W - W) for tall

    Args:
        W: Weight matrix of shape (n_in, n_out)

    Returns:
        Gradient with same shape as W
    """
    n_in, n_out = W.shape
    is_wide = n_in < n_out

    if is_wide:
        # Wide: WW^T should be (n_out/n_in) * I
        # Loss = ||WW^T - scale*I||^2_F
        # Gradient = 2 * (WW^T @ W - scale * W)
        scale = n_out / n_in
        gram = W @ W.T  # (n_in, n_in)
        grad = 2.0 * (gram @ W - scale * W)
    else:
        # Tall/Square: W^TW should be I
        # Loss = ||W^TW - I||^2_F
        # Gradient = 2 * (W @ W^TW - W)
        gram = W.T @ W  # (n_out, n_out)
        grad = 2.0 * (W @ gram - W)

    return grad


class AdamOState(NamedTuple):
    """State for AdamO optimizer."""
    adam_state: optax.OptState
    output_layer_path: Optional[str]


def adamo(
    learning_rate: optax.ScalarOrSchedule = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    ortho_coeff: float = 1e-3,
    exclude_output: bool = True,
    eps_root: float = 0.0,
) -> optax.GradientTransformation:
    """Adam with decoupled orthonormalization regularization.

    This optimizer applies orthonormalization regularization decoupled from the
    loss function, similar to how AdamW decouples weight decay from L2 regularization.

    The key insight is that loss-based ortho regularization gets scaled by Adam's
    adaptive learning rate (1/sqrt(v_t)), which varies per-parameter. Decoupled
    ortho applies a consistent regularization strength independent of gradient history.

    The orthonormalization update is:
        params -= lr * ortho_coeff * ortho_grad(W)

    where ortho_grad(W) = 2 * (W @ W.T @ W - scale * W) for wide matrices
                        = 2 * (W @ W.T @ W - W) for tall matrices

    Args:
        learning_rate: Adam learning rate (scalar or schedule)
        b1: Adam first moment decay rate
        b2: Adam second moment decay rate
        eps: Adam epsilon for numerical stability
        ortho_coeff: Orthonormalization strength (decoupled, not scaled by adaptive lr).
                     Recommended starting value: 1e-3
        exclude_output: If True, skip output layers (detected by heuristic).
                        Recommended True since output layer must fit arbitrary value ranges.
        eps_root: Adam eps_root parameter

    Returns:
        A GradientTransformation that can be used with optax.chain() or standalone.

    Example:
        # Standalone usage
        optimizer = adamo(learning_rate=1e-3, ortho_coeff=1e-3)

        # With gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            adamo(learning_rate=1e-3, ortho_coeff=1e-3),
        )
    """

    # Create base Adam optimizer
    adam_opt = optax.adam(
        learning_rate=learning_rate,
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
    )

    def init_fn(params: Any) -> AdamOState:
        """Initialize optimizer state."""
        adam_state = adam_opt.init(params)

        # Find output layer path if we need to exclude it
        output_layer_path = None
        if exclude_output:
            kernels = get_dense_kernels(params)
            if kernels:
                output_layer_path = _find_output_layer_path(list(kernels.keys()))

        return AdamOState(adam_state=adam_state, output_layer_path=output_layer_path)

    def update_fn(
        updates: Any,
        state: AdamOState,
        params: Optional[Any] = None,
    ) -> Tuple[Any, AdamOState]:
        """Apply Adam update plus decoupled orthonormalization."""
        # First apply Adam update
        adam_updates, new_adam_state = adam_opt.update(updates, state.adam_state, params)

        if params is None or ortho_coeff == 0.0:
            # No params provided or ortho disabled, skip ortho step
            return adam_updates, AdamOState(
                adam_state=new_adam_state,
                output_layer_path=state.output_layer_path,
            )

        # Get learning rate for current step (handles schedules)
        if callable(learning_rate):
            # Extract step count from Adam state
            # Adam state is a tuple of (ScaleByAdamState, ScaleByScheduleState) or similar
            # The count is usually in the first state
            count = getattr(new_adam_state[0], 'count', 0)
            lr = learning_rate(count)
        else:
            lr = learning_rate

        # Apply decoupled orthonormalization to 2D weight matrices
        def add_ortho_update(path: str, update: Any, param: Any) -> Any:
            """Add orthonormalization gradient to update for eligible parameters."""
            # Only apply to 2D kernel parameters (not biases)
            if not path.endswith("kernel"):
                return update

            # Handle array-like check
            if not _is_array_like(param):
                return update

            # Squeeze singleton dimensions
            param_arr = jnp.asarray(param)
            while param_arr.ndim > 2 and param_arr.shape[0] == 1:
                param_arr = param_arr.squeeze(axis=0)

            if param_arr.ndim != 2:
                return update

            # Skip output layer if configured
            if exclude_output and state.output_layer_path is not None:
                if path == state.output_layer_path:
                    return update

            # Compute ortho gradient and add to update
            # Note: updates are subtracted from params, so we add positive ortho_grad
            ortho_grad = _compute_ortho_grad(param_arr)

            # Scale by lr * ortho_coeff (decoupled from Adam's adaptive scaling)
            scaled_ortho = lr * ortho_coeff * ortho_grad

            # Reshape back if needed
            if update.shape != scaled_ortho.shape:
                scaled_ortho = scaled_ortho.reshape(update.shape)

            return update + scaled_ortho

        # Traverse updates and params together, adding ortho updates
        def traverse_and_update(updates_node, params_node, path=""):
            if isinstance(updates_node, (dict, FrozenDict)):
                result = {}
                for key in updates_node.keys():
                    new_path = f"{path}/{key}" if path else key
                    result[key] = traverse_and_update(
                        updates_node[key], params_node[key], new_path
                    )
                return type(updates_node)(result) if isinstance(updates_node, FrozenDict) else result
            elif _is_array_like(updates_node):
                return add_ortho_update(path, updates_node, params_node)
            else:
                return updates_node

        new_updates = traverse_and_update(adam_updates, params)

        return new_updates, AdamOState(
            adam_state=new_adam_state,
            output_layer_path=state.output_layer_path,
        )

    return optax.GradientTransformation(init_fn, update_fn)
