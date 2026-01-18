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

from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


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
        elif isinstance(node, jnp.ndarray) and node.ndim == 2:
            # This is likely a kernel (2D weight matrix)
            if path.endswith("kernel"):
                if layer_filter is None or layer_filter(path):
                    kernels[path] = node

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


def aggregate_spectral_diagnostics(diagnostics: Dict[str, Dict]) -> Dict[str, jnp.ndarray]:
    """Flatten spectral diagnostics into a dict suitable for logging.

    Args:
        diagnostics: Output from compute_spectral_diagnostics

    Returns:
        Flat dict with keys like 'spectral/max_singular_value', etc.
    """
    if "_aggregate" not in diagnostics:
        return {}

    agg = diagnostics["_aggregate"]
    return {
        "spectral/max_singular_value": agg["max_singular_value"],
        "spectral/min_singular_value": agg["min_singular_value"],
        "spectral/max_condition_number": agg["max_condition_number"],
        "spectral/mean_condition_number": agg["mean_condition_number"],
    }
