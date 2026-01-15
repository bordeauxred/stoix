"""Tests for GroupSort activation function.

Tests validate:
- Basic sorting behavior within groups
- Different group sizes (2, 4, 8)
- Batch dimension handling
- Gradient norm preservation (1-Lipschitz property)
- JIT compatibility
- Edge cases and error conditions
- Integration with parse_activation_fn registry
"""

from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from stoix.networks.utils import groupsort, parse_activation_fn


class GroupSortTest(parameterized.TestCase):
    """Test suite for GroupSort activation function."""

    # ========== Basic Functionality Tests ==========

    def test_basic_sorting_group_size_2(self) -> None:
        """Test basic sorting with group_size=2 (maxmin activation)."""
        x = jnp.array([3.0, 1.0, 4.0, 2.0])
        result = groupsort(x, group_size=2)

        # Group [3, 1] -> [1, 3] (sorted ascending)
        # Group [4, 2] -> [2, 4] (sorted ascending)
        expected = jnp.array([1.0, 3.0, 2.0, 4.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_basic_sorting_group_size_4(self) -> None:
        """Test sorting with group_size=4."""
        x = jnp.array([4.0, 2.0, 3.0, 1.0, 8.0, 6.0, 7.0, 5.0])
        result = groupsort(x, group_size=4)

        # Group [4, 2, 3, 1] -> [1, 2, 3, 4]
        # Group [8, 6, 7, 5] -> [5, 6, 7, 8]
        expected = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_already_sorted_input(self) -> None:
        """Test that already sorted input remains unchanged."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = groupsort(x, group_size=2)
        expected = jnp.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_reverse_sorted_input(self) -> None:
        """Test that reverse sorted input gets properly sorted."""
        x = jnp.array([2.0, 1.0, 4.0, 3.0])
        result = groupsort(x, group_size=2)
        expected = jnp.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_negative_values(self) -> None:
        """Test handling of negative values."""
        x = jnp.array([1.0, -1.0, -3.0, 2.0])
        result = groupsort(x, group_size=2)
        # Group [1, -1] -> [-1, 1]
        # Group [-3, 2] -> [-3, 2]
        expected = jnp.array([-1.0, 1.0, -3.0, 2.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_equal_values_in_group(self) -> None:
        """Test handling of equal values within a group."""
        x = jnp.array([2.0, 2.0, 1.0, 3.0])
        result = groupsort(x, group_size=2)
        expected = jnp.array([2.0, 2.0, 1.0, 3.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    # ========== Batch Dimension Tests ==========

    @chex.all_variants()
    def test_2d_batch(self) -> None:
        """Test with 2D input (batch, features)."""
        x = jnp.array([[3.0, 1.0, 4.0, 2.0], [8.0, 5.0, 7.0, 6.0]])
        # Bind group_size before variant to avoid tracing static args
        groupsort_fn = self.variant(partial(groupsort, group_size=2))
        result = groupsort_fn(x)

        expected = jnp.array([[1.0, 3.0, 2.0, 4.0], [5.0, 8.0, 6.0, 7.0]])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    @chex.all_variants()
    def test_3d_batch(self) -> None:
        """Test with 3D input (batch1, batch2, features)."""
        x = jnp.array([[[3.0, 1.0], [4.0, 2.0]], [[8.0, 5.0], [7.0, 6.0]]])
        groupsort_fn = self.variant(partial(groupsort, group_size=2))
        result = groupsort_fn(x)

        expected = jnp.array([[[1.0, 3.0], [2.0, 4.0]], [[5.0, 8.0], [6.0, 7.0]]])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    @chex.all_variants()
    def test_4d_batch(self) -> None:
        """Test with 4D input typical of RL (timesteps, envs, agents, features)."""
        shape = (2, 3, 4, 8)  # 8 features, group_size=2 -> 4 groups
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, shape)

        groupsort_fn = self.variant(partial(groupsort, group_size=2))
        result = groupsort_fn(x)

        # Verify shape preserved
        self.assertEqual(result.shape, shape)

        # Verify each group is sorted
        result_grouped = result.reshape(*shape[:-1], 4, 2)
        self.assertTrue(jnp.all(result_grouped[..., 0] <= result_grouped[..., 1]))

    # ========== Gradient Property Tests ==========

    def test_gradient_norm_preservation(self) -> None:
        """Test that gradient norm is approximately preserved (1-Lipschitz).

        The Jacobian of GroupSort is a permutation matrix, so gradient norms
        should be exactly preserved.
        """
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (8,))

        # Compute Jacobian
        jacobian = jax.jacobian(lambda z: groupsort(z, group_size=2))(x)

        # Jacobian should be orthogonal (permutation matrix has orthonormal rows/cols)
        # ||J||_2 = 1 for permutation matrices
        singular_values = jnp.linalg.svd(jacobian, compute_uv=False)

        # All singular values should be 1 (up to numerical precision)
        np.testing.assert_allclose(singular_values, jnp.ones_like(singular_values), atol=1e-5)

    def test_gradient_flow_through_network(self) -> None:
        """Test gradient flow through multiple GroupSort layers."""
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (16,))

        def deep_network(x: jnp.ndarray) -> jnp.ndarray:
            """Simple deep network with GroupSort activations."""
            for _ in range(10):  # 10 layers
                x = groupsort(x, group_size=2)
            return x.sum()

        # Compute gradient
        grad = jax.grad(deep_network)(x)

        # Gradient should not vanish or explode
        grad_norm = jnp.linalg.norm(grad)
        self.assertGreater(grad_norm, 0.1, "Gradient vanished")
        self.assertLess(grad_norm, 10.0, "Gradient exploded")

    @parameterized.named_parameters(
        ("group_size_2", 2),
        ("group_size_4", 4),
        ("group_size_8", 8),
    )
    def test_lipschitz_constant(self, group_size: int) -> None:
        """Test that GroupSort is 1-Lipschitz: ||f(x) - f(y)|| <= ||x - y||."""
        key = jax.random.PRNGKey(2)
        features = 16
        num_tests = 100

        for i in range(num_tests):
            key, k1, k2 = jax.random.split(key, 3)
            x = jax.random.normal(k1, (features,))
            y = jax.random.normal(k2, (features,))

            fx = groupsort(x, group_size=group_size)
            fy = groupsort(y, group_size=group_size)

            input_dist = jnp.linalg.norm(x - y)
            output_dist = jnp.linalg.norm(fx - fy)

            # 1-Lipschitz: output distance <= input distance
            self.assertLessEqual(
                output_dist,
                input_dist + 1e-6,  # Small tolerance for numerical precision
                f"Lipschitz violation at iteration {i}",
            )

    # ========== JIT and Vmap Compatibility Tests ==========

    def test_jit_compilation(self) -> None:
        """Test that GroupSort works with JIT compilation."""
        x = jnp.array([3.0, 1.0, 4.0, 2.0])

        # Bind group_size before JIT (static argument)
        jit_groupsort = jax.jit(partial(groupsort, group_size=2))
        result = jit_groupsort(x)

        expected = jnp.array([1.0, 3.0, 2.0, 4.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_vmap_over_batch(self) -> None:
        """Test that GroupSort works with vmap."""
        x = jnp.array([[3.0, 1.0], [4.0, 2.0], [8.0, 5.0]])

        vmap_groupsort = jax.vmap(lambda z: groupsort(z, group_size=2))
        result = vmap_groupsort(x)

        expected = jnp.array([[1.0, 3.0], [2.0, 4.0], [5.0, 8.0]])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_jit_vmap_combination(self) -> None:
        """Test JIT + vmap combination typical of RL training."""
        key = jax.random.PRNGKey(3)
        x = jax.random.normal(key, (32, 64))  # 32 envs, 64 features

        @jax.jit
        def batched_groupsort(x: jnp.ndarray) -> jnp.ndarray:
            return jax.vmap(lambda z: groupsort(z, group_size=2))(x)

        result = batched_groupsort(x)
        self.assertEqual(result.shape, (32, 64))

    # ========== Registry Integration Tests ==========

    def test_parse_activation_fn_groupsort(self) -> None:
        """Test that groupsort is accessible via parse_activation_fn."""
        activation = parse_activation_fn("groupsort")
        x = jnp.array([3.0, 1.0, 4.0, 2.0])
        result = activation(x)

        expected = jnp.array([1.0, 3.0, 2.0, 4.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_parse_activation_fn_groupsort4(self) -> None:
        """Test that groupsort4 (group_size=4) is accessible via parse_activation_fn."""
        activation = parse_activation_fn("groupsort4")
        x = jnp.array([4.0, 2.0, 3.0, 1.0])
        result = activation(x)

        expected = jnp.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_parse_activation_fn_groupsort8(self) -> None:
        """Test that groupsort8 (group_size=8) is accessible via parse_activation_fn."""
        activation = parse_activation_fn("groupsort8")
        x = jnp.array([8.0, 4.0, 6.0, 2.0, 7.0, 3.0, 5.0, 1.0])
        result = activation(x)

        expected = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    # ========== Edge Case Tests ==========

    def test_group_size_equals_features(self) -> None:
        """Test when group_size equals total features (full sort)."""
        x = jnp.array([4.0, 2.0, 3.0, 1.0])
        result = groupsort(x, group_size=4)

        expected = jnp.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_single_element_groups(self) -> None:
        """Test with group_size=1 (identity operation)."""
        x = jnp.array([4.0, 2.0, 3.0, 1.0])
        result = groupsort(x, group_size=1)

        # group_size=1 means each element is its own group, so no change
        np.testing.assert_allclose(result, x, atol=1e-6)

    def test_large_features(self) -> None:
        """Test with large feature dimension typical of RL networks."""
        key = jax.random.PRNGKey(4)
        x = jax.random.normal(key, (256,))

        result = groupsort(x, group_size=2)
        self.assertEqual(result.shape, (256,))

        # Verify sorting within groups
        result_grouped = result.reshape(128, 2)
        self.assertTrue(jnp.all(result_grouped[:, 0] <= result_grouped[:, 1]))

    def test_zeros_input(self) -> None:
        """Test with all-zeros input."""
        x = jnp.zeros(8)
        result = groupsort(x, group_size=2)
        np.testing.assert_allclose(result, x, atol=1e-6)

    def test_inf_values(self) -> None:
        """Test handling of infinity values."""
        x = jnp.array([jnp.inf, 1.0, -jnp.inf, 0.0])
        result = groupsort(x, group_size=2)

        # Group [inf, 1] -> [1, inf]
        # Group [-inf, 0] -> [-inf, 0]
        expected = jnp.array([1.0, jnp.inf, -jnp.inf, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    # ========== Error Handling Tests ==========

    def test_invalid_group_size_error(self) -> None:
        """Test that invalid group_size raises assertion error."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 features

        with self.assertRaises(AssertionError) as context:
            groupsort(x, group_size=2)  # 5 not divisible by 2

        self.assertIn("divisible by group_size", str(context.exception))

    def test_invalid_group_size_3(self) -> None:
        """Test with features not divisible by group_size=3."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0])  # 4 features

        with self.assertRaises(AssertionError):
            groupsort(x, group_size=3)  # 4 not divisible by 3

    # ========== Numerical Stability Tests ==========

    def test_small_differences(self) -> None:
        """Test sorting with very small differences between values."""
        eps = 1e-7
        x = jnp.array([1.0 + eps, 1.0, 1.0 + 2 * eps, 1.0 - eps])
        result = groupsort(x, group_size=2)

        # Should correctly sort despite small differences
        result_grouped = result.reshape(2, 2)
        self.assertTrue(jnp.all(result_grouped[:, 0] <= result_grouped[:, 1]))

    def test_mixed_dtypes_float32(self) -> None:
        """Test with explicit float32 dtype."""
        x = jnp.array([3.0, 1.0, 4.0, 2.0], dtype=jnp.float32)
        result = groupsort(x, group_size=2)

        self.assertEqual(result.dtype, jnp.float32)
        expected = jnp.array([1.0, 3.0, 2.0, 4.0], dtype=jnp.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_mixed_dtypes_float64(self) -> None:
        """Test with explicit float64 dtype."""
        with jax.experimental.enable_x64():
            x = jnp.array([3.0, 1.0, 4.0, 2.0], dtype=jnp.float64)
            result = groupsort(x, group_size=2)

            self.assertEqual(result.dtype, jnp.float64)


if __name__ == "__main__":
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()
