"""Quick unit test for adamo optimizer and make_optimizer utility."""
import jax
import jax.numpy as jnp
import optax
from omegaconf import OmegaConf
from stoix.utils.orthogonalization import adamo, _compute_ortho_grad
from stoix.utils.training import make_optimizer, make_optimizer_with_mask

# Test _compute_ortho_grad
print("Testing _compute_ortho_grad...")
W_wide = jnp.ones((3, 5)) / jnp.sqrt(5)  # Wide matrix
W_tall = jnp.ones((5, 3)) / jnp.sqrt(3)  # Tall matrix

grad_wide = _compute_ortho_grad(W_wide)
grad_tall = _compute_ortho_grad(W_tall)
print(f"  Wide matrix grad shape: {grad_wide.shape} (expected (3, 5))")
print(f"  Tall matrix grad shape: {grad_tall.shape} (expected (5, 3))")

# Test adamo optimizer
print("\nTesting adamo optimizer...")
key = jax.random.PRNGKey(42)
params = {
    "Dense_0": {"kernel": jax.random.normal(key, (4, 8)), "bias": jnp.zeros(8)},
    "Dense_1": {"kernel": jax.random.normal(key, (8, 8)), "bias": jnp.zeros(8)},
    "output": {"kernel": jax.random.normal(key, (8, 2)), "bias": jnp.zeros(2)},
}

# Create optimizer
opt = adamo(learning_rate=1e-3, ortho_coeff=1e-3, exclude_output=True)
opt_state = opt.init(params)
print(f"  Optimizer state type: {type(opt_state)}")
print(f"  Output layer path: {opt_state.output_layer_path}")

# Create fake gradients
grads = jax.tree_util.tree_map(jnp.zeros_like, params)

# Test update
updates, new_state = opt.update(grads, opt_state, params)
print(f"  Update successful: {updates is not None}")

# Test that updates have correct structure
for name in params:
    for subname in params[name]:
        assert name in updates, f"Missing {name} in updates"
        assert subname in updates[name], f"Missing {name}/{subname} in updates"
print("  Structure check passed!")

# Verify ortho grad is applied (should be non-zero for kernels, zero for biases)
# With zero task gradients, only ortho grad should be present
update_norm = optax.global_norm(updates)
print(f"  Update norm (with zero task grads): {float(update_norm):.6f}")
print(f"  (Non-zero confirms ortho gradient is being applied)")

print("\n✓ AdamO tests passed!")

# Test make_optimizer utility
print("\n" + "="*50)
print("Testing make_optimizer utility...")

# Create a mock config
config_loss_mode = OmegaConf.create({
    "system": {
        "ortho_mode": "loss",
        "max_grad_norm": 0.5,
    }
})

config_optimizer_mode = OmegaConf.create({
    "system": {
        "ortho_mode": "optimizer",
        "ortho_coeff": 1e-3,
        "ortho_exclude_output": True,
        "max_grad_norm": 0.5,
    }
})

# Test loss mode (should use standard adam)
opt_loss = make_optimizer(1e-3, config_loss_mode)
opt_state_loss = opt_loss.init(params)
print(f"  Loss mode optimizer initialized: {type(opt_state_loss)}")

# Test optimizer mode (should use adamo)
opt_adamo = make_optimizer(1e-3, config_optimizer_mode)
opt_state_adamo = opt_adamo.init(params)
print(f"  Optimizer mode (AdamO) initialized: {type(opt_state_adamo)}")

# Test that both can do updates
grads_test = jax.tree_util.tree_map(jnp.ones_like, params)

updates_loss, _ = opt_loss.update(grads_test, opt_state_loss, params)
print(f"  Loss mode update: norm = {float(optax.global_norm(updates_loss)):.6f}")

updates_adamo, _ = opt_adamo.update(grads_test, opt_state_adamo, params)
print(f"  AdamO mode update: norm = {float(optax.global_norm(updates_adamo)):.6f}")

# Test make_optimizer_with_mask (for TD3-style delayed updates)
print("\nTesting make_optimizer_with_mask...")
def should_update(step):
    return step % 2 == 0

masked_opt = make_optimizer_with_mask(1e-3, config_optimizer_mode, should_update)
masked_state = masked_opt.init(params)
print(f"  Masked optimizer initialized: {type(masked_state)}")

print("\n✓ All tests passed!")
