import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
from typing import Iterator, Tuple
from renformer.data_utils import create_batch


# =====================================================
# Training Step
# =====================================================

def gaussian_nll(mean, log_std, target):
    """Negative log-likelihood of a diagonal Gaussian."""
    var = jnp.exp(2.0 * log_std)
    return 0.5 * jnp.mean((target - mean) ** 2 / var + 2.0 * log_std)

@jax.jit
def train_step(params, opt_state, model, optimizer, x, y, rng):
    def loss_fn(params):
        mean, log_std = model.apply(params, x, mask=(x[..., :model.out_features] != 0), train=True, rngs={"dropout": rng})
        return gaussian_nll(mean, log_std, y)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# =====================================================
# Training Loop
# =====================================================

def train(
    model,
    X_train,
    Y_train,
    epochs=20,
    batch_size=32,
    lr=1e-4,
    seed=0,
):
    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, jnp.array(X_train[:1]), train=True)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    for epoch in range(1, epochs + 1):
        losses = []
        for xb, yb in create_batch(X_train, Y_train, batch_size):
            rng, step_rng = jax.random.split(rng)
            params, opt_state, loss = train_step(
                params, opt_state, model, optimizer, xb, yb, step_rng
            )
            losses.append(loss)

        print(f"Epoch {epoch:03d} | Loss: {jnp.mean(jnp.array(losses)):.6f}")

    return params

