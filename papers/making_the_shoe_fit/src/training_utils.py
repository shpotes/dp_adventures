from functools import partial
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

def compute_metrics(logits, labels, split=""):
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
    accuracy = jnp.mean(
        jnp.argmax(logits, -1) == jnp.argmax(labels, -1)
    )
    metrics = {
        f"{split} loss": loss,
        f"{split} accuracy": accuracy
    }

    metrics = jax.device_get(metrics)
    #summary = jax.tree_map(lambda x: x.item(), metrics)
    
    return metrics

def create_train_state(model, tx, rng, input_shape=[28, 28, 1]):
    params = model.init(rng, jnp.empty([1, *input_shape]))["params"]
    
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

@partial(jax.jit, static_argnums=0)
def _train_step(model, state, batch):
    # TODO: add DP support

    def loss_fn(params):
        logits = model.apply({"params": params}, batch["image"])
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=batch["label"]))

        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, labels=batch["label"], split="train")
    
    return state, metrics

@partial(jax.jit, static_argnums=0)
def _test_step(model, params, batch):
    logits = model.apply({"params": params}, batch["image"])
    return compute_metrics(logits=logits, labels=batch["label"], split="test")