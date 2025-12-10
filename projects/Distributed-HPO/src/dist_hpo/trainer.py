import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import functools

# Standalone update functions
@functools.partial(jax.pmap, axis_name='batch')
def train_step(state, batch):
    """
    Executes a single training step across devices.
    """
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']
        ).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # Aggregate gradients
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

@functools.partial(jax.pmap, axis_name='batch')
def eval_step(state, batch):
    logits = state.apply_fn({'params': state.params}, batch['image'])
    predicted_class = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predicted_class == batch['label'])
    accuracy = jax.lax.pmean(accuracy, axis_name='batch')
    return accuracy

class Trainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.tx = optax.adam(learning_rate)
        
        # Bind the functions for convenience if needed, though they are pmapped already
        self.train_step = train_step
        self.eval_step = eval_step
        
    def create_state(self, key, input_shape):
        """Creates the initial TrainState."""
        variables = self.model.init(key, jnp.ones(input_shape))
        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=variables['params'],
            tx=self.tx,
        )
