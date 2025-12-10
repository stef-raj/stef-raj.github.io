import os
# Force 8 simulated devices on CPU
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from dist_hpo.model import CNN
from dist_hpo.trainer import Trainer

def get_datasets():
    print("Fetching MNIST via sklearn...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.reshape(-1, 28, 28, 1).astype(np.float32) / 255.
    y = y.astype(np.int32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)
    
    # Return dicts to match previous interface
    train_ds = {'image': X_train, 'label': y_train}
    test_ds = {'image': X_test, 'label': y_test}
    return train_ds, test_ds

def main():
    print(f"JAX Devices: {jax.local_device_count()}")
    assert jax.local_device_count() == 2, "Expected 2 devices (simulated)"

    # Hyperparams
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 5
    
    train_ds, test_ds = get_datasets()
    train_images, train_labels = train_ds['image'], train_ds['label']
    
    # Initialize Model & State on Host (1 copy)
    model = CNN()
    trainer = Trainer(model, learning_rate)
    
    rng = jax.random.PRNGKey(0)
    init_state = trainer.create_state(rng, (1, 28, 28, 1))
    
    # Replicate State across devices
    # params -> (8, params_shape...)
    replicated_state = replicate(init_state)
    
    num_train_steps = len(train_images) // batch_size
    
    print("Starting Distributed Training...")
    
    for epoch in range(num_epochs):
        # Shuffle logic omitted for brevity in sim
        
        epoch_loss = 0
        
        for i in range(num_train_steps):
            batch_images = train_images[i*batch_size : (i+1)*batch_size]
            batch_labels = train_labels[i*batch_size : (i+1)*batch_size]
            
            # Shard data: (2, batch/2, ...)
            # We must reshape input to (num_devices, batch_per_device, ...)
            device_batch_size = batch_size // 2
            batch_images = batch_images.reshape((2, device_batch_size, 28, 28, 1))
            batch_labels = batch_labels.reshape((2, device_batch_size))
            
            batch = {'image': batch_images, 'label': batch_labels}
            
            # Parallel Update
            replicated_state, loss = trainer.train_step(replicated_state, batch)
            
            # Loss is returned as (2,), seeing as pmean makes them all equal, just take first
            epoch_loss += loss[0]
            
        print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss/num_train_steps:.4f}")
        
    print("Training Complete. Evaluating...")
    # Evaluation (Just taking first batch of test set for demo)
    test_batch_size = 128
    test_imgs = test_ds['image'][:test_batch_size]
    test_lbls = test_ds['label'][:test_batch_size]
    
    test_imgs = test_imgs.reshape((2, 64, 28, 28, 1))
    test_lbls = test_lbls.reshape((2, 64))
    
    acc = trainer.eval_step(replicated_state, {'image': test_imgs, 'label': test_lbls})
    print(f"Test Accuracy (Batch): {acc[0]:.4f}")

if __name__ == "__main__":
    main()
