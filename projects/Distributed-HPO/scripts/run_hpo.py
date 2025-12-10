import os
# Force 2 devices again just for consistency, though Ax usually runs trials sequentially or async
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from flax.jax_utils import replicate
from ax.service.ax_client import AxClient, ObjectiveProperties
from dist_hpo.model import CNN
from dist_hpo.trainer import Trainer

def get_datasets():
    # Helper to load small subset for HPO speed
    print("Fetching MNIST via sklearn...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.reshape(-1, 28, 28, 1).astype(np.float32) / 255.
    y = y.astype(np.int32)
    
    # Use small subset
    X_train = X[:5000]
    y_train = y[:5000]
    X_test = X[5000:6000]
    y_test = y[5000:6000]
    
    train_ds = {'image': X_train, 'label': y_train}
    test_ds = {'image': X_test, 'label': y_test}
    return train_ds, test_ds

def evaluate_model(parameterization):
    lr = parameterization.get('learning_rate', 0.001)
    
    # Init
    model = CNN()
    trainer = Trainer(model, learning_rate=lr)
    rng = jax.random.PRNGKey(0)
    init_state = trainer.create_state(rng, (1, 28, 28, 1))
    replicated_state = replicate(init_state)
    
    train_ds, test_ds = get_datasets()
    
    batch_size = 128
    
    # Train for small epochs
    images, labels = train_ds['image'], train_ds['label']
    steps = len(images) // batch_size
    
    for _ in range(3): # 3 epochs
        for i in range(steps):
             batch_imgs = images[i*batch_size : (i+1)*batch_size]
             batch_lbls = labels[i*batch_size : (i+1)*batch_size]
             
             # Reshape for 2 devices
             batch_imgs = batch_imgs.reshape((2, -1, 28, 28, 1))
             batch_lbls = batch_lbls.reshape((2, -1))
             
             replicated_state, _ = trainer.train_step(replicated_state, {'image': batch_imgs, 'label': batch_lbls})
             
    # Eval
    test_imgs = test_ds['image'][:128]
    test_lbls = test_ds['label'][:128]
    test_imgs = test_imgs.reshape((2, -1, 28, 28, 1))
    test_lbls = test_lbls.reshape((2, -1))
    
    acc = trainer.eval_step(replicated_state, {'image': test_imgs, 'label': test_lbls})
    return acc[0].item() # Return scalar accuracy

def main():
    ax_client = AxClient()
    
    ax_client.create_experiment(
        name="mnist_hpo",
        parameters=[
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [1e-4, 1e-2],
                "log_scale": True,
            },
        ],
        objectives={"accuracy": ObjectiveProperties(minimize=False)},
    )
    
    print("Running HPO...")
    
    for _ in range(5): # Run 5 trials
        parameters, trial_index = ax_client.get_next_trial()
        print(f"Trial {trial_index} params: {parameters}")
        
        acc = evaluate_model(parameters)
        print(f"Trial {trial_index} result: {acc:.4f}")
        
        ax_client.complete_trial(trial_index=trial_index, raw_data=acc)
        
    print("Best params:", ax_client.get_best_parameters())

if __name__ == "__main__":
    main()
