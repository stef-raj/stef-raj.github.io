import numpy as np
from sklearn.datasets import make_swiss_roll
from manifolds_kernels.utils.plotting import plot_swiss_roll

def main():
    print("Generating Swiss Roll...")
    n_samples = 2000
    # X shape (n_samples, 3), t shape (n_samples,)
    X, t = make_swiss_roll(n_samples=n_samples, noise=0.1)
    
    # Scale for nicer plotting
    X = X.astype(np.float32)
    
    # Visualize
    plot_swiss_roll(X, t, save_path="viz_manifold.png")
    
    print("Manifold Hypothesis:")
    print("The data lies on a 2D surface (the unrolled strip) embedded in 3D space.")
    print("Euclidean distance in 3D can be misleading (points across the gap are far on manifold but close in 3D).")

if __name__ == "__main__":
    main()
