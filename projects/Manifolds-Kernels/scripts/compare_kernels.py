import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

from manifolds_kernels.models.krr import KernelRidgeRegression
from manifolds_kernels.utils.plotting import plot_kernel_results

def main():
    print("Generating Concentric Circles (Non-linear)...")
    X, y = make_circles(n_samples=500, factor=0.3, noise=0.1)
    
    # Map y from {0, 1} to {-1, 1} for regression logic
    y_reg = 2 * y - 1
    
    # 1. Linear Model (using KRR with linear kernel)
    print("Training Linear KRR...")
    lin_model = KernelRidgeRegression(kernel='linear', alpha=0.1)
    lin_model.fit(X, y_reg)
    
    # 2. RBF Kernel Model
    print("Training RBF KRR...")
    rbf_model = KernelRidgeRegression(kernel='rbf', alpha=0.1, gamma=2.0)
    rbf_model.fit(X, y_reg)
    
    # Visualization
    plot_kernel_results(X, y, lin_model, title="Linear Kernel (Fails on Non-linear)", save_path="linear_viz.png")
    plot_kernel_results(X, y, rbf_model, title="RBF Kernel (Success via Kernel Trick)", save_path="rbf_viz.png")
    
    print("Kernel Trick Implementation Complete.")
    print("Linear model cannot separate circles.")
    print("RBF model lifts data to infinite dim feature space, making it linearly separable there.")

if __name__ == "__main__":
    main()
