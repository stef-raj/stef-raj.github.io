import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_swiss_roll(data, color, save_path="manifold_viz.png"):
    fig = plt.figure(figsize=(12, 5))
    
    # 3D View
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, cmap=plt.cm.Spectral, s=10)
    ax.set_title("3D Swiss Roll (Manifold)")
    ax.view_init(10, 80)
    
    # 2D Unrolled View (using the color parameter as the intrinsic coordinate approx)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(color, data[:, 1], c=color, cmap=plt.cm.Spectral, s=10)
    ax2.set_title("Intrinsic Manifold (Unrolled)")
    ax2.set_xlabel("Unrolled Dimension 1 (t)")
    ax2.set_ylabel("Dimension 2 (height)")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")

def plot_kernel_results(X, y, model, title="Kernel Ridge Regression", save_path="kernel_viz.png"):
    plt.figure(figsize=(8, 6))
    
    # Plot Data
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red', label='Class 0', s=20)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', label='Class 1', s=20)
    
    # Decision Boundary
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on grid
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    # Assuming model has a predict method that handles numpy/mlx
    Z = model.predict(grid_pts)
    Z = np.array(Z).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=20, cmap=plt.cm.RdBu, alpha=0.3)
    plt.colorbar()
    
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    print(f"Saved {save_path}")
