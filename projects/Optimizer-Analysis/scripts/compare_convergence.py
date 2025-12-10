import numpy as np
import matplotlib.pyplot as plt
from optimizer_analysis.objectives import Rosenbrock
from optimizer_analysis.optimizers import SGD, Adadelta, Adam, NAdamW

def run_optimization(optimizer_cls, name, steps=1000, start_point=(-1.5, 2.0), **kwargs):
    obj = Rosenbrock()
    optimizer = optimizer_cls(**kwargs)
    
    params = np.array(start_point)
    path = [params]
    values = [obj(params)]
    
    for _ in range(steps):
        grads = obj.grad(params)
        params = optimizer.step(params, grads)
        path.append(params)
        values.append(obj(params))
        
    return np.array(path), np.array(values)

def main():
    # Setup
    start_point = (-1.5, 1.5)
    steps = 2000
    
    optimizers = [
        (SGD, "SGD", {"learning_rate": 0.002, "momentum": 0.9}),
        (Adadelta, "Adadelta", {"rho": 0.9}),
        (Adam, "Adam", {"learning_rate": 0.1}), # High LR for Adam on Rosenbrock usually fine
        (NAdamW, "NAdamW", {"learning_rate": 0.1, "weight_decay": 1e-4}),
    ]
    
    plt.figure(figsize=(12, 5))
    
    # Contour Plot of Rosenbrock
    x = np.linspace(-2.0, 2.0, 400)
    y = np.linspace(-1.0, 3.0, 400)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    
    plt.subplot(1, 2, 1)
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet', alpha=0.5)
    plt.plot(1, 1, 'r*', markersize=10, label='Global Min')
    
    for opt_cls, name, kwargs in optimizers:
        print(f"Running {name}...")
        path, values = run_optimization(opt_cls, name, steps=steps, start_point=start_point, **kwargs)
        
        # Plot Path
        plt.plot(path[:, 0], path[:, 1], label=name, alpha=0.8, linewidth=1.5)
        
        # Plot Convergence (Loss)
        plt.subplot(1, 2, 2)
        plt.plot(values, label=name)
        
    plt.subplot(1, 2, 1)
    plt.title("Optimizer Trajectories (Rosenbrock)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title("Convergence Rates (Log Scale)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("optimizer_comparison.png")
    print("Saved optimizer_comparison.png")

if __name__ == "__main__":
    main()
