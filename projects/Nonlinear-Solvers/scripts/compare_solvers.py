import numpy as np
import matplotlib.pyplot as plt
from nonlinear_solvers.solvers import NewtonRaphson, GaussNewton, LevenbergMarquardt
from nonlinear_solvers.problems import ExponentialFitting

def run_solver(solver, problem, start_guess, max_iters=20):
    x = np.array(start_guess, dtype=float)
    history = [problem.cost(x)]
    
    for i in range(max_iters):
        try:
            x = solver.step(x, problem)
            history.append(problem.cost(x))
        except np.linalg.LinAlgError:
            print(f"Solver failed at iter {i}")
            break
            
    return history, x

def main():
    # Setup Data
    np.random.seed(42)
    t = np.linspace(0, 5, 20)
    # True params: a=3.0, b=1.5
    y_true = 3.0 * np.exp(-1.5 * t)
    # Add significant noise to make residuals non-zero (so NR and GN differ)
    y_data = y_true + 0.1 * np.random.normal(size=len(t))
    
    problem = ExponentialFitting(t, y_data)
    
    # Starting guess (far enough to be interesting)
    # True: [3.0, 1.5]
    start_guess = [1.0, 0.5]
    
    solvers = [
        (NewtonRaphson(damping=1.0), "Newton-Raphson"),
        (GaussNewton(damping=1.0), "Gauss-Newton"),
        (LevenbergMarquardt(initial_lambda=1.0), "Levenberg-Marquardt")
    ]
    
    plt.figure(figsize=(10, 6))
    
    print(f"Initial Cost: {problem.cost(start_guess):.4f}")
    
    for solver, name in solvers:
        hist, final_x = run_solver(solver, problem, start_guess)
        print(f"{name}: Final Cost {hist[-1]:.6f}, Params {final_x}")
        plt.plot(hist, 'o-', label=name, alpha=0.7)
        
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (RSS/2)')
    plt.title('Nonlinear Solver Convergence (Exponential Fit)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.savefig("solver_comparison.png")
    print("Saved solver_comparison.png")

if __name__ == "__main__":
    main()
