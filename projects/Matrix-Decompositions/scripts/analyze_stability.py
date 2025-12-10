import time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matrix_decomps.solvers import solve_lu, solve_cholesky, solve_qr, solve_svd
from matrix_decomps.least_squares import solve_normal_equations, solve_ls_svd

def benchmark_solvers():
    sizes = [100, 500, 1000]
    results = {
        'LU': [], 'Cholesky': [], 'QR': [], 'SVD': []
    }
    
    print("Benchmarking Solvers (Solve Ax=b)...")
    for n in sizes:
        # Create SPD matrix for Cholesky compatibility
        # A = Q D Q^T
        Q, _ = la.qr(np.random.randn(n, n))
        D = np.diag(np.random.rand(n) + 0.1) # positive eigenvalues
        A = Q @ D @ Q.T
        b = np.random.randn(n)
        
        print(f"Size {n}...")
        
        # LU
        start = time.time()
        solve_lu(A, b)
        results['LU'].append(time.time() - start)
        
        # Cholesky
        start = time.time()
        solve_cholesky(A, b)
        results['Cholesky'].append(time.time() - start)
        
        # QR
        start = time.time()
        solve_qr(A, b)
        results['QR'].append(time.time() - start)
        
        # SVD
        start = time.time()
        solve_svd(A, b)
        results['SVD'].append(time.time() - start)
        
    return sizes, results

def analyze_LS_stability():
    print("\nAnalyzing Least Squares Stability (Hilbert Matrix)...")
    sizes = range(2, 12, 1) # condition number explodes fast
    err_normal = []
    err_svd = []
    cond_nums = []
    
    for n in sizes:
        A = la.hilbert(n)
        x_true = np.ones(n)
        b = A @ x_true
        
        # Condition number
        cond = np.linalg.cond(A)
        cond_nums.append(cond)
        
        # Normal Eqs
        x_norm = solve_normal_equations(A, b)
        err_norm = np.linalg.norm(x_norm - x_true)
        err_normal.append(err_norm)
        
        # SVD
        x_svd_sol = solve_ls_svd(A, b)
        e_svd = np.linalg.norm(x_svd_sol - x_true)
        err_svd.append(e_svd)
        
    return sizes, cond_nums, err_normal, err_svd

def main():
    # 1. Benchmark Runtime
    sizes, times = benchmark_solvers()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, t in times.items():
        plt.plot(sizes, t, 'o-', label=name)
    plt.xlabel("Matrix Size N")
    plt.ylabel("Time (s)")
    plt.title("Linear Solver Runtime Cost")
    plt.legend()
    plt.grid(True)
    
    # 2. Stability Analysis
    n_vals, conds, err_norm, err_svd = analyze_LS_stability()
    
    plt.subplot(1, 2, 2)
    plt.plot(n_vals, err_norm, 'r-x', label="Normal Eqs (k^2)")
    plt.plot(n_vals, err_svd, 'b-o', label="SVD (k)")
    plt.plot(n_vals, [1e-16 * c for c in conds], 'k--', alpha=0.3, label="~eps * cond(A)")
    plt.plot(n_vals, [1e-16 * c**2 for c in conds], 'k:', alpha=0.3, label="~eps * cond(A)^2")
    
    plt.yscale('log')
    plt.xlabel("Hilbert Matrix Size N")
    plt.ylabel("Error ||x - x_true||")
    plt.title("Least Squares Stability")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("stability_analysis.png")
    print("Saved stability_analysis.png")

if __name__ == "__main__":
    main()
