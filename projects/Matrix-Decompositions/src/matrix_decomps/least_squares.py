import numpy as np
import scipy.linalg as la

def solve_normal_equations(A, b):
    """
    Solve Least Squares via Normal Equations:
    A^T A x = A^T b
    
    Warning: Condition number is squared (k^2), unstable for ill-conditioned A.
    """
    ATA = A.T @ A
    ATb = A.T @ b
    # Solve ATA x = ATb using Cholesky (since ATA is SPD) or standard solve
    try:
        # We generally use Cholesky for Normal Eqs as ATA is symmetric positive definite
        L = la.cholesky(ATA, lower=True)
        return la.cho_solve((L, True), ATb)
    except la.LinAlgError:
        # Fallback if not positive definite (numerical issues)
        return la.solve(ATA, ATb)

def pseudo_inverse(A, rcond=1e-15):
    """
    Compute Moore-Penrose Pseudo-Inverse A^+ using SVD.
    A = U S V^T
    A^+ = V S^+ U^T
    where S^+ is diagonal with 1/s_i for s_i > threshold, else 0.
    """
    U, s, Vt = la.svd(A, full_matrices=False)
    
    # Invert singular values
    s_inv = np.zeros_like(s)
    mask = s > rcond * s.max()
    s_inv[mask] = 1.0 / s[mask]
    
    # Construct A^+ = V @ S^+ @ U^T
    return Vt.T @ np.diag(s_inv) @ U.T

def solve_ls_svd(A, b):
    """
    Solve Least Squares using SVD directly (Pseudo-Inverse).
    x = A^+ b
    Stable (condition number k).
    """
    pinv = pseudo_inverse(A)
    return pinv @ b
