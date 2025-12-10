import numpy as np
import scipy.linalg as la

def solve_lu(A, b):
    """
    Solve Ax = b using LU decomposition.
    PA = LU
    Ly = Pb
    Ux = y
    """
    lu, piv = la.lu_factor(A)
    return la.lu_solve((lu, piv), b)

def solve_cholesky(A, b):
    """
    Solve Ax = b using Cholesky decomposition.
    A = LL^T
    Ly = b
    L^T x = y
    Only for SPD matrices.
    """
    # lower=True means A = L L^T
    L = la.cholesky(A, lower=True)
    return la.cho_solve((L, True), b)

def solve_qr(A, b):
    """
    Solve Ax = b using QR decomposition.
    A = QR
    Rx = Q^T b
    """
    Q, R = la.qr(A)
    # y = Q^T b
    y = Q.T @ b
    # Solve Rx = y (back substitution)
    return la.solve_triangular(R, y)

def solve_svd(A, b):
    """
    Solve Ax = b using SVD.
    A = U Sigma V^T
    x = V Sigma^-1 U^T b
    Use pseudo-inverse (pinv) approach if singular, but here generic full rank.
    """
    U, s, Vt = la.svd(A)
    # x = V @ diag(1/s) @ U.T @ b
    # Check for near-zero singular values for stability (pseudo-inverse like)
    s_inv = 1.0 / s
    # In practice for linear solve we iterate, here explicit inverse construction for demo of theory
    return Vt.T @ (s_inv[:, None] * (U.T @ b))
