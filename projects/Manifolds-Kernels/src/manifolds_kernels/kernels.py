import mlx.core as mx

def rbf_kernel(X, Y, gamma=1.0):
    """
    Computes RBF kernel: exp(-gamma * ||x - y||^2)
    X: (N, D)
    Y: (M, D)
    """
    # Squared Euclidean Distance
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
    
    X_sq = mx.sum(mx.square(X), axis=1, keepdims=True) # (N, 1)
    Y_sq = mx.sum(mx.square(Y), axis=1, keepdims=True) # (M, 1)
    
    # Needs broadcasting: (N, 1) + (1, M) -> (N, M)
    sq_dist = X_sq + Y_sq.T - 2 * (X @ Y.T)
    
    return mx.exp(-gamma * sq_dist)

def linear_kernel(X, Y):
    return X @ Y.T

def polynomial_kernel(X, Y, degree=3.0, coef0=1.0):
    return (X @ Y.T + coef0) ** degree
