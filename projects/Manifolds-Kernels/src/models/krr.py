import mlx.core as mx
# mlx.linalg is not a top-level module, use mx.linalg or direct imports if needed.
# However, for inverse, we might need to check if it's implemented in mx.linalg

from ..kernels import rbf_kernel, linear_kernel

class KernelRidgeRegression:
    def __init__(self, kernel='rbf', alpha=1.0, gamma=1.0):
        self.kernel_name = kernel
        self.alpha = alpha # Regularization strength
        self.gamma = gamma
        self.X_train = None
        self.dual_coef_ = None # These are the 'alpha' in dual formulation
        
    def _compute_kernel(self, X, Y):
        if self.kernel_name == 'rbf':
            return rbf_kernel(X, Y, gamma=self.gamma)
        elif self.kernel_name == 'linear':
            return linear_kernel(X, Y)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")

    def fit(self, X, y):
        """
        Solve (K + alpha * I) * dual_coef = y
        """
        self.X_train = mx.array(X)
        y = mx.array(y)
        
        K = self._compute_kernel(self.X_train, self.X_train)
        N = K.shape[0]
        
        # Regularization
        K_reg = K + self.alpha * mx.eye(N)
        
        # Solve linear system
        # Inverse on CPU as per MLX limitations for some ops
        with mx.stream(mx.cpu):
             inv_K = mx.linalg.inv(K_reg)
             
        self.dual_coef_ = inv_K @ y
        
    def predict(self, X_test):
        X_test = mx.array(X_test)
        K_test = self._compute_kernel(X_test, self.X_train)
        return K_test @ self.dual_coef_
