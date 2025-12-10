import numpy as np

class Problem:
    def residuals(self, x):
        raise NotImplementedError
    def jacobian(self, x):
        raise NotImplementedError
    def cost(self, x):
        r = self.residuals(x)
        return 0.5 * np.sum(r**2)
    def grad(self, x):
        # Gradient of cost for Hessian-based methods
        J = self.jacobian(x)
        r = self.residuals(x)
        return J.T @ r
    def hessian(self, x):
        # Full Hessian = J^T J + sum(r_i * Hessian(r_i))
        # This makes Newton-Raphson distinct from Gauss-Newton
        a, b = x
        J = self.jacobian(x)
        r = self.residuals(x)
        JTJ = J.T @ J
        
        # Second derivative term S = sum(r_i * H_i)
        # H_i is 2x2 matrix for i-th residual
        # d2r/da2 = 0
        # d2r/dadb = t_i * exp(-b * t_i)
        # d2r/db2 = -a * t_i^2 * exp(-b * t_i)
        
        S = np.zeros((2, 2))
        
        exp_term = np.exp(-b * self.t)
        h_ab = self.t * exp_term
        h_bb = -a * (self.t**2) * exp_term
        
        # We sum: r_i * [ [0, h_ab_i], [h_ab_i, h_bb_i] ]
        # summing over i
        
        S[0, 1] = np.sum(r * h_ab)
        S[1, 0] = S[0, 1]
        S[1, 1] = np.sum(r * h_bb)
        
        return JTJ + S

class ExponentialFitting(Problem):
    """
    Fit model: y = a * exp(-b * t)
    Parameters x = [a, b]
    Resid: r_i = y_i - a * exp(-b * t_i)
    """
    def __init__(self, t_data, y_data):
        self.t = t_data
        self.y = y_data
        
    def residuals(self, x):
        a, b = x
        return self.y - a * np.exp(-b * self.t)
    
    def jacobian(self, x):
        # r_i = y_i - a * exp(-b * t_i)
        # dr/da = -exp(-b * t_i)
        # dr/db = -a * exp(-b * t_i) * (-t_i) = a * t_i * exp(-b * t_i)
        a, b = x
        exp_term = np.exp(-b * self.t)
        
        dr_da = -exp_term
        dr_db = a * self.t * exp_term
        
        # J has shape (N, 2)
        return np.column_stack([dr_da, dr_db])
