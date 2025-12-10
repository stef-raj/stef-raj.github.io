import numpy as np

class Solver:
    def step(self, x, problem):
        pass

class NewtonRaphson(Solver):
    """
    Newton-Raphson: x_new = x - H^-1 * grad
    Applicable to general optimization (minimizing scalar f(x)).
    Problem must provide: grad(x), hessian(x)
    """
    def __init__(self, damping=1.0):
        self.damping = damping

    def step(self, x, problem):
        grad = problem.grad(x)
        hess = problem.hessian(x)
        # Solve H * delta = -grad
        delta = np.linalg.solve(hess, -grad)
        return x + self.damping * delta

class GaussNewton(Solver):
    """
    Gauss-Newton: x_new = x - (J^T J)^-1 J^T r
    Applicable to Least Squares: min ||r(x)||^2
    Problem must provide: residuals(x), jacobian(x)
    """
    def __init__(self, damping=1.0):
        self.damping = damping

    def step(self, x, problem):
        r = problem.residuals(x)
        J = problem.jacobian(x)
        
        # Approximate Hessian H ~ J^T J
        # Gradient g = J^T r
        # Solve (J^T J) * delta = -J^T r
        JTJ = J.T @ J
        JTr = J.T @ r
        
        # We use pinv for stability usually, or lstsq, but strict formulation uses solve
        # Let's use solve but handle singular matrix with a tiny epsilon if needed, 
        # or just rely on numpy raising LinAlgError for "pure" GN failure demonstration
        try:
             delta = np.linalg.solve(JTJ, -JTr)
        except np.linalg.LinAlgError:
             # Fallback to gradient descent direction if singular (pseudo-inverse)
             delta = np.linalg.lstsq(JTJ, -JTr, rcond=None)[0]

        return x + self.damping * delta

class LevenbergMarquardt(Solver):
    """
    Levenberg-Marquardt: x_new = x - (J^T J + lambda * I)^-1 J^T r
    Interpolates between Gauss-Newton and Gradient Descent.
    """
    def __init__(self, initial_lambda=1.0, acceptance_ratio=1e-3, scale_lambda_up=2.0, scale_lambda_down=3.0):
        self.lam = initial_lambda
        self.acceptance_ratio = acceptance_ratio
        self.scale_up = scale_lambda_up
        self.scale_down = scale_lambda_down

    def step(self, x, problem):
        # LM actually needs to evaluate the function to check if the step is good.
        # This breaks the simple "step" interface slightly if we want to change lambda internally.
        # We will implement a 'robust' step that might return the same x if rejected.
        
        r = problem.residuals(x)
        J = problem.jacobian(x)
        
        current_cost = 0.5 * np.sum(r**2)
        
        JTJ = J.T @ J
        JTr = J.T @ r
        
        # Attempt step
        # (J^T J + lambda I) delta = -J^T r
        # Damping diagonal
        D = np.eye(len(x))
        # Often D is diag(JTJ) in "Marquardt" variant, but "Levenberg" uses I. Let's use I.
        
        while True:
            # Solve damped system
            A = JTJ + self.lam * D
            delta = np.linalg.solve(A, -JTr)
            
            x_candidate = x + delta
            r_candidate = problem.residuals(x_candidate)
            candidate_cost = 0.5 * np.sum(r_candidate**2)
            
            # Check improvement
            # Actual reduction
            actual_reduction = current_cost - candidate_cost
            
            # Predicted reduction (linearized model)
            # L(0) - L(delta) = -J^T r dot delta - 0.5 delta^T J^T J delta
            # This is 0.5 * delta^T (mu * delta - g) approx?
            # Standard formula: predict = -g^T d - 0.5 d^T H d
            # Here g = J^T r, H = J^T J.
            # pred = - (J^T r).T delta - 0.5 delta.T (J^T J) delta
            predicted_reduction = - np.dot(JTr, delta) - 0.5 * np.dot(delta, JTJ @ delta)
            
            rho = actual_reduction / predicted_reduction if predicted_reduction > 1e-15 else 0
            
            if rho > self.acceptance_ratio:
                # Accept
                self.lam = max(self.lam / self.scale_down, 1e-7)
                return x_candidate
            else:
                # Reject - increase lambda and loop
                # In a strict "step" call we might just return x and update lambda state
                # But to ensure progress in the script loop, we loop here.
                # Use a max loop safeguard
                self.lam *= self.scale_up
                if self.lam > 1e9:
                    # Give up step to avoid infinite loop
                    return x
