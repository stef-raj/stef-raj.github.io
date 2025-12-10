import numpy as np

class Objective:
    def __call__(self, x):
        """Value at x"""
        raise NotImplementedError
    
    def grad(self, x):
        """Gradient at x"""
        raise NotImplementedError

class Rosenbrock(Objective):
    """
    Rosenbrock function: f(x, y) = (a - x)^2 + b * (y - x^2)^2
    Global minimum at (a, a^2).
    Standard parameters: a=1, b=100 -> min at (1, 1).
    """
    def __init__(self, a=1.0, b=100.0):
        self.a = a
        self.b = b
        
    def __call__(self, params):
        x, y = params[0], params[1]
        return (self.a - x)**2 + self.b * (y - x**2)**2
    
    def grad(self, params):
        x, y = params[0], params[1]
        # df/dx = -2(a-x) - 4b(y - x^2)*x
        dx = -2 * (self.a - x) - 4 * self.b * (y - x**2) * x
        # df/dy = 2b(y - x^2)
        dy = 2 * self.b * (y - x**2)
        return np.array([dx, dy])
