import numpy as np

class Optimizer:
    def step(self, params, grads):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
        
    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
            
        self.velocity = self.momentum * self.velocity - self.lr * grads
        return params + self.velocity

class Adadelta(Optimizer):
    def __init__(self, rho=0.9, epsilon=1e-6):
        self.rho = rho
        self.epsilon = epsilon
        self.eg2 = None # Accumulate squared gradients
        self.edx2 = None # Accumulate squared updates
        
    def step(self, params, grads):
        if self.eg2 is None:
            self.eg2 = np.zeros_like(params)
            self.edx2 = np.zeros_like(params)
            
        # Accumulate Gradient
        self.eg2 = self.rho * self.eg2 + (1 - self.rho) * grads**2
        
        # Compute Update
        rms_dx = np.sqrt(self.edx2 + self.epsilon)
        rms_g = np.sqrt(self.eg2 + self.epsilon)
        
        dx = -(rms_dx / rms_g) * grads
        
        # Accumulate Update
        self.edx2 = self.rho * self.edx2 + (1 - self.rho) * dx**2
        
        return params + dx

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class NAdamW(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0
        
    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.t += 1
        
        # Weight Decay (Decoupled)
        params = params - self.lr * self.weight_decay * params
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Nesterov Momentum term: beta1 * m_hat + (1-beta1) * g / (1-beta1^t)
        # However, standard NAdam implementation often simplifies this.
        # Let's use the standard formula:
        # theta_{t+1} = theta_t - lr * (beta1 * m_hat + (1-beta1)/ (1-beta1^t) * g) / (sqrt(v_hat) + eps)
        
        # Nesterov correction to m_hat
        m_nesterov = self.beta1 * m_hat + (1 - self.beta1) * grads / (1 - self.beta1**self.t)
        
        return params - self.lr * m_nesterov / (np.sqrt(v_hat) + self.epsilon)
