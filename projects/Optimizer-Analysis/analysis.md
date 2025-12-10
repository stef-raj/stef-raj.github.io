# Optimizer Analysis: Deep Dive

This document analyzes the mathematical foundations of modern gradient-based optimizers and discusses the trade-offs between first-order and second-order methods.

## 1. Optimizer Derivations

All methods aim to minimize an objective function $J(\theta)$.

### 1.1 First-Order Momentum (SGD + Momentum)
Standard SGD oscillates in ravines. Momentum accumulates a velocity vector $v$ to smooth out the path.
$$ v_{t} = \gamma v_{t-1} + \eta \nabla_\theta J(\theta) $$
$$ \theta_{t} = \theta_{t-1} - v_t $$

### 1.2 Adadelta
Designed to resolve two drawbacks of Adagrad:
1. Continual decay of learning rates (due to accumulating all past squared gradients).
2. Need for a manually selected global learning rate.

Adadelta restricts the window of accumulated past gradients to some fixed size $w$ using an optional exponential moving average.
- Accumulate Gradient: $E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho) g_t^2$
- Compute Update (using RMS of past updates $E[\Delta x^2]$):
$$ \Delta \theta_t = - \frac{\text{RMS}[\Delta \theta]_{t-1}}{\text{RMS}[g]_t} g_t $$
- Accumulate Updates: $E[\Delta \theta^2]_t = \rho E[\Delta \theta^2]_{t-1} + (1-\rho) \Delta \theta_t^2$
$$ \theta_{t+1} = \theta_t + \Delta \theta_t $$

Notice no learning rate $\eta$ is required. The units of update match the units of parameters.

### 1.3 Adam (Adaptive Moment Estimation)
Combines ideas from Momentum (1st moment) and RMSProp (2nd moment).
- 1st Moment (Mean): $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
- 2nd Moment (Uncentered Variance): $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
- **Bias Correction**: Since $m_0=0, v_0=0$, estimates are biased towards zero initially.
  $$ \hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} $$
- Update:
  $$ \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t $$

### 1.4 NAdamW (Nesterov Adam + Weight Decay)
Improvements:
1. **Nesterov Momentum**: Lookahead momentum. Replaces $\hat{m}_t$ in the update rule with a lookahead version.
   $$ \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} (\beta_1 \hat{m}_t + \frac{(1-\beta_1) g_t}{1-\beta_1^t}) $$
2. **Decoupled Weight Decay (AdamW)**: $L2$ regularization in Adam is usually implemented as adding $\lambda \theta$ to gradient $g_t$. This couples the regularization with the adaptive learning rate logic ($1/\sqrt{\hat{v}_t}$). AdamW decouples this by decaying weights *directly*:
   $$ \theta_t = \theta_t - \eta \lambda \theta_{t-1} $$
   It performs this decay *before* or *after* the gradient-based update, ensuring the decay rate is consistent regardless of gradient magnitude.

---

## 2. First-Order vs Second-Order Optimization

### 2.1 Definitions
- **First-Order (SGD, Adam)**: Use Jacobian/Gradient ($\nabla J$). Linear approximation of the surface.
- **Second-Order (Newton, Gauss-Newton)**: Use Hessian Matrix ($H$). Quadratic approximation.
  $$ \theta_{new} = \theta_{old} - H^{-1} \nabla J $$

### 2.2 Why First-Order for Deep Learning?

1.  **Computational Complexity**:
    -   Example: A model with $N=10^8$ parameters.
    -   Hessian is $N \times N$, requiring $10^{16}$ floats (Petabytes of RAM).
    -   Inverting Hessian ($H^{-1}$) is $O(N^3)$. Impossible for large networks.

2.  **Saddle Points**:
    -   In high dimensions, saddle points are exponentially more common than local minima.
    -   Newton's method ($H^{-1} \nabla J$) is attracted to critical points where $\nabla J=0$, often jumping onto saddle points, whereas Gradient Descent naturally rolls "downhill" away from them.

3.  **Noise Robustness**:
    -   DL training uses Mini-batches (High noise). Approximating curvature ($H$) from noisy mini-batches is extremely unstable compared to approximating the mean gradient.

**Conclusion**: While second-order methods converge in fewer *steps*, each step is prohibitively expensive. First-order methods with "approximate curvature" features (like Adam's adaptive learning rate) strike the best balance.
