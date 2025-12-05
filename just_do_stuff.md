# 🎓 Just Know Do Stuff: A Project-Based Curriculum

This project-based curriculum is directly inspired by Patrick Kidger's influential post, "Just Know Stuff," which outlines the crucial knowledge base for success in a machine learning PhD.

## 🤖 Machine Learning Projects

* **Custom Autodifferentiation Implementations**
    * Implement and understand **forward- and reverse-mode autodifferentiation**.
    * Write **custom gradient operations** in both PyTorch and JAX.
    * *Link to project details:* [Project: Custom Autodiff](/projects/autodiff/)
* **Efficient Jacobian and Trace Estimation**
    * Investigate and implement techniques for **optimal Jacobian accumulation**.
    * Learn and implement the **Hutchinson's trace estimator** and **Hutch++ trace estimator**.
    * *Link to project details:* [Project: Trace Estimation](/projects/trace-estimation/)
* **Convolutional and Attention Layer Implementation**
    * Implement the core mechanics of Strassen's algorithm and **Winograd convolutions**.
    * Write a custom implementation of a **convolutional layer**.
    * Write a custom implementation of **multihead attention**.
    * *Link to project details:* [Project: Layer Implementation](/projects/layer-implementation/)
* **Universal Approximation Exploration**
    * Study the proofs and implications of the **Universal Approximation Theorem** (Leshno et al. 1993 or Pinkus 1999).
    * *Link to project details:* [Project: UAT Deep Dive](/projects/uat-deep-dive/)
* **Architectural Implementations**
    * Implement a **Graph Neural Network** (GNN) and investigate the **oversmoothing** problem.
    * Build a toy implementation of a modern **Transformer architecture**.
    * Build a toy implementation of a **U-Net** architecture.
    * *Link to project details:* [Project: Modern Architectures](/projects/modern-architectures/)
* **Understanding Differential Equation Relationships**
    * Write a brief report/code proving the connection between **Residual Networks** and **discretised Ordinary Differential Equations (ODEs)**.
    * Investigate the relationship between **Gated Recurrent Units (GRUs)** and discretised ODEs.
    * Study how **Stochastic Gradient Descent (SGD)** can be viewed as a **discretised differential equation (gradient flow)**.
    * *Link to project details:* [Project: ML and Differential Equations](/projects/ml-diffeq/)
* **Generative Models and Divergences**
    * Study and implement the core differences between **KL divergence**, **Wasserstein distance**, and **MMD distance**.
    * Implement a basic **score-based diffusion model** from scratch.
    * Study and implement the basics of **normalising flows**, **VAEs**, and **WGANs**.
    * *Link to project details:* [Project: Generative Models](/projects/generative-models/)
* **Reinforcement Learning Implementation**
    * Learn the basics of **policy gradients**.
    * Implement the **PPO algorithm** to solve the **cart-pole problem**.
    * *Link to project details:* [Project: PPO Implementation](/projects/ppo-rl/)
* **Distributed Training and Hyperparameter Optimization**
    * Implement basic distributed training using **JAX's `pmap`** (across multiple GPUs or simulated devices).
    * Conduct hyperparameter optimization using **Bayesian optimization** (e.g., using the Ax library).
    * *Link to project details:* [Project: Distributed and HPO](/projects/distributed-hpo/)
* **Optimiser Deep Dive**
    * Derive the formulas for optimizers like **Adadelta, Adam, NAdamW**, etc., and document their core innovations.
    * Document the reasons for using **first-order optimisation techniques** (SGD) over second-order methods (Gauss–Newton, Newton–Raphson, Levenberg–Marquardt).
    * *Link to project details:* [Project: Optimiser Analysis](/projects/optimizer-analysis/)

---

## 💻 Scientific Computing Projects

* **Nonlinear Solvers Implementation**
    * Implement and compare the performance of classic nonlinear solvers: **Gauss–Newton**, **Newton–Raphson**, and **Levenberg–Marquardt**.
    * *Link to project details:* [Project: Nonlinear Solvers](/projects/nonlinear-solvers/)
* **Matrix Decompositions and Linear Solves**
    * Implement and compare solving a linear system ($Ax=b$) using **QR, LU, SVD, and Cholesky decompositions**. Document their computational costs and stability.
    * Solve a **linear least squares** problem via the **normal equations** and analyze the condition number squaring effect.
    * *Link to project details:* [Project: Matrix Decompositions](/projects/matrix-decompositions/)
* **Numerical Differential Equation Solvers**
    * Implement and compare the **Explicit Euler method** and **Heun's method**.
    * *(Optional)* Implement the **Implicit Euler method** and analyze its stability condition based on the contraction mapping theorem.
    * *Link to project details:* [Project: Diffeq Solvers](/projects/diffeq-solvers/)
* **Numerical Integration and Sampling**
    * Implement and compare **Monte-Carlo sampling** and **Quasi Monte-Carlo sampling**, documenting their convergence rates.
    * Implement a basic form of **quadrature**.
    * *Link to project details:* [Project: Numerical Integration](/projects/numerical-integration/)
* **Floating-Point Arithmetic Case Studies**
    * Create code examples that clearly demonstrate **non-associativity**, **catastrophic cancellation**, and the need for functions like `expm1` and `logsumexp`.
    * *Link to project details:* [Project: Floating Point Quirks](/projects/floating-point-quirks/)

---

## 💾 Software Development Projects

* **Advanced Python Deep Dive**
    * Create a simple package demonstrating the use of **descriptors**, **weak references**, and a minimal example of a **metaclass**.
    * Write a function that leverages **closures** in a non-trivial way.
    * *Link to project details:* [Project: Advanced Python](/projects/advanced-python/)
* **Package Management and CI/CD**
    * Create a minimal Python project, package it using `setuptools` or Poetry, and publish it (even to a test PyPI server).
    * Set up a **GitHub Actions** workflow to run tests automatically (**CI/CD**).
    * *Link to project details:* [Project: Package and CI/CD](/projects/package-ci/)
* **Python Code Quality Setup**
    * Configure a project with **`pre-commit` hooks** using **Black**, **flake8**, and **isort** (or use ruff).
    * *Link to project details:* [Project: Code Quality Tools](/projects/code-quality-tools/)
* **C++ Integration with Python**
    * Write a simple numerical function in C++ demonstrating **pass-by-reference vs pass-by-copy** and **pointers**.
    * Write **Python bindings** for this C++ function using **pybind11** (and potentially LibTorch).
    * *(Optional)* Use **OpenMP** to parallelize the C++ function.
    * *Link to project details:* [Project: CXX Bindings](/projects/cxx-bindings/)
* **Julia and Multiple Dispatch**
    * Write a set of Julia functions demonstrating the power of **multiple dispatch** for numerical programs.
    * Write a simple Julia **macro** to demonstrate **homoiconicity**.
    * *Link to project details:* [Project: Julia and Dispatch](/projects/julia-dispatch/)
* **Functional Programming & Type Theory**
    * Study and implement simple functions in **Haskell** to understand **monads** and **referential transparency**.
    * Compare and contrast **sum types** and **union types**.
    * *Link to project details:* [Project: Functional Programming](/projects/functional-programming/)
* **Data Structures and Complexity**
    * Implement the **Fibonacci sequence** using **dynamic programming** (and compare it to **`functools.lru_cache`**).
    * Write a technical note explaining the **big-O complexity** of a Python `dict` (hash map) and the memory allocation trick for Python lists.
    * *Link to project details:* [Project: Complexity and DS](/projects/complexity-ds/)

---

## 📐 Mathematics Projects

* **Foundational Analysis**
    * Write a formal report on the concepts of **Convexity** and **Lipschitz continuity**, explaining their role in bounding functions and optimization.
    * *Link to project details:* [Project: Convex and Lipschitz](/projects/convex-lipschitz/)
* **Measure Theory and Probability**
    * Study the basics of **probability via measure theory**.
    * Study **integration via measure theory**, focusing on **Fubini’s theorem** and the **Leibniz Integral Rule**.
    * *(Optional)* Study **Radon–Nikodym derivatives** and their connection to **KL divergence** and **Importance Sampling**.
    * *Link to project details:* [Project: Measure Theory Basics](/projects/measure-theory/)
* **Topology and Core Analysis**
    * Summarize the core concepts of **open sets, closed sets, compactness, and continuous functions** in topology.
    * Review the basics of **Real Analysis** (epsilon-delta proofs) and **Functional Analysis** (e.g., the Weierstraß Approximation Theorem).
    * *Link to project details:* [Project: Analysis and Topology](/projects/analysis-topology/)
* **Differential Equations and Calculus**
    * Study **Ordinary Differential Equations**, focusing on **linearization around equilibria**.
    * Review and apply vector calculus concepts: **Div, Grad, and Curl**.
    * *Link to project details:* [Project: Applied Calculus](/projects/applied-calculus/)

---

## 📈 Statistics Projects

* **Regularisation Equivalence**
    * Demonstrate the equivalence between **Tikhonov/L2 regularisation** (regularized maximum likelihood) and **maximum a-posteriori (MAP)** estimation.
    * *Link to project details:* [Project: Regularisation Equivalence](/projects/regularisation/)
* **Variance Minimisation Techniques**
    * Implement and compare the effectiveness of **Antithetic sampling** and **Control Variates** on a simple Monte-Carlo integration problem.
    * Use **Importance Sampling** to estimate an integral that is difficult to sample from directly.
    * *Link to project details:* [Project: Variance Reduction](/projects/variance-reduction/)
* **Monte-Carlo Methods**
    * Implement a basic **Markov Chain Monte-Carlo (MCMC)** method (e.g., Metropolis-Hastings).
    * Research **Hamiltonian Monte-Carlo (HMC)** and its advantages.
    * *Link to project details:* [Project: Advanced Monte-Carlo](/projects/advanced-mc/)
* **High-Dimensional Intuition**
    * Study and report on the phenomenon of **Gaussian "soap bubbles" in high dimensions** and the concept of **"typical sets" in MCMC**.
    * *Link to project details:* [Project: High-D Intuition](/projects/high-d-intuition/)