## Symbols and VLA forward function

Let the vision-language-action model be denoted by

$$
\hat{A}^0 = \Phi_\theta(A_t^\tau, h_v, h_\ell, h_s, h_t),
$$

where $\theta$ are the model parameters and $\hat{A}^0$ is the predicted clean action (or equivalent action-space output of the policy head).

The model takes the following inputs:

$$
h_v = f_{\mathrm{vision}}(x_v), \qquad
h_\ell = f_{\mathrm{lang}}(x_\ell), \qquad
h_s = f_{\mathrm{state}}(x_s), \qquad
h_t = f_{\mathrm{time}}(\tau),
$$

where $x_v$ is the visual observation, $x_\ell$ is the language input, $x_s$ is the robot state, and $\tau \in (0,1]$ is the diffusion time.

The noisy action latent is defined as

$$
A_t^\tau = \tau \epsilon + (1-\tau) A^*,
$$

where $\epsilon \sim \mathcal{N}(0,I)$ and $A^*$ is the ground-truth action.

Let the training or analysis loss be

$$
L = \mathcal{L}\big(\Phi_\theta(A_t^\tau, h_v, h_\ell, h_s, h_t), A^*\big).
$$

Throughout, all inputs except the variable under analysis are held fixed.

## Approach 1: local gradient norm on the target variable

As an example, we consider the vision embedding and compute the gradient of the loss with respect to $h_v$ to measure local sensitivity:

$$
g_v := \nabla_{h_v} L.
$$

The scalar sensitivity score is defined as the gradient norm

$$
S_v := \|g_v\|_2.
$$

In practice, for each evaluation example, one performs a forward pass to compute $L$, backpropagates through the model while treating $h_v$ as the differentiation target, and records $\| \nabla_{h_v} L \|_2$.

This quantity measures the first-order sensitivity of the loss to infinitesimal perturbations of the vision embedding at the current point.

## Approach 2: multi-step gradient-based perturbation to estimate the minimum reachable loss

As an example, we again consider the vision embedding. The goal is to estimate how far the loss can be reduced by perturbing only the vision embedding within $N$ optimization steps, where $N$ is a hyperparameter.

Define a perturbation variable $\delta_v \in \mathbb{R}^{\dim(h_v)}$ and initialize

$$
\delta_v^{(0)} = 0.
$$

The objective is

$$
L_v^{\min}(N) := \min_{\delta_v \in \mathbb{R}^{\dim(h_v)}} \mathcal{L}\big(\Phi_\theta(A_t^\tau, h_v + \delta_v, h_\ell, h_s, h_t), A^*\big),
$$

approximated by $N$ steps of gradient descent on $\delta_v$.

At step $k \in \{0,\dots,N-1\}$, compute

$$
L^{(k)} := \mathcal{L}\big(\Phi_\theta(A_t^\tau, h_v + \delta_v^{(k)}, h_\ell, h_s, h_t), A^*\big),
$$

and update

$$
\delta_v^{(k+1)} = \delta_v^{(k)} - \eta \nabla_{\delta_v^{(k)}} L^{(k)},
$$

where $\eta > 0$ is the step size (another hyperparameter).

The estimated minimum reachable loss after $N$ steps is

$$
\widehat{L}_v^{\min}(N) := \min_{k \in \{0,\dots,N\}} L^{(k)}.
$$

Equivalently, one may report the final-step value

$$
\widehat{L}_{v,\mathrm{final}}(N) := L^{(N)}.
$$

This procedure does not claim a global optimum. It provides an optimization-based estimate of the minimum loss reachable by perturbing only the vision embedding under a fixed computational budget of $N$ gradient steps.


### Equivalence of $\nabla_{\delta_v} L(0)$ and $\nabla_{h_v} L$

We note that $\nabla_{\delta_v} L(0) = \nabla_{h_v} L$ since $\delta_v$ is an additive reparameterization of $h_v$.

