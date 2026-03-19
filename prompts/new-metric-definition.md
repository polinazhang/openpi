This metric, **gradient guidance vector**, quantifies the guidance term that needs to be applied to the learned velocity field to encourage the final generation to match ground truth actions. It is computed in the static inference mode, which is separate from the VLA model’s normal training and deployment-time inference.

For context:

**Static inference** is an analysis-only mode. Given demonstration samples, the VLA takes the state and other inputs, runs a “fake” forward denoising process, and does not execute the resulting actions. The generated latents are then compared against the ground-truth actions from the demonstration trajectories.

**Static inference metrics** measure the discrepancy between the direction predicted by the model and the ground-truth direction it is expected to follow. Examples include cosine similarity and EDR, defined in `prompts/edr-cosine.md`.

**condition-training** refers to static inference under training-time conditions, where the model starts from x_t = tau*noise + (1-tau)*actions, with sampled tau, and runs one step.

**condition-inference** refers to static inference under inference-time conditions, where the model starts from x_t = noise, and runs num_steps matching the model inference settings.

## Mathematical Formulation

In all notations, $\tau$ descends from 1 to 0.

**condition-training**

Sample $\tau \sim \mathrm{Beta}(1.5, 1.0) * 0.999 + 0.001$:

$$
A_t^\tau = \tau \epsilon + (1-\tau) A^*
$$

and run one model evaluation:

$$
\mathbf{v}\!\left(A_t^\tau, o_t, \tau\right)
$$


**condition-inference**

Note: each step recomputes a one-step surrogate of the final sample. Think of it as computing 0.9->0, 0.8->0, 0.7->0. At inference step k, $A_t^\tau$ is the current rollout state carried from step k-1 (initialized only once at $A_t^1=\epsilon$).

Set $n=\text{num\_steps}$, $\Delta = \frac{1}{n}$, initialize $A_t^1 = \epsilon$. For each step:

$$
A_t^{\tau-\Delta} = A_t^\tau - \Delta \,\mathbf{v} \left(A_t^\tau, o_t, \tau\right)
$$

**Metric Calculation**

During each step of those inferences (1 for condition-training and $n$ for condition-inference), compute:

$$
v_{guidance}
= \min\!\left(\beta, \frac{1 - \tau}{\tau \cdot r_\tau^2}\right)
\left( A^* - \widehat{A}_t^0 \right)^\top
\frac{\partial \widehat{A}_t^0}{\partial A_t^\tau}
$$

$$
\text{where} \quad
\widehat{A}_t^0 = A_t^\tau - \tau\, \mathbf{v}(A_t^\tau, o_t, \tau)
$$

$$
r_\tau^2 = \frac{(1 - \tau)^2}{\tau^2 + (1 - \tau)^2}
$$

$\beta$ is a parameter we set.

$A^*$ refers to the ground truth action.

$A_t^\tau$ is another notation for $x_t$ referred in other files and the openpi code.
