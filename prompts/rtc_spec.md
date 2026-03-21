We propose the *gradient guidance vector*, which quantifies the guidance term that needs to be applied to the learned velocity field to match model predictions with ground truth actions.


## Formulation

In all notations, $\tau$ descends from 1 to 0. 

(Therefore $\widehat{A}_t^0$ is correct. This notation represents the final model output.)


For every step of inference, compute:

$$
v_{guidance}
= 
\left( A^* - \widehat{A}_t^0 \right)^\top
\frac{\partial \widehat{A}_t^0}{\partial A_t^\tau}
$$

$$
\text{where} \quad
\widehat{A}_t^0 = A_t^\tau - \tau\, \mathbf{v}(A_t^\tau, o_t, \tau)
$$



$v_{guidance}$ can be optionally scaled by
$\min\!\left(\beta, \frac{1 - \tau}{\tau \cdot r_\tau^2}\right)$ to account for variation across $\tau$, where $\beta$ is a hyperparameter to clip the infinity edges.

## Intuition

$A^* - \widehat{A}_t^0$: the error in the output space.

The Jacobian $\frac{\partial \widehat{A}_t^0}{\partial A_t^\tau}$: maps the error back to the action latent space.

## Appendix

**Replicating training settings** for the action gap

Sample $\tau \sim \mathrm{Beta}(1.5, 1.0) * 0.999 + 0.001$:

$$
A_t^\tau = \tau \epsilon + (1-\tau) A^*
$$

and run one model evaluation:

$$
\mathbf{v}\!\left(A_t^\tau, o_t, \tau\right)
$$


**Replicating inference settings** for the general gap


Set $n=\text{num\_steps}$, $\Delta = \frac{1}{n}$, initialize $A_t^1 = \epsilon$. For each step:

$$
A_t^{\tau-\Delta} = A_t^\tau - \Delta \,\mathbf{v} \left(A_t^\tau, o_t, \tau\right)
$$

Note: each step recomputes a one-step surrogate of the final sample. Think of it as computing 0.9->0, 0.8->0, 0.7->0. At inference step k, $A_t^\tau$ is the current rollout state carried from step k-1 (initialized only once at $A_t^1=\epsilon$).
