We propose the *gradient guidance vector*, which quantifies the guidance term that needs to be applied to the learned velocity field to match model predictions with ground truth actions.

## Formulation

In all notations, $\tau$ descends from $1$ to $0$.

(Therefore $\widehat{A}_t^0$ is correct. This notation represents the final model output.)


For every step of inference, compute

$$
v_{\mathrm{guidance}} = (A^* - \widehat{A}_t^0)^\top
\frac{\partial \widehat{A}_t^0}{\partial A_t^\tau}
$$

where

$$
\widehat{A}_t^0 = A_t^\tau - \tau\, v(A_t^\tau, o_t, \tau)
$$

Optionally, scale $v_{\mathrm{guidance}}$ by

$$
\min(\beta, \frac{1 - \tau}{\tau \cdot r_\tau^2})
$$

to account for variation across $\tau$, where $\beta$ is a hyperparameter to clip the infinity edges.

## Intuition

$A^* - \widehat{A}_t^0$: the error in the output space.

The Jacobian $\frac{\partial \widehat{A}_t^0}{\partial A_t^\tau}$: maps that error back to the action latent space.

## Appendix

**Replicating training settings** for the action gap

Sample

$$
\tau \sim \mathrm{Beta}(1.5, 1.0) \cdot 0.999 + 0.001
$$


$$
A_t^\tau = \tau \epsilon + (1-\tau) A^*
$$

and run one model evaluation:

$$
v(A_t^\tau, o_t, \tau)
$$

**Replicating inference settings** for the general gap

Set $n$ to the number of steps, let $\Delta = \frac{1}{n}$, and initialize $A_t^1 = \epsilon$. For each step,

$$
A_t^{\tau-\Delta} = A_t^\tau - \Delta\, v(A_t^\tau, o_t, \tau)
$$

Note: each step recomputes a one-step surrogate of the final sample. Think of it as computing $0.9 \to 0$, $0.8 \to 0$, $0.7 \to 0$. At inference step $k$, $A_t^\tau$ is the current rollout state carried from step $k-1$ (initialized only once at $A_t^1=\epsilon$).