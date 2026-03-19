

## Mathematical Meaning (Flow-Matching)

- target residual: `u_t = ε - A_t`
- optimal scale:
  - `α^(l) = <v_t^(l), u_t> / (||v_t^(l)||^2 + ε)`
- EDR:
  - `EDR^(l) = || α^(l) v_t^(l) - u_t ||_2`
- cosine:
  - `cosine^(l) = <v_t^(l), u_t> / (||v_t^(l)|| * ||u_t|| + ε)`

OpenPI implementation:
- metric math (`dot`, `norm`, `scale`, `scaled residual`, cosine): `openarm/static_inference.py:191-203`
- final layer loss: `torch.linalg.norm(final_prediction - target, dim=-1)` at `openarm/static_inference.py:374`
