## Mathematical Definition of Cosine

Under the context of flow matching

- target residual: `u_t = ε - A*`
- cosine:
  - `cosine^(l) = <v_t^(l), u_t> / (||v_t^(l)|| * ||u_t|| + ε)`

The layer index l is retained in notation only for optional per-layer analyses; for pi05 implementation, you should only consider the last layer where `v_t^(l)` is identically the model's velocity prediction used in the flow-matching loss. 