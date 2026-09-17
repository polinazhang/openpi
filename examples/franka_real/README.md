# Franka inference

See [repo-configs/README.md](../../repo-configs/README.md).
Set the absolute checkpoint directory in `repo-configs/config.py`. Its own
`assets/franka/norm_stats.json` is required; no alternate assets are used.
Task/prompt and evaluation name are also in `repo-configs/config.py`.
Metadata storage, camera settings and episode limits remain in
`examples/franka_real/config.py`.
