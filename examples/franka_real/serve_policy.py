import logging

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

from examples.franka_real import config as _runtime_config


_DEFAULT_POLICY = {
    _runtime_config.ModelFamily.PI0: (
        "pi0_franka_object",
        "gs://openpi-assets/checkpoints/pi0_base",
    ),
    _runtime_config.ModelFamily.PI05: (
        "pi05_franka_object",
        "gs://openpi-assets/checkpoints/pi05_base",
    ),
}


def main() -> None:
    cfg = _runtime_config.POLICY_SERVER

    config_name, default_checkpoint = _DEFAULT_POLICY[cfg.model_family]
    checkpoint_dir = cfg.checkpoint_dir or default_checkpoint

    policy = _policy_config.create_trained_policy(
        _config.get_config(config_name),
        checkpoint_dir,
        evaluation_suite_name=cfg.evaluation_suite_name,
        data_dir=cfg.data_dir,
        default_prompt=cfg.default_prompt,
    )

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=cfg.host,
        port=cfg.port,
        metadata=policy.metadata,
    )
    logging.info("Serving Franka policy '%s' from %s on %s:%s", config_name, checkpoint_dir, cfg.host, cfg.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
