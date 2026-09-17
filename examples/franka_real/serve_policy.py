import logging

from openpi.serving import websocket_policy_server

from examples.franka_real import config as _runtime_config


def main() -> None:
    cfg = _runtime_config.POLICY_SERVER

    from examples.franka_real.checkpoint_policy import load_policy
    policy = load_policy(cfg)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=cfg.host,
        port=cfg.port,
        metadata=policy.metadata,
    )
    logging.info("Serving Franka policy '%s' from %s on %s:%s", "pi05_franka_cartesian", cfg.checkpoint_dir, cfg.host, cfg.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
