import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent

from examples.franka_real import config as _config
from examples.franka_real import env as _env


def main() -> None:
    runtime_cfg = _config.ROBOT_RUNTIME

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=runtime_cfg.policy_host,
        port=runtime_cfg.policy_port,
    )
    logging.info("Server metadata: %s", ws_client_policy.get_server_metadata())

    runtime = _runtime.Runtime(
        environment=_env.FrankaRealEnvironment(runtime_cfg),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=runtime_cfg.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=runtime_cfg.max_hz,
        num_episodes=runtime_cfg.num_episodes,
        max_episode_steps=runtime_cfg.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
