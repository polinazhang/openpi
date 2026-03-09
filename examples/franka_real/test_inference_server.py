"""Basic responsiveness test for the Franka HTTP inference server.

Run in a second terminal while the server is running:
    python /path/to/openpi/examples/franka_real/test_inference_server.py
"""

from __future__ import annotations

import argparse
import pickle
import time
from typing import Any

import numpy as np
import requests


def _build_test_observation(image_size: int = 224) -> dict[str, Any]:
    return {
        "head_image": np.random.randint(0, 256, size=(image_size, image_size, 3), dtype=np.uint8),
        "left_wrist_image": np.random.randint(0, 256, size=(image_size, image_size, 3), dtype=np.uint8),
        "right_wrist_image": np.random.randint(0, 256, size=(image_size, image_size, 3), dtype=np.uint8),
        "state": np.zeros((8,), dtype=np.float32),
        "prompt": "",
    }


def _get_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--infer-path", default="/infer")
    parser.add_argument("--health-path", default="/health")
    parser.add_argument("--num-requests", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    health_url = _get_url(args.host, args.port, args.health_path)
    infer_url = _get_url(args.host, args.port, args.infer_path)

    print(f"[test] health check -> {health_url}")
    health_resp = requests.get(health_url, timeout=args.timeout)
    health_resp.raise_for_status()
    print(f"[test] health OK: {health_resp.json()}")

    latencies_ms: list[float] = []
    for idx in range(args.num_requests):
        req = {
            "request_id": idx,
            "observation": _build_test_observation(args.image_size),
        }
        payload = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)

        t0 = time.monotonic()
        resp = requests.post(
            infer_url,
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=args.timeout,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        resp.raise_for_status()
        data = pickle.loads(resp.content)

        if not data.get("ok", False):
            raise RuntimeError(f"inference request failed: {data}")
        if "action" not in data:
            raise RuntimeError(f"missing action in response: {data}")

        action = data["action"]
        actions = action["actions"] if isinstance(action, dict) and "actions" in action else action
        actions_np = np.asarray(actions)
        if actions_np.ndim == 1:
            action_dim = actions_np.shape[0]
        elif actions_np.ndim >= 2:
            action_dim = actions_np.shape[-1]
        else:
            raise RuntimeError(f"invalid action array shape: {actions_np.shape}")

        print(
            f"[test] request {idx}: HTTP {resp.status_code}, "
            f"latency={elapsed_ms:.1f}ms, action_shape={actions_np.shape}, action_dim={action_dim}, "
            f"server_timing={data.get('server_timing', {})}"
        )

    arr = np.asarray(latencies_ms, dtype=np.float32)
    print(
        "[test] completed: "
        f"num_requests={len(latencies_ms)}, mean={float(arr.mean()):.1f}ms, "
        f"p50={float(np.quantile(arr, 0.5)):.1f}ms, p95={float(np.quantile(arr, 0.95)):.1f}ms"
    )


if __name__ == "__main__":
    main()
