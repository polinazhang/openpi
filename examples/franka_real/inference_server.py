"""Standalone OpenPI inference server for Franka using a minimal HTTP API.

Run this script from the OpenPI uv environment as:
    python /path/to/openpi/examples/franka_real/inference_server.py
"""

from __future__ import annotations

import http.server
import json
import logging
import pathlib
import pickle
import sys
import threading
import time
import traceback
from typing import Any


_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import config as _runtime_config
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


logger = logging.getLogger(__name__)

_DEFAULT_POLICY = {
    _runtime_config.ModelFamily.PI0: (
        "pi0_franka_object",
    ),
    _runtime_config.ModelFamily.PI05: (
        "pi05_franka_object",
    ),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


class _InferenceHandler(http.server.BaseHTTPRequestHandler):
    policy = None
    policy_lock = threading.Lock()
    metadata: dict[str, Any] | None = None
    cfg = _runtime_config.POLICY_SERVER
    current_evaluation_suite_name = _runtime_config.POLICY_SERVER.evaluation_suite_name

    def _write_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_binary(self, code: int, payload: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> bytes:
        content_length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(content_length)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == self.cfg.health_path:
            self._write_json(
                200,
                {
                    "ok": True,
                    "model_family": self.cfg.model_family.value,
                    "checkpoint_dir": self.cfg.checkpoint_dir,
                    "norm_stats_path": self.cfg.norm_stats_path,
                    "data_dir": self.cfg.data_dir,
                    "evaluation_suite_name": self.current_evaluation_suite_name,
                },
            )
            return

        if self.path == self.cfg.metadata_path:
            payload = pickle.dumps(self.metadata, protocol=pickle.HIGHEST_PROTOCOL)
            self._write_binary(200, payload)
            return

        self._write_json(404, {"ok": False, "error": f"unknown path: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == self.cfg.begin_episode_path:
            try:
                body = self._read_body()
                payload = json.loads(body.decode("utf-8")) if body else {}
                evaluation_suite_name = payload.get("evaluation_suite_name")
                if not isinstance(evaluation_suite_name, str) or not evaluation_suite_name.strip():
                    raise ValueError("begin_episode requires non-empty string 'evaluation_suite_name'")

                with self.policy_lock:
                    if hasattr(self.policy, "_metadata_logger"):
                        from openpi.policies.metadata_logger import MetadataLogger

                        self.policy._metadata_logger = MetadataLogger(self.cfg.data_dir, evaluation_suite_name)
                    self.current_evaluation_suite_name = evaluation_suite_name
                logger.info("Episode metadata suite set to: %s", evaluation_suite_name)
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "evaluation_suite_name": evaluation_suite_name,
                    },
                )
            except Exception as exc:
                self._write_json(400, {"ok": False, "error": str(exc)})
            return

        if self.path == self.cfg.end_trajectory_path:
            try:
                with self.policy_lock:
                    if hasattr(self.policy, "end_trajectory"):
                        self.policy.end_trajectory()
                self._write_json(200, {"ok": True})
            except Exception as exc:
                logger.exception("end_trajectory failed")
                self._write_json(500, {"ok": False, "error": str(exc), "traceback": traceback.format_exc()})
            return

        if self.path != self.cfg.infer_path:
            self._write_json(404, {"ok": False, "error": f"unknown path: {self.path}"})
            return

        try:
            req = pickle.loads(self._read_body())
            if not isinstance(req, dict):
                raise TypeError(f"expected dict request payload, got {type(req).__name__}")
            if "observation" not in req:
                raise KeyError("request payload missing required key: 'observation'")

            obs = req["observation"]
            request_id = req.get("request_id")

            t0 = time.monotonic()
            with self.policy_lock:
                action = self.policy.infer(obs)
            infer_ms = (time.monotonic() - t0) * 1000.0

            resp = {
                "ok": True,
                "request_id": request_id,
                "action": action,
                "server_timing": {"infer_ms": infer_ms},
            }
            self._write_binary(200, pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as exc:
            logger.exception("Inference request failed")
            self._write_json(
                500,
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    cfg = _runtime_config.POLICY_SERVER

    config_name = _DEFAULT_POLICY[cfg.model_family][0]
    if not cfg.checkpoint_dir:
        raise ValueError("PolicyServerConfig.checkpoint_dir must be set. No default checkpoint fallback is used.")
    if not cfg.norm_stats_path:
        raise ValueError("PolicyServerConfig.norm_stats_path must be set.")
    checkpoint_dir = cfg.checkpoint_dir
    norm_stats_path = pathlib.Path(cfg.norm_stats_path)
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {norm_stats_path}")

    logger.info(
        "Loading Franka policy '%s' from %s (suite=%s, data_dir=%s, norm_stats_path=%s)",
        config_name,
        checkpoint_dir,
        cfg.evaluation_suite_name,
        cfg.data_dir,
        norm_stats_path,
    )
    norm_stats = _normalize.deserialize_json(norm_stats_path.read_text())

    policy = _policy_config.create_trained_policy(
        _config.get_config(config_name),
        checkpoint_dir,
        evaluation_suite_name=cfg.evaluation_suite_name,
        data_dir=cfg.data_dir,
        default_prompt=cfg.default_prompt,
        norm_stats=norm_stats,
    )

    _InferenceHandler.policy = policy
    _InferenceHandler.metadata = policy.metadata
    _InferenceHandler.cfg = cfg
    _InferenceHandler.current_evaluation_suite_name = cfg.evaluation_suite_name

    logger.info("Policy metadata: %s", _json_safe(policy.metadata))
    logger.info("Starting inference HTTP server on %s:%s", cfg.host, cfg.port)
    logger.info(
        "Endpoints: %s %s %s %s %s",
        cfg.health_path,
        cfg.metadata_path,
        cfg.infer_path,
        cfg.begin_episode_path,
        cfg.end_trajectory_path,
    )
    httpd = http.server.ThreadingHTTPServer((cfg.host, cfg.port), _InferenceHandler)
    httpd.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
