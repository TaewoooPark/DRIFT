"""DRIFT shard server — TCP listen → recv_msg → engine → send_msg (spec §7).

Two ways to run:

  * **pre-assigned** — `--start/--end` given: the layer range is fixed at launch
    (used by the parity gate, the benchmark, and the manual multi-terminal flow).
  * **unassigned (fungible)** — no range given: the node starts empty and the
    orchestrator pushes its range with a `configure` message. This is what
    `drift node` / `drift up` use so the user never hand-writes layer ranges.

Sequential, single-session-at-a-time (concurrency is later). Identity comes from
config.yaml but CLI flags + the DRIFT_PORT env override it.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading

import yaml

from . import protocol
from .engine_torch import TorchShardEngine

# The engine (model forward + per-session KV) is serialized: concurrent sessions
# may connect, but only one forward runs at a time (the GPU serializes anyway).
# What this buys is overlap — while one session's reply travels the network, the
# next session computes — which is the throughput win on a network-bound link.
_ENGINE_LOCK = threading.Lock()


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Node:
    """Holds an optional engine and (re)configures it on demand."""

    def __init__(self, name: str, model_id: str, dtype: str, device: str):
        self.name = name
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        self.engine: TorchShardEngine | None = None

    def configure(self, start: int, end: int, model_id: str | None = None,
                  dtype: str | None = None, device: str | None = None) -> dict:
        """Build + load the engine for a layer range (idempotent per range)."""
        self.model_id = model_id or self.model_id
        self.dtype = dtype or self.dtype
        self.device = device or self.device
        self.engine = TorchShardEngine(
            model_id=self.model_id, start_layer=start, end_layer=end,
            device=self.device, dtype=self.dtype, name=self.name,
        )
        self.engine.load()
        return {"ok": True, **self.engine.ping_info()}

    def handle(self, msg: dict) -> dict:
        mtype = msg.get("type")
        if mtype == "ping":
            from .common import env_info

            if self.engine is not None:
                return {"ok": True, "assigned": True, **self.engine.ping_info(), **env_info()}
            return {"ok": True, "assigned": False, "name": self.name,
                    "device": self.device, "loaded": False, **env_info()}
        if mtype == "configure":
            return self.configure(
                start=msg["start_layer"], end=msg["end_layer"],
                model_id=msg.get("model_id"), dtype=msg.get("dtype"),
                device=msg.get("device"),
            )
        if self.engine is None:
            return {"ok": False, "error": "node not configured — send a 'configure' message first"}
        if mtype == "reset":
            self.engine.reset(msg["session_id"])
            return {"ok": True}
        if mtype in ("prefill", "decode"):
            hidden = protocol.bytes_to_tensor(
                msg["tensor"], msg["shape"], msg["dtype"], self.engine.device
            )
            out = self.engine.forward(
                session_id=msg["session_id"], hidden=hidden,
                position_ids=msg.get("position_ids"), input_ids=msg.get("input_ids"),
                mode=mtype,
            )
            return {"ok": True, "shape": list(out.shape), "dtype": self.engine.dtype,
                    "tensor": protocol.tensor_to_bytes(out, self.engine.dtype), "error": None}
        return {"ok": False, "error": f"unknown message type: {mtype}"}


def serve(node: Node, host: str, port: int, banner: str | None = None) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    if banner is not None:
        print(banner, flush=True)
    else:
        assigned = "unassigned (waiting for configure)" if node.engine is None \
            else f"layers [{node.engine.start_layer}:{node.engine.end_layer})"
        print(f"[shard {node.name}] listening on {host}:{port} — {assigned} · "
              f"device={node.device}", flush=True)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=_serve_conn, args=(node, conn), daemon=True).start()


def _serve_conn(node: Node, conn: socket.socket) -> None:
    """Handle one client connection (one session) until it closes."""
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            try:
                msg = protocol.recv_msg(conn)
            except ConnectionError:
                break
            try:
                with _ENGINE_LOCK:  # serialize the forward; overlap the network
                    reply = node.handle(msg)
            except Exception as e:  # surface engine errors to the caller
                reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            try:
                protocol.send_msg(conn, reply)
            except (ConnectionError, OSError):
                break
    finally:
        conn.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DRIFT shard server")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--name")
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--device")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int)
    ap.add_argument("--preload", action="store_true", help="load weights before serving")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    port = args.port or int(os.environ.get("DRIFT_PORT", cfg["port"]))
    device = args.device or cfg.get("device", "cpu")
    name = args.name or "shard"
    node = Node(name=name, model_id=cfg["model_id"], dtype=cfg.get("dtype", "float16"),
                device=device)

    # Pre-assigned mode: a fixed range at launch (parity gate / bench / manual).
    if args.start is not None and args.end is not None:
        if args.preload:
            node.configure(start=args.start, end=args.end)
        else:
            node.engine = TorchShardEngine(
                model_id=node.model_id, start_layer=args.start, end_layer=args.end,
                device=device, dtype=node.dtype, name=name,
            )
    try:
        serve(node, args.host, port)
    except KeyboardInterrupt:
        print(f"\n[shard {name}] shutting down", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
