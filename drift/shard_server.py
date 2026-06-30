"""DRIFT shard server — TCP listen → recv_msg → engine → send_msg (spec §7).

Sequential, single-session-at-a-time (concurrency is later, spec §13). Identity
comes from config.yaml but CLI flags + the DRIFT_PORT env override it for
localhost multi-port runs (docs/06).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

import yaml

from . import protocol
from .engine_torch import TorchShardEngine


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _handle(engine: TorchShardEngine, msg: dict) -> dict:
    mtype = msg.get("type")
    if mtype == "ping":
        info = engine.ping_info()
        return {"ok": True, **info}
    if mtype == "reset":
        engine.reset(msg["session_id"])
        return {"ok": True}
    if mtype in ("prefill", "decode"):
        hidden = protocol.bytes_to_tensor(
            msg["tensor"], msg["shape"], msg["dtype"], engine.device
        )
        out = engine.forward(
            session_id=msg["session_id"],
            hidden=hidden,
            position_ids=msg.get("position_ids"),
            input_ids=msg.get("input_ids"),
            mode=mtype,
        )
        return {
            "ok": True,
            "shape": list(out.shape),
            "dtype": engine.dtype,
            "tensor": protocol.tensor_to_bytes(out, engine.dtype),
            "error": None,
        }
    return {"ok": False, "error": f"unknown message type: {mtype}"}


def serve(engine: TorchShardEngine, host: str, port: int, preload: bool) -> None:
    if preload:
        engine.load()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    info = engine.ping_info()
    print(
        f"[shard {info['name']}] listening on {host}:{port} — "
        f"layers [{info['start_layer']}:{info['end_layer']}) · device={info['device']} · "
        f"preloaded={preload}",
        flush=True,
    )
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                try:
                    msg = protocol.recv_msg(conn)
                except ConnectionError:
                    break
                try:
                    reply = _handle(engine, msg)
                except Exception as e:  # surface engine errors to the caller
                    reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                protocol.send_msg(conn, reply)
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
    start = args.start if args.start is not None else cfg["shards"][0]["start_layer"]
    end = args.end if args.end is not None else cfg["shards"][0]["end_layer"]
    device = args.device or cfg.get("device", "cpu")
    name = args.name or "shard"

    engine = TorchShardEngine(
        model_id=cfg["model_id"],
        start_layer=start,
        end_layer=end,
        device=device,
        dtype=cfg.get("dtype", "float16"),
        name=name,
    )
    try:
        serve(engine, args.host, port, preload=args.preload)
    except KeyboardInterrupt:
        print(f"\n[shard {name}] shutting down", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
