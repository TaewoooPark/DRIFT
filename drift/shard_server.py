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

from . import crypto, protocol
from .engine_torch import TorchShardEngine

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Node:
    """Holds an optional engine and (re)configures it on demand.

    The engine (model forward + per-session KV) is serialized by ``self._lock``:
    concurrent sessions may connect, but only one forward runs at a time (the GPU
    serializes anyway). The lock is held around the *compute only* — the chain
    relay (network I/O to the next node) runs outside it, so one session's
    downstream hop overlaps the next session's compute.
    """

    def __init__(self, name: str, model_id: str, dtype: str, device: str):
        self.name = name
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        self.engine: TorchShardEngine | None = None
        self.tamper = False  # test hook: a dishonest node (see drift.verify)
        self._lock = threading.Lock()          # serializes engine compute
        self._down: dict = {}                  # (host,port,session) -> socket to the next hop
        self._down_lock = threading.Lock()     # guards the _down cache

    def configure(self, start: int, end: int, model_id: str | None = None,
                  dtype: str | None = None, device: str | None = None,
                  embed_duty: bool = False, head_duty: bool = False) -> dict:
        """Build + load the engine for a layer range (idempotent per range).

        embed_duty / head_duty give this node the thin-head edge modules
        (embed_tokens on the first node, norm + lm_head on the last)."""
        self.model_id = model_id or self.model_id
        self.dtype = dtype or self.dtype
        self.device = device or self.device
        with self._lock:  # loading mutates shared state; don't race a concurrent forward
            self.engine = TorchShardEngine(
                model_id=self.model_id, start_layer=start, end_layer=end,
                device=self.device, dtype=self.dtype, name=self.name,
                embed_duty=embed_duty, head_duty=head_duty,
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
                embed_duty=bool(msg.get("embed_duty")), head_duty=bool(msg.get("head_duty")),
            )
        if self.engine is None:
            return {"ok": False, "error": "node not configured — send a 'configure' message first"}
        if mtype == "reset":
            self.engine.reset(msg["session_id"])
            self._close_session_downsocks(msg["session_id"])  # tear down chain relays too
            return {"ok": True}
        if mtype in ("prefill", "decode"):
            with self._lock:  # serialize the forward; the relay below runs unlocked
                if msg.get("embed"):
                    # Thin-head entry: this node embeds the token ids itself, so the
                    # head sent no tensor — only ints crossed its boundary.
                    hidden = self.engine.embed(msg["input_ids"])
                else:
                    hidden = protocol.bytes_to_tensor(
                        msg["tensor"], msg["shape"], msg["dtype"], self.engine.device
                    )
                out = self.engine.forward(
                    session_id=msg["session_id"], hidden=hidden,
                    position_ids=msg.get("position_ids"), input_ids=msg.get("input_ids"),
                    mode=mtype,
                )
                if self.tamper:  # test hook: corrupt the output so drift.verify flags it
                    out = out * 1.2 + 0.1
                # Thin-head exit: the last node norms + heads + argmaxes, so only a
                # token id crosses back — the head does no tensor math.
                payload = ({"token": self.engine.head_argmax(out)} if self.engine.head_duty
                           else {"shape": list(out.shape), "dtype": self.engine.dtype,
                                 "tensor": protocol.tensor_to_bytes(out, self.engine.dtype)})
            # Chain mode (M7): relay straight to the next hop / collect sink instead
            # of returning. route/collect are optional — absent → classic star.
            if msg.get("route") is not None:
                return self._relay(msg, payload)
            return {"ok": True, "error": None, **payload}
        return {"ok": False, "error": f"unknown message type: {mtype}"}

    # ------------------------------------------------------------- chain relay
    def _downsock(self, target: tuple, session_id: str):
        """A cached client channel to a downstream node / the collect sink
        (encrypted when a network key is set).

        Keyed by (host, port, session) so concurrent sessions never interleave on
        one channel; a single session sends one message at a time, so no per-socket
        lock is needed — only the cache dict is guarded.
        """
        key = (target[0], int(target[1]), session_id)
        with self._down_lock:
            ch = self._down.get(key)
        if ch is None:
            ch = crypto.dial(target[0], int(target[1]))  # client handshake happens here
            with self._down_lock:
                self._down[key] = ch
        return ch

    def _drop_downsock(self, target: tuple, session_id: str) -> None:
        key = (target[0], int(target[1]), session_id)
        with self._down_lock:
            ch = self._down.pop(key, None)
        if ch is not None:
            ch.close()

    def _relay(self, msg: dict, payload: dict) -> dict:
        """Forward the result to the next node, or — if this is the tail (empty
        route) — to the head's collect sink. `payload` is a hidden-state tensor
        (normal / middle node) or a `{token}` (thin-head last node). Returns a tiny
        ack so a dropped downstream propagates back up the chain as an error."""
        route = msg["route"]
        target = route[0] if route else msg["collect"]
        down = {
            "type": msg["type"], "session_id": msg["session_id"], "seq_id": msg.get("seq_id"),
            "position_ids": msg.get("position_ids"), "input_ids": msg.get("input_ids"),
            "route": route[1:], "collect": msg["collect"],
            **payload,
        }
        tgt = (target[0], int(target[1]))
        try:
            ch = self._downsock(tgt, msg["session_id"])
            ch.send(down)
            ack = ch.recv()
        except (ConnectionError, OSError, ValueError) as e:
            self._drop_downsock(tgt, msg["session_id"])
            return {"ok": False, "error": f"relay to {tgt[0]}:{tgt[1]} failed: {e}"}
        if not ack.get("ok"):
            return {"ok": False, "error": f"downstream {tgt[0]}:{tgt[1]}: {ack.get('error')}"}
        return {"ok": True, "relayed": True, "error": None}

    def _close_session_downsocks(self, session_id: str) -> None:
        with self._down_lock:
            keys = [k for k in self._down if k[2] == session_id]
            for k in keys:
                ch = self._down.pop(k, None)
                if ch is not None:
                    ch.close()


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
    """Handle one client connection (one session) until it closes.

    Completes the server handshake first — encrypted when a network key is set,
    plaintext otherwise. A dialer that doesn't hold the key can't finish it, so
    the connection is dropped before any DRIFT message is processed.
    """
    try:
        ch = crypto.accept_wrap(conn)
    except (ConnectionError, OSError, ValueError):
        conn.close()
        return
    try:
        while True:
            try:
                msg = ch.recv()
            except (ConnectionError, OSError, ValueError):
                break
            try:
                reply = node.handle(msg)  # locks the compute internally; relay runs unlocked
            except Exception as e:  # surface engine errors to the caller
                reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            try:
                ch.send(reply)
            except (ConnectionError, OSError):
                break
    finally:
        ch.close()


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
    ap.add_argument("--tamper", action="store_true",
                    help="TEST ONLY: corrupt this node's output so `drift.verify` flags it")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    port = args.port or int(os.environ.get("DRIFT_PORT", cfg["port"]))
    device = args.device or cfg.get("device", "cpu")
    name = args.name or "shard"
    node = Node(name=name, model_id=cfg["model_id"], dtype=cfg.get("dtype", "float16"),
                device=device)
    node.tamper = args.tamper

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
