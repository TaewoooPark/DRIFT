"""`drift up` / `drift run` — the head node.

Discovers or spawns worker nodes, reads the model's layer count, splits it across
the nodes automatically, pushes each node its range (`configure`), then drives the
decode loop. The user supplies a prompt, not a topology.

This module grows across releases: `up` (localhost, one command) lands first;
`run` (cross-machine, streaming REPL, discovery) builds on the same core.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from .common import (free_port, load_config, model_num_layers, pick_device,
                     split_layers)
from .orchestrator import HeadModel, Orchestrator, SocketTransport


def _wait_ready(transport: SocketTransport, names: list[str], timeout: float = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if all(transport.ping(n).get("ok") for n in names):
                return True
        except Exception:
            for sk in transport.socks.values():
                try:
                    sk.close()
                except Exception:
                    pass
            transport.socks.clear()
            time.sleep(1.0)
    return False


def build_over_nodes(model_id: str, dtype: str, head_device: str,
                     endpoints: list[dict]) -> Orchestrator:
    """Given node endpoints [{name,host,port,device?}], auto-split the model,
    configure each node with its range, and return a ready Orchestrator."""
    n_layers = model_num_layers(model_id)
    ranges = split_layers(n_layers, len(endpoints))
    shards = [dict(e) for e in endpoints]
    names = [s["name"] for s in shards]
    transport = SocketTransport(shards, dtype, head_device)
    if not _wait_ready(transport, names):
        raise RuntimeError("nodes did not become reachable — check `drift doctor --nodes …`")
    for s, (a, b) in zip(shards, ranges):
        info = transport.configure(s["name"], a, b, model_id, dtype, s.get("device"))
        print(f"  {s['name']} @ {s['host']}:{s['port']} → layers [{a}:{b}) "
              f"on {info.get('device')}", flush=True)
    head = HeadModel(model_id, head_device, dtype)
    return Orchestrator(head, transport, names, head_device)


def up_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="drift up",
        description="localhost: spawn N nodes, auto-split the model, and generate")
    ap.add_argument("n", nargs="?", type=int, default=2, help="number of local nodes (default 2)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--prompt")
    ap.add_argument("--max-new-tokens", type=int)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    model_id = cfg["model_id"]
    dtype = cfg.get("dtype", "float16")
    head_device = pick_device(cfg.get("device"))
    prompt = args.prompt or cfg["generation"]["prompt"]
    n_new = args.max_new_tokens or cfg["generation"]["max_new_tokens"]

    ports = [free_port() for _ in range(args.n)]
    print(f"[up] launching {args.n} local nodes on ports {ports} (device={head_device}) …",
          flush=True)
    procs = [subprocess.Popen(
        [sys.executable, "-m", "drift.node", "--port", str(p), "--host", "127.0.0.1", "--quiet"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for p in ports]
    try:
        endpoints = [{"name": f"n{i}", "host": "127.0.0.1", "port": p}
                     for i, p in enumerate(ports)]
        orch = build_over_nodes(model_id, dtype, head_device, endpoints)
        print("\n[up] generating …\n", flush=True)
        out = orch.generate(prompt, n_new, stop_on_eos=True)
        print(out["text"], flush=True)
    finally:
        for pr in procs:
            pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=10)
            except Exception:
                pr.kill()
    return 0


if __name__ == "__main__":
    sys.exit(up_main())
