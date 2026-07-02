"""`drift up` / `drift run` — the head node.

Discovers or spawns worker nodes, reads the model's layer count, splits it across
the nodes automatically, pushes each node its range (`configure`), then drives the
decode loop — streaming tokens as they land.

  drift up N                 localhost: spawn N nodes, auto-split, chat/generate
  drift run --nodes a:1,b:2  point at running `drift node`s (any machines)
  drift run                  use config.yaml shards (or zeroconf discovery)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from .common import (free_port, load_config, model_num_layers, pick_device,
                     split_layers)
from .orchestrator import HeadModel, Orchestrator, SocketTransport


# ------------------------------------------------------------------- assembly
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


def _check_env(transport: SocketTransport, names: list[str]) -> None:
    """Catch cross-machine parity hazards before assigning layers.

    Endianness is a hard stop (the fp16 wire bytes are native-endian); torch /
    transformers version skew only warns (mask/RoPE internals can shift and break
    bitwise parity, but the run may still be usable under the relaxed gate).
    """
    from .common import env_info

    local = env_info()
    for n in names:
        info = transport.ping(n)
        remote_endian = info.get("endian")
        if remote_endian and remote_endian != local["endian"]:
            raise RuntimeError(
                f"node {n} is {remote_endian}-endian but the head is {local['endian']}-endian; "
                "the fp16 hidden-state bytes are native-endian on the wire and would be misread"
            )
        for key in ("torch", "transformers"):
            rv = info.get(key)
            if rv and rv != local[key]:
                print(f"[warn] node {n} {key}={rv} != head {key}={local[key]} — mask/RoPE "
                      f"internals can differ across versions and break bitwise parity", flush=True)


def build_over_nodes(model_id: str, dtype: str, head_device: str,
                     endpoints: list[dict]) -> tuple[Orchestrator, list[dict]]:
    """Endpoints [{name,host,port,device?}] → auto-split, configure, return
    (orchestrator, plan). `plan` is per-node {name,host,port,start,end,device}."""
    n_layers = model_num_layers(model_id)
    ranges = split_layers(n_layers, len(endpoints))
    shards = [dict(e) for e in endpoints]
    names = [s["name"] for s in shards]
    transport = SocketTransport(shards, dtype, head_device)
    if not _wait_ready(transport, names):
        raise RuntimeError("nodes not reachable — try `drift doctor --nodes <host:port,…>`")
    _check_env(transport, names)
    plan = []
    for s, (a, b) in zip(shards, ranges):
        info = transport.configure(s["name"], a, b, model_id, dtype, s.get("device"))
        plan.append({**s, "start": a, "end": b, "device": info.get("device")})
    head = HeadModel(model_id, head_device, dtype, sliced=True)
    return Orchestrator(head, transport, names, head_device), plan


def _status_bar(model_id: str, plan: list[dict], head_device: str) -> None:
    print(f"\n  model : {model_id}")
    print(f"  head  : embed + norm + lm_head  · device={head_device}")
    for s in plan:
        print(f"  node  : {s['host']}:{s['port']}  layers [{s['start']}:{s['end']})"
              f"  · device={s['device']}")
    print()


def _stream(orch: Orchestrator, prompt: str, n_new: int) -> None:
    for piece in orch.generate_stream(prompt, n_new, stop_on_eos=True):
        print(piece, end="", flush=True)
    print(flush=True)


def _repl(orch: Orchestrator, n_new: int) -> None:
    print("chat ready — type a prompt (empty line or Ctrl-D to quit)\n")
    turn = 0
    while True:
        try:
            prompt = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt or prompt in ("exit", "quit", ":q"):
            break
        turn += 1
        print("drift › ", end="", flush=True)
        try:
            for piece in orch.generate_stream(prompt, n_new, stop_on_eos=True,
                                              session_id=f"repl-{turn}"):
                print(piece, end="", flush=True)
        except Exception as e:
            print(f"\n[error] {type(e).__name__}: {e}", flush=True)
        print("\n")


# ----------------------------------------------------------------- entrypoints
def _parse_nodes(spec: str) -> list[dict]:
    out = []
    for i, tok in enumerate(spec.split(",")):
        host, _, port = tok.strip().rpartition(":")
        if not port:
            raise SystemExit(f"--nodes entries must be host:port (got {tok!r})")
        out.append({"name": f"n{i}", "host": host or "127.0.0.1", "port": int(port)})
    return out


def _select_endpoints(args, cfg: dict) -> list[dict]:
    """Pick worker endpoints: explicit --nodes → LAN discovery → config shards."""
    if args.nodes:
        return _parse_nodes(args.nodes)
    if not args.no_discover:
        from . import discovery
        if discovery.HAVE_ZEROCONF:
            print(f"[run] discovering nodes on the LAN ({args.discover_timeout:.0f}s) …", flush=True)
            found = discovery.discover(timeout=args.discover_timeout)
            if found:
                eps = [{"name": f"n{i}", "host": f["host"], "port": f["port"],
                        "device": f.get("device")} for i, f in enumerate(found)]
                print("[run] found " + ", ".join(
                    f"{e['host']}:{e['port']}({e.get('device')})" for e in eps), flush=True)
                return eps
            print("[run] none found via discovery — falling back to config.yaml", flush=True)
    shards = cfg.get("shards") or []
    if not shards:
        raise SystemExit("no nodes — start `drift node` on each machine, "
                         "or pass --nodes host:port,…")
    return [{"name": s["name"], "host": s["host"], "port": s["port"]} for s in shards]


def up_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="drift up",
        description="localhost: spawn N nodes, auto-split the model, and chat/generate")
    ap.add_argument("n", nargs="?", type=int, default=2, help="number of local nodes (default 2)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--prompt", help="one-shot; omit for an interactive chat")
    ap.add_argument("--max-new-tokens", type=int)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    model_id, dtype = cfg["model_id"], cfg.get("dtype", "float16")
    head_device = pick_device(cfg.get("device"))
    n_new = args.max_new_tokens or cfg["generation"]["max_new_tokens"]

    ports = [free_port() for _ in range(args.n)]
    print(f"[up] launching {args.n} local nodes on {ports} (device={head_device}) …", flush=True)
    procs = [subprocess.Popen(
        [sys.executable, "-m", "drift.node", "--port", str(p), "--host", "127.0.0.1", "--quiet"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for p in ports]
    try:
        endpoints = [{"name": f"n{i}", "host": "127.0.0.1", "port": p}
                     for i, p in enumerate(ports)]
        orch, plan = build_over_nodes(model_id, dtype, head_device, endpoints)
        _status_bar(model_id, plan, head_device)
        if args.prompt:
            _stream(orch, args.prompt, n_new)
        else:
            _repl(orch, n_new)
    finally:
        for pr in procs:
            pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=10)
            except Exception:
                pr.kill()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="drift run",
        description="head: assign layers to running nodes and chat/generate")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--nodes", help="comma-separated host:port of running `drift node`s")
    ap.add_argument("--no-discover", action="store_true", help="skip LAN auto-discovery")
    ap.add_argument("--discover-timeout", type=float, default=3.0)
    ap.add_argument("--model", help="override model_id")
    ap.add_argument("--prompt", help="one-shot; omit for an interactive chat")
    ap.add_argument("--max-new-tokens", type=int)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    model_id = args.model or cfg["model_id"]
    dtype = cfg.get("dtype", "float16")
    head_device = pick_device(cfg.get("device"))
    n_new = args.max_new_tokens or cfg["generation"]["max_new_tokens"]

    endpoints = _select_endpoints(args, cfg)

    print(f"[run] {len(endpoints)} node(s); splitting {model_id} …", flush=True)
    orch, plan = build_over_nodes(model_id, dtype, head_device, endpoints)
    _status_bar(model_id, plan, head_device)
    if args.prompt:
        _stream(orch, args.prompt, n_new)
    else:
        _repl(orch, n_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
