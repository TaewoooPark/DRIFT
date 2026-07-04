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
import os
import subprocess
import sys
import time

from .common import (free_port, lan_ip, load_config, model_num_layers,
                     pick_device, split_layers)
from .orchestrator import (ChainTransport, HeadModel, Orchestrator,
                           SocketTransport)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


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


def _make_transport(shards: list[dict], dtype: str, head_device: str, chain: bool,
                    int8: bool = False):
    """A star or chain transport over `shards` (chain picks a reachable collect host).
    `int8` sends the hidden state as int8 on the wire (half the bytes, lossy)."""
    if not chain:
        t = SocketTransport(shards, dtype, head_device)
    else:
        # The tail dials the head's collect sink. For an all-localhost run use
        # loopback (avoids routing self-traffic over the LAN interface, where a
        # firewall can silently drop it); otherwise the head's LAN address.
        all_local = all(s["host"] in _LOCAL_HOSTS for s in shards)
        collect_host = "127.0.0.1" if all_local else lan_ip()
        t = ChainTransport(shards, dtype, head_device, collect_host=collect_host)
    if int8:
        t.wire_dtype = "int8"
    return t


class _Cluster:
    """Re-splits a model over whichever pool nodes are still alive (M9 failover).

    The pool is the initial nodes plus any spares. On a mid-run drop the
    orchestrator calls `rebuild()`, which pings the pool, splits the layers across
    the survivors, configures them, and returns a fresh (transport, order). The
    orchestrator then re-prefills the sequence-so-far — bitwise-identical resume.
    """

    def __init__(self, model_id, dtype, head_device, pool, chain, thin=False, int8=False):
        self.model_id = model_id
        self.dtype = dtype
        self.head_device = head_device
        self.pool = [dict(e) for e in pool]
        self.chain = chain
        self.thin = thin
        self.int8 = int8
        self.n_layers = model_num_layers(model_id)

    def _alive(self) -> list[dict]:
        alive = []
        for e in self.pool:
            t = SocketTransport([e], self.dtype, self.head_device)
            try:
                if t.ping(e["name"]).get("ok"):
                    alive.append(e)
            except Exception:
                pass
            finally:
                t.close()
        return alive

    def rebuild(self):
        from .orchestrator import NodeUnavailable
        alive = self._alive()
        if not alive:
            raise NodeUnavailable("no surviving nodes to rebuild the pipeline over")
        ranges = split_layers(self.n_layers, len(alive))
        shards = [dict(e) for e in alive]
        names = [s["name"] for s in shards]
        transport = _make_transport(shards, self.dtype, self.head_device, self.chain, self.int8)
        if not _wait_ready(transport, names):
            raise NodeUnavailable("survivors not reachable while rebuilding")
        _check_env(transport, names)
        last = len(shards) - 1
        for i, (s, (a, b)) in enumerate(zip(shards, ranges)):
            transport.configure(s["name"], a, b, self.model_id, self.dtype, s.get("device"),
                                embed_duty=(self.thin and i == 0),
                                head_duty=(self.thin and i == last))
        return transport, names


def build_over_nodes(model_id: str, dtype: str, head_device: str,
                     endpoints: list[dict], chain: bool = False,
                     spares: list[dict] | None = None, thin: bool = False,
                     int8: bool = False
                     ) -> tuple[Orchestrator, list[dict]]:
    """Endpoints [{name,host,port,device?}] → auto-split, configure, return
    (orchestrator, plan). `plan` is per-node {name,host,port,start,end,device}.

    `chain=True` uses the peer-to-peer ChainTransport (node→node→…→collect); the
    default star SocketTransport round-trips every hop through the head. `spares`
    are extra ready nodes held in reserve — on a mid-run drop the orchestrator
    re-splits over the survivors plus these (M9 failover). `thin=True` (implies
    chain) moves embed to the first node and norm+lm_head to the last, so the head
    holds zero model weights and exchanges only token ids (M10).
    """
    if thin:
        chain = True  # thin mode is chain-only (the head exchanges token ids, no tensor)
    n_layers = model_num_layers(model_id)
    ranges = split_layers(n_layers, len(endpoints))
    shards = [dict(e) for e in endpoints]
    names = [s["name"] for s in shards]
    transport = _make_transport(shards, dtype, head_device, chain, int8)
    if not _wait_ready(transport, names):
        raise RuntimeError("nodes not reachable — try `drift doctor --nodes <host:port,…>`")
    _check_env(transport, names)
    plan = []
    last = len(shards) - 1
    for i, (s, (a, b)) in enumerate(zip(shards, ranges)):
        info = transport.configure(s["name"], a, b, model_id, dtype, s.get("device"),
                                   embed_duty=(thin and i == 0), head_duty=(thin and i == last))
        plan.append({**s, "start": a, "end": b, "device": info.get("device"),
                     "pubkey": info.get("pubkey")})
    head = HeadModel(model_id, head_device, dtype, sliced=not thin, thin=thin)
    orch = Orchestrator(head, transport, names, head_device)
    orch.cluster = _Cluster(model_id, dtype, head_device,
                            [*endpoints, *(spares or [])], chain, thin=thin, int8=int8)
    # M11: verify each token's signed receipts against the head's anchors.
    from .receipts import ReceiptVerifier
    orch.verify = True
    orch.verifier = ReceiptVerifier()
    orch.n_layers = n_layers
    orch.journal = os.environ.get("DRIFT_JOURNAL")  # M13: opt-in contribution ledger
    return orch, plan


def _status_bar(model_id: str, plan: list[dict], head_device: str,
                chain: bool = False, thin: bool = False) -> None:
    topo = "chain (node→node→…→head)" if chain else "star (every hop through head)"
    head_holds = "tokenizer only (zero weights)" if thin else "embed + norm + lm_head"
    print(f"\n  model : {model_id}")
    print(f"  route : {topo}")
    print(f"  head  : {head_holds}  · device={head_device}")
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


def _expand_members(seeds: list[dict]) -> list[dict]:
    """Gossip-discover the full membership from seed nodes and return every member
    as an endpoint (M12). Falls back to the seeds if discovery turns up nothing."""
    from . import membership

    table = membership.PeerTable()
    for s in seeds:
        try:
            got = membership.fetch_peers(s["host"], s["port"])
            n = table.merge(got)
            print(f"[expand] {s['host']}:{s['port']} → {len(got)} peer(s) ({n} new)", flush=True)
        except Exception as e:
            print(f"[expand] {s['host']}:{s['port']} unreachable: {e}", flush=True)
    members = table.endpoints()
    if not members:
        print("[expand] no members discovered — using the seeds as given", flush=True)
        return seeds
    print(f"[expand] membership = {len(members)} node(s)", flush=True)
    return members


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
    ap.add_argument("--chain", action="store_true",
                    help="peer-to-peer chain: nodes stream to each other, not through the head")
    ap.add_argument("--thin", action="store_true",
                    help="zero-weight head: embed+lm_head move to the edge nodes (implies --chain)")
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
        orch, plan = build_over_nodes(model_id, dtype, head_device, endpoints,
                                      chain=args.chain, thin=args.thin)
        _status_bar(model_id, plan, head_device, chain=args.chain or args.thin, thin=args.thin)
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
    ap.add_argument("--chain", action="store_true",
                    help="peer-to-peer chain: nodes stream to each other, not through the head")
    ap.add_argument("--thin", action="store_true",
                    help="zero-weight head: embed+lm_head move to the edge nodes (implies --chain)")
    ap.add_argument("--int8", action="store_true",
                    help="send the hidden state as int8 (half the wire bytes; lossy, relaxed gate)")
    ap.add_argument("--expand", action="store_true",
                    help="treat --nodes as seeds: gossip-discover the whole membership and split across all of it")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    model_id = args.model or cfg["model_id"]
    dtype = cfg.get("dtype", "float16")
    head_device = pick_device(cfg.get("device"))
    n_new = args.max_new_tokens or cfg["generation"]["max_new_tokens"]

    endpoints = _select_endpoints(args, cfg)
    if args.expand:
        endpoints = _expand_members(endpoints)

    print(f"[run] {len(endpoints)} node(s); splitting {model_id} …", flush=True)
    orch, plan = build_over_nodes(model_id, dtype, head_device, endpoints,
                                  chain=args.chain, thin=args.thin, int8=args.int8)
    _status_bar(model_id, plan, head_device, chain=args.chain or args.thin, thin=args.thin)
    if args.prompt:
        _stream(orch, args.prompt, n_new)
    else:
        _repl(orch, n_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
