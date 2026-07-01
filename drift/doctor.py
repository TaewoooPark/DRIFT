"""`drift doctor` — preflight environment & config check.

Turns cryptic failures (ConnectionRefused, wrong device, bad split) into an
actionable checklist *before* you start a run. Read-only and fast.
"""

from __future__ import annotations

import argparse
import platform
import socket
import sys

OK = "\033[32m✓\033[0m"
WARN = "\033[33m⚠\033[0m"
BAD = "\033[31m✗\033[0m"


def _mark(ok: bool, warn: bool = False) -> str:
    return WARN if warn else (OK if ok else BAD)


def _check_python() -> bool:
    v = sys.version_info
    ok = (v.major, v.minor) == (3, 12)
    print(f"  {_mark(ok, warn=not ok)} Python {v.major}.{v.minor}.{v.micro}"
          + ("" if ok else "  → DRIFT targets 3.12 (PyTorch has no 3.14 wheel yet)"))
    return True


def _check_torch() -> str | None:
    try:
        import torch
    except Exception as e:
        print(f"  {BAD} torch not importable: {e}  → uv pip install torch")
        return None
    mps = torch.backends.mps.is_available()
    cuda = torch.cuda.is_available()
    dev = "mps" if mps else ("cuda" if cuda else "cpu")
    print(f"  {OK} torch {torch.__version__}  · mps={mps} cuda={cuda}"
          f"  → auto device = \033[1m{dev}\033[0m")
    if dev == "cpu":
        print(f"  {WARN} no GPU detected — runs will work but be slow "
              "(Mac needs Apple silicon; Windows/Linux needs a CUDA build of torch)")
    return dev


def _check_deps() -> None:
    required = ["transformers", "safetensors", "msgpack", "numpy", "yaml", "huggingface_hub"]
    for mod in required:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "")
            print(f"  {OK} {mod} {ver}")
        except Exception:
            pkg = "pyyaml" if mod == "yaml" else mod
            print(f"  {BAD} {mod} missing  → uv pip install {pkg}")
    try:
        import zeroconf  # noqa: F401
        print(f"  {OK} zeroconf (LAN auto-discovery available)")
    except Exception:
        print(f"  {WARN} zeroconf not installed — `drift run` auto-discovery off; "
              "use `--nodes host:port,…`  → uv pip install zeroconf")


def _check_config(path: str) -> None:
    try:
        from .common import load_config, model_num_layers
        cfg = load_config(path)
    except FileNotFoundError:
        print(f"  {WARN} {path} not found (fine for `drift up`/`--nodes`; needed otherwise)")
        return
    except Exception as e:
        print(f"  {BAD} {path} unreadable: {e}")
        return
    model_id = cfg.get("model_id", "?")
    print(f"  {OK} {path} · model={model_id} dtype={cfg.get('dtype')}")
    dtype = cfg.get("dtype", "float16")
    if dtype not in ("float16", "float32"):
        print(f"  {BAD} dtype '{dtype}' is not valid on the wire → use float16 or float32")
    shards = cfg.get("shards") or []
    if shards:
        try:
            n = model_num_layers(model_id)
            covered, ok = [], True
            cur = 0
            for s in sorted(shards, key=lambda x: x["start_layer"]):
                if s["start_layer"] != cur:
                    ok = False
                cur = s["end_layer"]
                covered.append((s["start_layer"], s["end_layer"]))
            ok = ok and cur == n
            print(f"  {_mark(ok)} shard tiling {covered} over {n} layers"
                  + ("" if ok else "  → ranges must be contiguous and cover [0, n_layers)"))
        except Exception as e:
            print(f"  {WARN} could not verify layer tiling ({e})")


def _check_ports(nodes: list[tuple[str, int]]) -> None:
    for host, port in nodes:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"  {OK} {host}:{port} reachable")
        except Exception as e:
            print(f"  {BAD} {host}:{port} not reachable: {type(e).__name__}"
                  "  → is `drift node` running there? port open in the firewall?")


def _firewall_hint() -> None:
    sysname = platform.system()
    if sysname == "Windows":
        print(f"  {WARN} Windows: first `drift node` run pops a Defender prompt — "
              "Allow access on Private networks. For discovery, enable Network Discovery.")
    elif sysname == "Darwin":
        print(f"  {WARN} macOS: if a peer can't reach this node, allow incoming "
              "connections for python in System Settings → Network → Firewall.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="drift doctor", description="preflight check")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--nodes", help="comma-separated host:port to probe for reachability")
    args = ap.parse_args(argv)

    print("DRIFT doctor\n------------")
    print("environment:")
    _check_python()
    dev = _check_torch()
    _check_deps()
    print("config:")
    _check_config(args.config)
    if args.nodes:
        print("nodes:")
        pairs = []
        for tok in args.nodes.split(","):
            host, _, port = tok.strip().rpartition(":")
            pairs.append((host or "127.0.0.1", int(port)))
        _check_ports(pairs)
    print("notes:")
    _firewall_hint()
    print(f"\nready. next: `drift up 2` (localhost) or `drift node` on each machine "
          f"+ `drift run` on the head. auto device = {dev or 'cpu'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
