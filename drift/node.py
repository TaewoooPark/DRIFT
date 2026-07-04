"""`drift node` — turn this machine into a worker.

Auto-detects the device, picks a port, binds so other machines can reach it, and
waits for the head to assign its layer range (a `configure` message). The user
never types a layer range or a device.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import discovery
from .common import free_port, lan_ip, load_config, pick_device
from .shard_server import Node, serve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="drift node",
                                 description="run this machine as a DRIFT worker node")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", help="mps | cuda | cpu (default: auto-detect)")
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address (0.0.0.0 = reachable from other machines)")
    ap.add_argument("--port", type=int,
                    help="listen port (default: $DRIFT_PORT or an OS-assigned free port)")
    ap.add_argument("--quiet", action="store_true", help="one-line banner (used by `drift up`)")
    ap.add_argument("--no-advertise", action="store_true", help="do not announce over mDNS")
    ap.add_argument("--tunnel", action="store_true",
                    help="expose this node over a public bore.pub tunnel (for NAT/Colab/cloud) — "
                         "no account needed; the head connects with `drift run --nodes <printed>`")
    ap.add_argument("--tamper", action="store_true",
                    help="TEST ONLY: corrupt this node's output so the head's receipt verifier flags it")
    args = ap.parse_args(argv)

    cfg = {}
    try:
        cfg = load_config(args.config)
    except Exception:
        pass  # a node needs no config; the head sends model + range via configure

    # A public tunnel with no network key is an open compute endpoint — anyone who
    # scans the bore.pub port could drive this node's GPU. Require a key for it.
    if args.tunnel:
        from .crypto import network_key
        if network_key() is None:
            print("[node] refusing --tunnel without a network key: a public endpoint would be "
                  "open compute for anyone who finds the port.\n"
                  "       run `drift keygen`, then `export DRIFT_NETWORK_KEY=<the printed hex>` "
                  "on this machine and the head.", flush=True)
            return 2

    device = pick_device(args.device)
    port = args.port or int(os.environ.get("DRIFT_PORT") or free_port())
    node = Node(name=f"node-{port}", model_id=cfg.get("model_id", "(assigned by head)"),
                dtype=cfg.get("dtype", "float16"), device=device)
    node.tamper = args.tamper
    ip = lan_ip()

    # Public tunnel (for a node behind NAT / on Colab / on a cloud VM).
    tunnel_addr, tunnel_proc = None, None
    if args.tunnel:
        from . import tunnel as _tunnel
        print(f"[node] opening a bore.pub tunnel to :{port} …", flush=True)
        try:
            tunnel_addr, tunnel_proc = _tunnel.open_bore(port)
        except Exception as e:
            print(f"[node] tunnel unavailable ({e}); LAN address only", flush=True)
        if tunnel_addr:
            print(f"[node] public address: {tunnel_addr}", flush=True)
        else:
            print("[node] tunnel did not come up (bore.pub busy?) — LAN address only", flush=True)

    advertised = not args.no_advertise and discovery.HAVE_ZEROCONF
    disc_line = ("this node auto-announces on the LAN — the head can just run `drift run`"
                 if advertised else
                 "zeroconf not active — the head must use `drift run --nodes …`")
    head_addr = tunnel_addr or f"{ip}:{port}"
    if args.quiet:
        extra = f" tunnel={tunnel_addr}" if tunnel_addr else ""
        banner = f"[node] {ip}:{port} device={device} advertise={advertised}{extra} — ready"
    else:
        tunnel_line = f"    tunnel  : {tunnel_addr}  (reachable from anywhere)\n" if tunnel_addr else ""
        banner = (
            "\n  DRIFT node ready\n"
            f"    address : {ip}:{port}\n"
            f"{tunnel_line}"
            f"    device  : {device}\n"
            f"    discover: {disc_line}\n"
            "    status  : waiting for the head to assign layers…\n\n"
            f"  head:  drift run --nodes {head_addr},<others…>\n"
        )

    handle = None if args.no_advertise else discovery.advertise(port, device, name=node.name)
    try:
        serve(node, args.host, port, banner=banner)
    except KeyboardInterrupt:
        print("\n[node] shutting down", flush=True)
    finally:
        discovery.unadvertise(handle)
        if tunnel_proc is not None:
            tunnel_proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
