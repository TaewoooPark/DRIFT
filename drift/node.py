"""`drift node` — turn this machine into a worker.

Auto-detects the device, picks a port, binds so other machines can reach it, and
waits for the head to assign its layer range (a `configure` message). The user
never types a layer range or a device.
"""

from __future__ import annotations

import argparse
import os
import sys

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
    args = ap.parse_args(argv)

    cfg = {}
    try:
        cfg = load_config(args.config)
    except Exception:
        pass  # a node needs no config; the head sends model + range via configure

    device = pick_device(args.device)
    port = args.port or int(os.environ.get("DRIFT_PORT") or free_port())
    node = Node(name=f"node-{port}", model_id=cfg.get("model_id", "(assigned by head)"),
                dtype=cfg.get("dtype", "float16"), device=device)
    ip = lan_ip()

    if args.quiet:
        banner = f"[node] {ip}:{port} device={device} — ready, waiting for the head"
    else:
        banner = (
            "\n  DRIFT node ready\n"
            f"    address : {ip}:{port}\n"
            f"    device  : {device}\n"
            "    status  : waiting for the head to assign layers…\n\n"
            f"  on the head machine, run:  drift run --nodes {ip}:{port},<other-nodes…>\n"
            "  (or just `drift run` if zeroconf discovery is installed on the LAN)\n"
        )
    try:
        serve(node, args.host, port, banner=banner)
    except KeyboardInterrupt:
        print("\n[node] shutting down", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
