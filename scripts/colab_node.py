"""Run a DRIFT CUDA node on Google Colab and expose it to your Mac.

DRIFT's head *dials* its nodes (node = TCP server), but Colab accepts no inbound
connections — so the node's port has to be tunneled out. This script starts a
`drift node` on the Colab GPU and opens an ngrok TCP tunnel to it, then prints
the exact command to run on your Mac.

Colab setup (one cell each)
---------------------------
    !nvidia-smi -L                                   # confirm a GPU runtime
    !git clone https://github.com/TaewoooPark/DRIFT
    %cd DRIFT
    !pip install -q -e . pyngrok
    # free ngrok authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
    import os; os.environ["NGROK_AUTHTOKEN"] = "PASTE_YOUR_TOKEN"
    !python scripts/colab_node.py            # blocks; keep this cell running

Then on the Mac (leave the Colab cell running):
    python -m drift.bench_m4 --nodes <printed host:port> --json m4_results.json
    #   ...or use BOTH GPUs — also `drift node --port 52600` on the Mac, then
    #   --nodes 127.0.0.1:52600,<printed host:port>   (remote takes the back half)

No-account alternative to ngrok: `bore` (https://github.com/ekzhang/bore) —
run `python scripts/colab_node.py --no-tunnel` and, in another cell,
`!curl -sSL https://... -o bore && chmod +x bore && ./bore local 52601 --to bore.pub`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DRIFT CUDA node for Colab (node + ngrok tunnel)")
    ap.add_argument("--port", type=int, default=52601)
    ap.add_argument("--ngrok-token", default=os.environ.get("NGROK_AUTHTOKEN"))
    ap.add_argument("--no-tunnel", action="store_true",
                    help="just run the node; expose the port with your own tunnel")
    ap.add_argument("--bore", action="store_true",
                    help="expose via bore.pub (no account, no token) instead of ngrok")
    args = ap.parse_args(argv)

    if not _has_cuda():
        print("!! no CUDA GPU detected — set the Colab runtime to a GPU "
              "(Runtime → Change runtime type → GPU).", flush=True)
        return 1

    import torch
    print(f"[colab] GPU: {torch.cuda.get_device_name(0)}  ·  torch {torch.__version__}", flush=True)

    # Start the DRIFT node (binds 0.0.0.0 so the tunnel can reach it). It waits
    # for the head to assign a layer range, so no model is loaded until you run
    # the benchmark on the Mac.
    node = subprocess.Popen(
        [sys.executable, "-m", "drift.node", "--port", str(args.port),
         "--host", "0.0.0.0", "--device", "cuda", "--no-advertise", "--quiet"],
    )
    time.sleep(3)
    if node.poll() is not None:
        print("!! drift node exited early — check the install (pip install -e .).", flush=True)
        return 1
    print(f"[colab] drift node listening on 0.0.0.0:{args.port} (device=cuda)", flush=True)

    if args.bore:
        from drift.tunnel import open_bore

        addr, _tp = open_bore(args.port)
        if not addr:
            print("!! bore did not report an address (bore.pub may be busy) — retry, "
                  "or use ngrok.", flush=True)
            node.terminate()
            return 1
        print("\n" + "=" * 64, flush=True)
        print(f"  CUDA node is reachable at:   {addr}", flush=True)
        print("  On your Mac, run:", flush=True)
        print(f"    python -m drift.bench_m4 --nodes {addr} --json m4_results.json", flush=True)
        print(f"  (both GPUs: also `drift node --port 52600` on the Mac, then "
              f"--nodes 127.0.0.1:52600,{addr} )", flush=True)
        print("=" * 64 + "\n  keep this cell running.\n", flush=True)
        try:
            node.wait()
        except KeyboardInterrupt:
            pass
        finally:
            node.terminate()
        return 0

    if args.no_tunnel:
        print(f"\n[colab] node up. Expose port {args.port} with your own tunnel, e.g.:\n"
              f"    ./bore local {args.port} --to bore.pub\n"
              f"then on the Mac:  python -m drift.bench_m4 --nodes <that host:port>\n", flush=True)
    else:
        try:
            from pyngrok import conf, ngrok
        except Exception:
            print("!! pyngrok not installed — `pip install pyngrok`, or use --no-tunnel.", flush=True)
            node.terminate()
            return 1
        if args.ngrok_token:
            conf.get_default().auth_token = args.ngrok_token
        try:
            tunnel = ngrok.connect(args.port, "tcp")
        except Exception as e:
            print(f"!! ngrok failed ({e}).\n   Set a free authtoken: "
                  f"os.environ['NGROK_AUTHTOKEN']='...', or use --no-tunnel.", flush=True)
            node.terminate()
            return 1
        addr = tunnel.public_url.replace("tcp://", "")
        print("\n" + "=" * 64, flush=True)
        print(f"  CUDA node is reachable at:   {addr}", flush=True)
        print("  On your Mac, run:", flush=True)
        print(f"    python -m drift.bench_m4 --nodes {addr} --json m4_results.json", flush=True)
        print("  (to use both GPUs: also `drift node --port 52600` on the Mac,", flush=True)
        print(f"     then --nodes 127.0.0.1:52600,{addr} )", flush=True)
        print("=" * 64 + "\n  keep this cell running. Ctrl-C / stop to tear down.\n", flush=True)

    try:
        node.wait()
    except KeyboardInterrupt:
        print("\n[colab] shutting down node + tunnel", flush=True)
    finally:
        node.terminate()
        if not args.no_tunnel:
            try:
                from pyngrok import ngrok
                ngrok.kill()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
