"""Integration gate — spawn real local nodes, run the split over TCP, and assert
the greedy token ids are **bitwise-identical** to the in-process reference.

The in-process path is already proven bitwise == a clean HF model by
`drift parity --selftest`, so it is a trustworthy oracle here. This gate then
isolates whatever the milestone added on top of the plain socket path:

    python -m drift.itest --nodes 2                 # star (every hop via head)
    python -m drift.itest --nodes 2 --chain         # M7 peer-to-peer chain
    python -m drift.itest --nodes 3 --chain --secure # + M8 encrypted wire
    python -m drift.itest --nodes 3 --chain --kill 1 # M9 mid-run failover
    python -m drift.itest --nodes 2 --chain --thin   # M10 zero-weight head

Because the reference and the split share a device and greedy decoding, a match
must be exact — any divergence is a real bug in the transport/relay, not float
noise.
"""

from __future__ import annotations

import argparse
import binascii
import os
import subprocess
import sys
import threading
import time

from .common import free_port, load_config, pick_device
from .orchestrator import build_inprocess
from .run import build_over_nodes

_CASES = [
    ("Give me a short introduction to large language models.", 32),
    ("def fibonacci(n):", 24),
    ("한국어로 인공지능을 한 문장으로 설명해줘.", 24),
]


def spawn_nodes(n: int, extra: list[str] | None = None) -> tuple[list[int], list[subprocess.Popen]]:
    """Launch n unassigned local `drift node` workers; return (ports, procs)."""
    ports = [free_port() for _ in range(n)]
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "drift.node", "--port", str(p), "--host", "127.0.0.1",
             "--quiet", "--no-advertise", *(extra or [])],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for p in ports
    ]
    return ports, procs


def _teardown(procs: list[subprocess.Popen]) -> None:
    for pr in procs:
        pr.terminate()
    for pr in procs:
        try:
            pr.wait(timeout=10)
        except Exception:
            pr.kill()


def _kill_test(cfg, model_id, dtype, dev, args) -> int:
    """M9: start `nodes` active + 1 spare, kill node K mid-generation, and assert
    the recovered sequence is bitwise-identical to an uninterrupted reference."""
    tag = "kill" + ("+secure" if args.secure else "") + ("+chain" if args.chain else "")
    prompt, n = "Count from one to forty in words.", 48

    print(f"[itest:{tag}] building in-process reference …", flush=True)
    ref = build_inprocess(cfg)
    ref_ids = ref.generate(prompt, n, stop_on_eos=False, session_id="ref")["token_ids"]

    print(f"[itest:{tag}] spawning {args.nodes} active + 1 spare node(s) …", flush=True)
    ports, procs = spawn_nodes(args.nodes + 1)
    try:
        active = [{"name": f"n{i}", "host": "127.0.0.1", "port": p}
                  for i, p in enumerate(ports[:args.nodes])]
        spare = [{"name": "spare", "host": "127.0.0.1", "port": ports[-1]}]
        orch, plan = build_over_nodes(model_id, dtype, dev, active, chain=args.chain, spares=spare)
        print(f"[itest:{tag}] split: " +
              " · ".join(f"{p['name']}[{p['start']}:{p['end']})" for p in plan) +
              "  (+1 spare held in reserve)", flush=True)

        result, err = {}, {}

        def run():
            try:
                result["ids"] = orch.generate(prompt, n, stop_on_eos=False,
                                              session_id="kill")["token_ids"]
            except Exception as e:  # noqa: BLE001 — surfaced below
                err["e"] = e

        th = threading.Thread(target=run)
        th.start()
        # Let a few tokens land (KV is built), then kill node K mid-decode.
        t0 = time.time()
        while orch.progress < 4 and th.is_alive() and time.time() - t0 < 120:
            time.sleep(0.02)
        print(f"[itest:{tag}] killing node index {args.kill} at progress={orch.progress} tokens …",
              flush=True)
        victim = procs[args.kill]
        victim.terminate()
        try:
            victim.wait(timeout=10)
        except Exception:
            victim.kill()
        th.join(timeout=300)

        if err:
            print(f"[itest:{tag}] FAIL — generation raised: {type(err['e']).__name__}: {err['e']}",
                  flush=True)
            return 1
        got = result.get("ids", [])
        div = next((i for i in range(min(len(got), len(ref_ids))) if got[i] != ref_ids[i]), None)
        exact = got == ref_ids
        recovered = orch.recoveries >= 1
        print(f"[itest:{tag}] recoveries={orch.recoveries}  tokens={len(got)}  "
              f"bitwise=={exact}  first_div={div}", flush=True)
        good = exact and recovered
        print(f"[itest:{tag}]",
              "PASS — bitwise-identical resume after a mid-run node kill" if good
              else "FAIL — " + ("no recovery was triggered" if exact and not recovered
                                else "recovered sequence diverged"), flush=True)
        return 0 if good else 1
    finally:
        _teardown(procs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DRIFT integration parity gate (real nodes over TCP)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--nodes", type=int, default=2, help="number of local worker nodes")
    ap.add_argument("--chain", action="store_true", help="peer-to-peer chain transport (M7)")
    ap.add_argument("--secure", action="store_true",
                    help="encrypt the wire with a throwaway network key (M8)")
    ap.add_argument("--kill", type=int, default=None, metavar="K",
                    help="M9: kill node index K mid-generation; assert bitwise-identical recovery")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    model_id, dtype = cfg["model_id"], cfg.get("dtype", "float16")
    dev = pick_device(cfg.get("device"))

    if args.secure:
        # A throwaway network key, shared with the spawned nodes via the inherited
        # environment. Proves the AEAD channel is bitwise-transparent to parity.
        os.environ["DRIFT_NETWORK_KEY"] = binascii.hexlify(os.urandom(32)).decode()

    if args.kill is not None:
        return _kill_test(cfg, model_id, dtype, dev, args)

    topo = ("chain" if args.chain else "star") + ("+secure" if args.secure else "")

    print(f"[itest:{topo}] building in-process reference …", flush=True)
    ref = build_inprocess(cfg)

    print(f"[itest:{topo}] spawning {args.nodes} local node(s) …", flush=True)
    ports, procs = spawn_nodes(args.nodes)
    try:
        endpoints = [{"name": f"n{i}", "host": "127.0.0.1", "port": p}
                     for i, p in enumerate(ports)]
        orch, plan = build_over_nodes(model_id, dtype, dev, endpoints, chain=args.chain)
        print(f"[itest:{topo}] split: " +
              " · ".join(f"{p['name']}[{p['start']}:{p['end']})" for p in plan), flush=True)

        all_ok = True
        for prompt, n in _CASES:
            r = ref.generate(prompt, n, stop_on_eos=False, session_id=f"ref-{n}")["token_ids"]
            g = orch.generate(prompt, n, stop_on_eos=False, session_id=f"itest-{n}")["token_ids"]
            div = next((i for i in range(min(len(r), len(g))) if r[i] != g[i]), None)
            ok = (r == g)
            all_ok &= ok
            print(f"[itest:{topo}] {'PASS' if ok else 'FAIL'} n={n:>3} "
                  f"first_div={div} prompt={prompt[:32]!r}", flush=True)
        print(f"[itest:{topo}]",
              "ALL PASS — bitwise == in-process reference" if all_ok else "FAIL", flush=True)
        return 0 if all_ok else 1
    finally:
        _teardown(procs)


if __name__ == "__main__":
    sys.exit(main())
