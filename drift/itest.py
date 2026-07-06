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
from .run import _expand_members, build_over_nodes


def _build_inprocess(cfg: dict):
    from .orchestrator import build_inprocess

    return build_inprocess(cfg)


_CASES = [
    ("Give me a short introduction to large language models.", 32),
    ("def fibonacci(n):", 24),
    ("한국어로 인공지능을 한 문장으로 설명해줘.", 24),
]


def spawn_nodes(n: int, extra: list[str] | None = None,
                tamper_idx: int | None = None) -> tuple[list[int], list[subprocess.Popen]]:
    """Launch n unassigned local `drift node` workers; return (ports, procs).

    Each node gets its OWN identity file (distinct Ed25519 keypair) so it models a
    distinct machine — the M11 receipts/reputation key by node pubkey. `tamper_idx`
    makes that one node corrupt its output so the head's verifier flags it.
    """
    ports = [free_port() for _ in range(n)]
    procs = []
    for i, p in enumerate(ports):
        env = dict(os.environ)
        env["DRIFT_IDENTITY_FILE"] = f"/tmp/drift_id_{p}.key"
        args = [sys.executable, "-m", "drift.node", "--port", str(p), "--host", "127.0.0.1",
                "--quiet", "--no-advertise", *(extra or [])]
        if i == tamper_idx:
            args.append("--tamper")
        procs.append(subprocess.Popen(args, env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    return ports, procs


def _teardown(procs: list[subprocess.Popen]) -> None:
    for pr in procs:
        pr.terminate()
    for pr in procs:
        try:
            pr.wait(timeout=10)
        except Exception:
            pr.kill()


def _int8_test(cfg, model_id, dtype, dev, args) -> int:
    """M14: int8 wire halves the bytes per hop (lossy → relaxed gate). Measure the
    match-rate vs the fp16 reference and the byte savings; never claims bitwise."""
    from transformers import AutoConfig

    hidden = getattr(AutoConfig.from_pretrained(model_id), "hidden_size", None) or \
        AutoConfig.from_pretrained(model_id).text_config.hidden_size
    print(f"[itest:int8] building in-process reference (fp16) …", flush=True)
    ref = _build_inprocess(cfg)

    ports, procs = spawn_nodes(args.nodes)
    try:
        endpoints = [{"name": f"n{i}", "host": "127.0.0.1", "port": p}
                     for i, p in enumerate(ports)]
        orch, plan = build_over_nodes(model_id, dtype, dev, endpoints, chain=True, int8=True)
        rates, divs = [], []
        for prompt, n in _CASES[:2]:
            r = ref.generate(prompt, n, stop_on_eos=False, session_id=f"ref-{n}")["token_ids"]
            g = orch.generate(prompt, n, stop_on_eos=False, session_id=f"int8-{n}")["token_ids"]
            matches = sum(1 for a, b in zip(r, g) if a == b)
            rate = matches / max(len(r), 1)
            div = next((i for i in range(min(len(r), len(g))) if r[i] != g[i]), None)
            rates.append(rate); divs.append(div)
            print(f"[itest:int8] n={n:>3} match_rate={rate:.1%} first_div={div}", flush=True)
    finally:
        _teardown(procs)

    from .protocol import _INT8_GROUP
    ng = (hidden + _INT8_GROUP - 1) // _INT8_GROUP
    fp16_bytes = hidden * 2
    int8_bytes = hidden + ng * 2  # H int8 payload + per-group fp16 scales
    ratio = int8_bytes / fp16_bytes
    mean_rate = sum(rates) / len(rates)
    suspects = orch.verifier.suspects() if orch.verifier else []
    print(f"[itest:int8] wire/token/hop: int8={int8_bytes} B vs fp16={fp16_bytes} B "
          f"({ratio:.0%}) · mean match {mean_rate:.1%} · receipts clean={not suspects}", flush=True)
    # PASS: the int8 path materially cuts the wire, stays self-verifying, and keeps
    # a usable match-rate. The rate is the *measured* relaxed-gate fidelity, never
    # claimed bitwise.
    good = (ratio < 0.75) and (mean_rate >= 0.5) and (not suspects)
    print("[itest:int8]",
          f"PASS — int8 wire {ratio:.0%} of fp16; measured fidelity {mean_rate:.1%} (relaxed gate)"
          if good else f"FAIL — ratio={ratio:.0%} fidelity={mean_rate:.1%}", flush=True)
    return 0 if good else 1


def _ledger_test(cfg, model_id, dtype, dev, args) -> int:
    """M13: run a generation with a receipt journal, then assert the ledger tally
    reconciles with the run and a forged line fails --verify."""
    import json

    from . import ledger
    from .receipts import read_journal, verify_receipt, from_json

    journal = f"/tmp/drift_journal_{os.getpid()}.jsonl"
    if os.path.exists(journal):
        os.remove(journal)
    os.environ["DRIFT_JOURNAL"] = journal
    prompt, n = "Give me a short introduction to large language models.", 20

    print(f"[itest:ledger] generating (journal → {journal}) …", flush=True)
    ports, procs = spawn_nodes(args.nodes)
    try:
        endpoints = [{"name": f"n{i}", "host": "127.0.0.1", "port": p}
                     for i, p in enumerate(ports)]
        orch, plan = build_over_nodes(model_id, dtype, dev, endpoints, chain=True)
        orch.generate(prompt, n, stop_on_eos=False, session_id="ledger")
    finally:
        _teardown(procs)

    rows = read_journal(journal)
    agg = ledger.aggregate(rows, verified_only=True)
    all_valid = all(verify_receipt(from_json(d)) for d in rows)
    # Each of the `nodes` shards signs one receipt per token → same token count each.
    tok_counts = sorted(a["tokens"] for a in agg.values())
    balanced = len(agg) == args.nodes and len(set(tok_counts)) == 1 and tok_counts[0] == n

    # A forged line must fail --verify (append a receipt with a broken signature).
    forged = dict(rows[0]); forged["sig"] = "00" * 64
    with open(journal, "a") as f:
        f.write(json.dumps(forged) + "\n")
    rc = ledger.main([journal, "--verify"])   # returns 1 when an invalid line is present
    forged_caught = rc == 1

    print(f"[itest:ledger] nodes={len(agg)} token_counts={tok_counts} "
          f"all_sigs_valid={all_valid} balanced={balanced} forged_caught={forged_caught}",
          flush=True)
    os.environ.pop("DRIFT_JOURNAL", None)
    try:
        os.remove(journal)
    except OSError:
        pass
    good = all_valid and balanced and forged_caught
    print("[itest:ledger]",
          "PASS — ledger reconciles with the run; forged line rejected" if good else "FAIL",
          flush=True)
    return 0 if good else 1


def _wait_port(port: int, timeout: float = 30.0) -> bool:
    import socket
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return True
        except OSError:
            time.sleep(0.3)
    return False


def _expand_test(cfg, model_id, dtype, dev, args) -> int:
    """M12: a seed + joiners that gossip-join it; assert the seed learns the whole
    membership and the head, expanding from just the seed, splits across all of it
    with bitwise parity."""
    from .membership import PeerTable, fetch_peers

    N = args.expand
    print(f"[itest:expand] building in-process reference …", flush=True)
    ref = _build_inprocess(cfg)

    def _spawn(port, extra):
        env = dict(os.environ)
        env["DRIFT_IDENTITY_FILE"] = f"/tmp/drift_id_{port}.key"
        env["DRIFT_ADVERTISE_HOST"] = "127.0.0.1"  # all-localhost: advertise loopback
        return subprocess.Popen(
            [sys.executable, "-m", "drift.node", "--port", str(port), "--host", "127.0.0.1",
             "--quiet", "--no-advertise", *extra],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    seed_port = free_port()
    print(f"[itest:expand] seed on {seed_port}; {N - 1} joiner(s) gossip-join it …", flush=True)
    procs = [_spawn(seed_port, [])]
    _wait_port(seed_port)
    for _ in range(N - 1):
        p = free_port()
        procs.append(_spawn(p, ["--join", f"127.0.0.1:{seed_port}"]))
    try:
        # Wait until the seed's peer table has converged to all N members.
        members, t0 = [], time.time()
        while time.time() - t0 < 60:
            try:
                tbl = PeerTable(); tbl.merge(fetch_peers("127.0.0.1", seed_port))
                members = tbl.endpoints()
                if len(members) >= N:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        print(f"[itest:expand] seed learned {len(members)}/{N} members via gossip", flush=True)

        endpoints = _expand_members([{"name": "seed", "host": "127.0.0.1", "port": seed_port}])
        orch, plan = build_over_nodes(model_id, dtype, dev, endpoints, chain=True)
        print(f"[itest:expand] split across {len(plan)} discovered node(s)", flush=True)

        ok_all = True
        for prompt, n in _CASES[:2]:
            r = ref.generate(prompt, n, stop_on_eos=False, session_id=f"ref-{n}")["token_ids"]
            g = orch.generate(prompt, n, stop_on_eos=False, session_id=f"exp-{n}")["token_ids"]
            ok = (r == g); ok_all &= ok
            print(f"[itest:expand] {'PASS' if ok else 'FAIL'} n={n:>3} "
                  f"discovered={len(endpoints)}", flush=True)
        good = ok_all and len(members) >= N
        print("[itest:expand]",
              "PASS — gossip-discovered all members, split bitwise" if good else "FAIL", flush=True)
        return 0 if good else 1
    finally:
        _teardown(procs)


def _tamper_test(cfg, model_id, dtype, dev, args) -> int:
    """M11: run an ordinary generation with node K corrupting its output, and
    assert the head's live receipt verifier flags K as SUSPECT (no separate
    challenge). Then confirm an honest run of the same shape flags nobody."""
    tag = "tamper" + ("+thin" if args.thin else "")
    prompt, n = "Give me a short introduction to large language models.", 16

    print(f"[itest:{tag}] spawning {args.nodes} node(s); node {args.tamper} tampers …", flush=True)
    ports, procs = spawn_nodes(args.nodes, tamper_idx=args.tamper)
    try:
        endpoints = [{"name": f"n{i}", "host": "127.0.0.1", "port": p}
                     for i, p in enumerate(ports)]
        orch, plan = build_over_nodes(model_id, dtype, dev, endpoints, chain=True, thin=args.thin)
        victim = plan[args.tamper].get("pubkey")
        orch.generate(prompt, n, stop_on_eos=False, session_id="tamper")
        suspects = set(orch.verifier.suspects())
        caught = victim in suspects
        print(f"[itest:{tag}] verifier checked {orch.verifier.checked} tokens; "
              f"tampering node {args.tamper} (pub {victim[:12]}…) → "
              f"{'CAUGHT' if caught else 'MISSED'}; {len(suspects)} suspect(s)", flush=True)
    finally:
        _teardown(procs)

    # Honest baseline: same shape, no tamper → the verifier must flag NObody.
    print(f"[itest:{tag}] honest baseline (no tamper) …", flush=True)
    ports2, procs2 = spawn_nodes(args.nodes)
    try:
        endpoints = [{"name": f"n{i}", "host": "127.0.0.1", "port": p}
                     for i, p in enumerate(ports2)]
        orch2, _ = build_over_nodes(model_id, dtype, dev, endpoints, chain=True, thin=args.thin)
        orch2.generate(prompt, n, stop_on_eos=False, session_id="honest")
        clean = orch2.verifier.suspects()
        print(f"[itest:{tag}] honest run: {len(clean)} suspect(s) "
              f"over {orch2.verifier.checked} tokens", flush=True)
    finally:
        _teardown(procs2)

    good = caught and not clean
    print(f"[itest:{tag}]",
          "PASS — tamper caught on live traffic, honest run clean" if good else "FAIL", flush=True)
    return 0 if good else 1


def _kill_test(cfg, model_id, dtype, dev, args) -> int:
    """M9: start `nodes` active + 1 spare, kill node K mid-generation, and assert
    the recovered sequence is bitwise-identical to an uninterrupted reference."""
    tag = "kill" + ("+secure" if args.secure else "") + ("+chain" if args.chain else "")
    prompt, n = "Count from one to forty in words.", 48

    print(f"[itest:{tag}] building in-process reference …", flush=True)
    ref = _build_inprocess(cfg)
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
    ap.add_argument("--thin", action="store_true",
                    help="zero-weight head; embed+head move to the edge nodes (M10, implies chain)")
    ap.add_argument("--kill", type=int, default=None, metavar="K",
                    help="M9: kill node index K mid-generation; assert bitwise-identical recovery")
    ap.add_argument("--tamper", type=int, default=None, metavar="K",
                    help="M11: node K corrupts its output; assert the receipt verifier flags it")
    ap.add_argument("--expand", type=int, default=None, metavar="N",
                    help="M12: N nodes gossip-join a seed; assert the head discovers + splits all")
    ap.add_argument("--ledger", action="store_true",
                    help="M13: journal receipts during a run, then check the contribution tally")
    ap.add_argument("--int8", action="store_true",
                    help="M14: int8 wire — measure the byte savings + relaxed match-rate")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    model_id, dtype = cfg["model_id"], cfg.get("dtype", "float16")
    dev = pick_device(cfg.get("device"))

    if args.secure:
        # A throwaway network key, shared with the spawned nodes via the inherited
        # environment. Proves the AEAD channel is bitwise-transparent to parity.
        os.environ["DRIFT_NETWORK_KEY"] = binascii.hexlify(os.urandom(32)).decode()

    if args.int8:
        return _int8_test(cfg, model_id, dtype, dev, args)
    if args.ledger:
        return _ledger_test(cfg, model_id, dtype, dev, args)
    if args.expand is not None:
        return _expand_test(cfg, model_id, dtype, dev, args)
    if args.tamper is not None:
        return _tamper_test(cfg, model_id, dtype, dev, args)
    if args.kill is not None:
        return _kill_test(cfg, model_id, dtype, dev, args)

    topo = ("chain" if (args.chain or args.thin) else "star") + \
           ("+secure" if args.secure else "") + ("+thin" if args.thin else "")

    print(f"[itest:{topo}] building in-process reference …", flush=True)
    ref = _build_inprocess(cfg)

    print(f"[itest:{topo}] spawning {args.nodes} local node(s) …", flush=True)
    ports, procs = spawn_nodes(args.nodes)
    try:
        endpoints = [{"name": f"n{i}", "host": "127.0.0.1", "port": p}
                     for i, p in enumerate(ports)]
        orch, plan = build_over_nodes(model_id, dtype, dev, endpoints,
                                      chain=args.chain, thin=args.thin)
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
        # M11: honest traffic must leave the receipt verifier with zero suspects.
        suspects = orch.verifier.suspects() if orch.verifier else []
        clean = not suspects
        print(f"[itest:{topo}] receipts verified over {orch.verifier.checked if orch.verifier else 0} "
              f"tokens · suspects={len(suspects)}", flush=True)
        all_ok = all_ok and clean
        print(f"[itest:{topo}]",
              "ALL PASS — bitwise == in-process reference, receipts clean" if all_ok else "FAIL",
              flush=True)
        return 0 if all_ok else 1
    finally:
        _teardown(procs)


if __name__ == "__main__":
    sys.exit(main())
