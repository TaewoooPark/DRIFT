"""M2/M3 parity gate (spec §9 M2/M3).

Runs the split path (in-process for M2, TCP for M3) and asserts the greedy
token-id sequence is BITWISE equal to the M1 reference (reference_out.npz).
Also reports the fp32 max-abs-diff of the first-step logits for diagnostics
(see docs/05 — boundary diff ~0 but ids differ => bug after the shards).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .common import load_config
from .orchestrator import build_inprocess, build_socket


def _first_divergence(a: list, b: list):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DRIFT parity test")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["inprocess", "socket"], default="inprocess")
    ap.add_argument("--ports", help="comma-separated localhost ports (socket mode)")
    ap.add_argument("--ref", default="reference_out.npz")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ref = np.load(args.ref)
    ref_ids = ref["token_ids"].tolist()
    ref_first = ref["first_logits"]
    n = len(ref_ids)
    prompt = cfg["generation"]["prompt"]

    ports = [int(x) for x in args.ports.split(",")] if args.ports else None
    label = "M2 in-process" if args.mode == "inprocess" else "M3 TCP"
    print(f"[parity:{label}] loading split path ...", flush=True)
    orch = build_inprocess(cfg) if args.mode == "inprocess" else build_socket(cfg, ports)

    out = orch.generate(prompt, n, stop_on_eos=False)
    got = out["token_ids"]

    max_logit_diff = float(np.abs(ref_first - out["first_logits"]).max())
    div = _first_divergence(ref_ids, got)
    exact = got == ref_ids

    print(f"[parity:{label}] first-step logits max-abs-diff (fp32): {max_logit_diff:.3e}", flush=True)
    print(f"[parity:{label}] ref first10: {ref_ids[:10]}", flush=True)
    print(f"[parity:{label}] got first10: {got[:10]}", flush=True)
    if exact:
        print(f"[parity:{label}] PASS — {n}/{n} token ids match the reference bitwise", flush=True)
        return 0
    print(f"[parity:{label}] FAIL — first divergence at token index {div}", flush=True)
    print(f"[parity:{label}]   ref[{div}]={ref_ids[div] if div is not None else '-'} "
          f"got[{div}]={got[div] if (div is not None and div < len(got)) else '-'}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
