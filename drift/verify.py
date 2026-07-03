"""Trustless spot-check verification (grounded in the M4 measurement).

M4 showed an honest node's output tracks the single machine within a small,
known envelope. So a head can *challenge* a node with a fixed input and check the
hidden state it returns against a locally-computed reference within a tolerance —
a node whose output falls outside the envelope is faulty or lying. This is
redundant compute on ONE challenge (a cheap spot-check), not re-running the whole
model, and it is the seed of trustless verification: the step that turns "trusted
workers" into "a node you don't have to trust."

    python -m drift.verify --nodes 127.0.0.1:52600,127.0.0.1:52601 [--tol 0.05]

A same-device honest node lands at ~0 (bitwise); a tampered node blows past the
tolerance. Across GPU vendors the honest envelope is larger — raise --tol to the
value M4 measured for your models.
"""

from __future__ import annotations

import argparse
import sys

import torch

from .common import load_config, pick_device
from .engine_torch import TorchShardEngine
from .run import _parse_nodes, build_over_nodes

_TORCH_DTYPE = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def _challenge(hidden_size: int, device: str, dtype: torch.dtype, S: int = 8):
    """A fixed, reproducible challenge — no RNG, so the reference is deterministic."""
    base = torch.linspace(-1.0, 1.0, hidden_size)
    rows = [torch.roll(base, i * 7) * (0.5 + 0.05 * i) for i in range(S)]
    hidden = torch.stack(rows).unsqueeze(0).to(device=device, dtype=dtype)
    return hidden, list(range(S)), [1] * S


def verify_node(model_id, dtype, device, transport, name, start, end, tol):
    """Challenge one node and compare its output to a local oracle for its range."""
    td = _TORCH_DTYPE[dtype]
    oracle = TorchShardEngine(model_id, start, end, device, dtype, name=f"oracle:{name}")
    oracle.load()
    hidden, pos, ids = _challenge(oracle.config.hidden_size, device, td)

    sess = f"verify-{name}"
    ref = oracle.forward(sess, hidden.clone(), pos, ids, "prefill")
    oracle.reset(sess)
    got = transport.forward(name, sess, hidden.clone(), pos, ids, "prefill")
    transport.reset(name, sess)

    diff = float((ref.float().cpu() - got.float().cpu()).abs().max())
    return {"node": name, "range": [start, end], "max_abs_diff": diff,
            "tol": tol, "verdict": "HONEST" if diff <= tol else "SUSPECT"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DRIFT trustless node verification (spot-check)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--nodes", required=True, help="host:port,... of running drift nodes")
    ap.add_argument("--model", help="override model_id")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max-abs hidden diff allowed (same-device honest ~0; raise for cross-vendor)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    model_id = args.model or cfg["model_id"]
    dtype = cfg.get("dtype", "float16")
    device = pick_device(cfg.get("device"))

    endpoints = _parse_nodes(args.nodes)
    orch, plan = build_over_nodes(model_id, dtype, device, endpoints)  # configure + plan

    print(f"[verify] challenging {len(plan)} node(s) · tol={args.tol}", flush=True)
    results, ok = [], True
    for p in plan:
        r = verify_node(model_id, dtype, device, orch.transport, p["name"],
                        p["start"], p["end"], args.tol)
        results.append(r)
        ok = ok and r["verdict"] == "HONEST"
        print(f"[verify] {r['verdict']:>7}  node {p['name']} "
              f"[{p['start']}:{p['end']}) device={p['device']}  "
              f"max|Δ|={r['max_abs_diff']:.3e}", flush=True)
    print("[verify]", "ALL HONEST" if ok else "SUSPECT NODE(S) FOUND", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
