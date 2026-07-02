"""M2/M3/M4 parity gate (spec §9).

Runs the split path (in-process for M2, TCP for M3) and asserts the greedy
token-id sequence is BITWISE equal to the M1 reference (reference_out.npz).
Also reports the fp32 max-abs-diff of the first-step logits for diagnostics
(see docs/05 — boundary diff ~0 but ids differ => bug after the shards).

Same-device runs are held to the strict bitwise gate. For the cross-device M4
step (Mac MPS + Windows CUDA), the two vendors' fp16 kernels round differently,
so greedy decoding may diverge in *later* tokens — `--prefix-match K` switches to
a relaxed gate that requires only the first K ids to match. Divergence WITHIN the
first K is a real bug (bisect), not float noise.
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


_SELFTEST_CASES = [
    ("Explain quantum entanglement simply.", 80),
    ("Write a haiku about the sea.", 40),
    ("def fibonacci(n):", 60),
    ("한국어로 인공지능을 한 문장으로 설명해줘.", 50),
    ("Hi.", 1),                                  # edge: single-token generation
    ("Count from one to forty in words.", 180),  # edge: long decode sequence
]


def selftest(cfg: dict) -> int:
    """In-process bitwise parity across several prompts/lengths vs a clean
    reference model — guards against overfitting to one prompt (no npz needed)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    from .common import build_input_ids
    from .orchestrator import build_inprocess

    dev = cfg.get("device", "cpu")
    refm = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=torch.float16).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    orch = build_inprocess(cfg)  # separate shared model; split path

    @torch.no_grad()
    def ref_gen(prompt, n):
        ids = build_input_ids(tok, prompt).to(dev)
        cache = DynamicCache(config=refm.config)
        out = refm(input_ids=ids, past_key_values=cache, use_cache=True)
        nxt = int(out.logits[:, -1, :].argmax(-1))
        gen = [nxt]
        for _ in range(n - 1):
            out = refm(input_ids=torch.tensor([[nxt]], device=dev),
                       past_key_values=cache, use_cache=True)
            nxt = int(out.logits[:, -1, :].argmax(-1))
            gen.append(nxt)
        return gen

    all_ok = True
    for prompt, n in _SELFTEST_CASES:
        r = ref_gen(prompt, n)
        g = orch.generate(prompt, n, stop_on_eos=False)["token_ids"]
        ok = r == g
        all_ok &= ok
        print(f"[selftest] {'PASS' if ok else 'FAIL'} n={n:>3} "
              f"first_div={_first_divergence(r, g)} prompt={prompt[:34]!r}", flush=True)
    print("[selftest]", "ALL PASS" if all_ok else "FAIL", flush=True)
    return 0 if all_ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DRIFT parity test")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["inprocess", "socket"], default="inprocess")
    ap.add_argument("--ports", help="comma-separated localhost ports (socket mode)")
    ap.add_argument("--ref", default="reference_out.npz")
    ap.add_argument("--selftest", action="store_true",
                    help="multi-prompt in-process parity vs a clean reference (no npz)")
    ap.add_argument("--prefix-match", type=int, default=None, metavar="K",
                    help="relaxed cross-device gate: require the first K token ids to match "
                         "the reference; later divergence from MPS↔CUDA fp16 rounding is "
                         "allowed. Divergence WITHIN the first K is a real bug, not float noise.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.selftest:
        return selftest(cfg)
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

    # Relaxed gate (cross-device): early tokens must match; later fp16-rounding
    # divergence between MPS and CUDA kernels is expected, not a bug.
    if args.prefix_match is not None:
        k = min(args.prefix_match, len(ref_ids), len(got))
        matched = sum(1 for i in range(min(len(ref_ids), len(got))) if ref_ids[i] == got[i])
        if ref_ids[:k] == got[:k]:
            print(f"[parity:{label}] RELAXED PASS — first {k}/{k} ids match; "
                  f"{matched}/{n} total (later drift allowed for cross-device fp16)", flush=True)
            return 0
        print(f"[parity:{label}] RELAXED FAIL — diverged at token {div} < {k} "
              f"(a real bug, not float noise): ref={ref_ids[div]} got={got[div]}", flush=True)
        return 1

    if exact:
        print(f"[parity:{label}] PASS — {n}/{n} token ids match the reference bitwise", flush=True)
        return 0
    print(f"[parity:{label}] FAIL — first divergence at token index {div}", flush=True)
    print(f"[parity:{label}]   ref[{div}]={ref_ids[div] if div is not None else '-'} "
          f"got[{div}]={got[div] if (div is not None and div < len(got)) else '-'}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
