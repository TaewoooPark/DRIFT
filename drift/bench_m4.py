"""M4 cross-machine benchmark — a real Mac (MPS) head + remote CUDA node(s).

Unlike `drift.bench` (single machine, bitwise), M4 spans GPU vendors, so the
kernels round fp16 differently and greedy decoding drifts in *later* tokens.
The meaningful M4 metrics are therefore the RELAXED ones:

  * where the split path first diverges from the single-machine reference
    (early match ⇒ the split is correct; later drift ⇒ fp16 vendor noise),
  * the prefix-match rate and first-step logit gap,
  * the decoded text of both paths (so a human can see it stays coherent),
  * real cross-machine latency (TPOT) over the actual link.

The reference is the full model on the head's own device (MPS on the Mac), i.e.
"does splitting across two GPU vendors reproduce running it all on my Mac?"

Runbook
-------
1. Start a CUDA `drift node` somewhere reachable and note its address. On Colab:
   `python scripts/colab_node.py` (see that file) prints a tunnel `host:port`.
2. (optional, to use BOTH GPUs) start a local MPS node on the Mac:
   `drift node --port 52600`
3. On the Mac, run this benchmark, remote node last so it takes the back half:
   `python -m drift.bench_m4 --nodes 127.0.0.1:52600,<tunnel-host>:<port> --json m4_results.json`
   or a single remote CUDA node holding all layers:
   `python -m drift.bench_m4 --nodes <tunnel-host>:<port> --json m4_results.json`
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .common import build_input_ids, load_config, pick_device
from .run import _parse_nodes, build_over_nodes

_CASES = [
    ("Give me a short introduction to large language models.", 50),
    ("def fibonacci(n):", 40),
    ("한국어로 인공지능을 한 문장으로 설명해줘.", 40),
]


def _reference(model_id: str, dtype: str, device: str):
    """Full model on the head's device — the single-machine oracle for M4."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    td = {"float16": torch.float16, "float32": torch.float32,
          "bfloat16": torch.bfloat16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=td).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_id)

    @torch.no_grad()
    def gen(prompt: str, n: int):
        ids = build_input_ids(tok, prompt).to(device)
        cache = DynamicCache(config=model.config)
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1, :]
        first = logits[0].detach().float().cpu().numpy()
        nx = int(logits.argmax(-1))
        seq = [nx]
        for _ in range(n - 1):
            out = model(input_ids=torch.tensor([[nx]], device=device),
                        past_key_values=cache, use_cache=True)
            nx = int(out.logits[:, -1, :].argmax(-1))
            seq.append(nx)
        return seq, first

    return gen, tok


def _first_divergence(a: list, b: list):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _tpot(orch, prompt: str, n1: int, n2: int) -> float:
    """Per-decode-token latency by subtraction — cancels prefill + fixed cost."""
    orch.generate(prompt, n1, stop_on_eos=False)  # warmup
    t0 = time.perf_counter()
    orch.generate(prompt, n1, stop_on_eos=False)
    t1 = time.perf_counter()
    orch.generate(prompt, n2, stop_on_eos=False)
    t2 = time.perf_counter()
    return ((t2 - t1) - (t1 - t0)) / (n2 - n1) * 1000.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DRIFT M4 cross-machine benchmark")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--nodes", required=True,
                    help="host:port,... of running drift nodes (include the remote CUDA one)")
    ap.add_argument("--model", help="override model_id")
    ap.add_argument("--prefix", type=int, default=16, help="K for the relaxed prefix-match report")
    ap.add_argument("--json", help="also write raw results here")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    model_id = args.model or cfg["model_id"]
    dtype = cfg.get("dtype", "float16")
    head_device = pick_device(cfg.get("device"))

    endpoints = _parse_nodes(args.nodes)
    print(f"[m4] head={head_device} · {len(endpoints)} node(s) · splitting {model_id} …", flush=True)
    orch, plan = build_over_nodes(model_id, dtype, head_device, endpoints)
    for p in plan:
        print(f"  node {p['host']}:{p['port']}  layers [{p['start']}:{p['end']})  device={p['device']}",
              flush=True)
    node_devices = [p["device"] for p in plan]
    is_cross = len(set(node_devices + [head_device])) > 1

    print("[m4] building the single-machine reference on the head …", flush=True)
    ref_gen, tok = _reference(model_id, dtype, head_device)

    K = args.prefix
    per_case = []
    for prompt, n in _CASES:
        r_ids, r_first = ref_gen(prompt, n)
        out = orch.generate(prompt, n, stop_on_eos=False)
        g_ids, g_first = out["token_ids"], out["first_logits"]
        div = _first_divergence(r_ids, g_ids)
        k = min(K, len(r_ids), len(g_ids))
        prefix_ok = r_ids[:k] == g_ids[:k]
        matches = sum(1 for x, y in zip(r_ids, g_ids) if x == y)
        ldiff = float(np.abs(r_first - g_first).max())
        per_case.append({
            "prompt": prompt, "n": n,
            "first_divergence": div,
            "prefix_match": {"k": k, "ok": prefix_ok},
            "match_rate": matches / len(r_ids),
            "exact": r_ids == g_ids,
            "logit_max_abs_diff": ldiff,
            "ref_text": tok.decode(r_ids),
            "split_text": out["text"],
        })
        tag = "exact" if r_ids == g_ids else f"diverges@{div}"
        print(f"[m4] n={n:>3} {tag:>13} · prefix[{k}]={'ok' if prefix_ok else 'FAIL'} "
              f"· match={matches}/{len(r_ids)} · logitΔ={ldiff:.3e} · {prompt[:32]!r}", flush=True)

    print("[m4] timing cross-machine TPOT …", flush=True)
    try:
        tpot = _tpot(orch, "Explain the theory of relativity in one paragraph.", 8, 32)
        latency = {"tpot_ms": tpot, "tok_per_s": 1000.0 / tpot, "n1": 8, "n2": 32}
        print(f"[m4] cross-machine TPOT = {tpot:.1f} ms/token ({1000.0/tpot:.2f} tok/s)", flush=True)
    except Exception as e:  # a flaky link shouldn't lose the fidelity numbers
        latency = {"error": f"{type(e).__name__}: {e}"}
        print(f"[m4] TPOT unavailable: {e}", flush=True)

    hidden = None
    try:
        from transformers import AutoConfig
        hc = AutoConfig.from_pretrained(model_id)
        hidden = getattr(hc, "hidden_size", None) or hc.text_config.hidden_size
    except Exception:
        pass

    results = {
        "model_id": model_id, "dtype": dtype, "head_device": head_device,
        "cross_vendor": is_cross, "plan": plan,
        "fidelity_relaxed": {
            "per_case": per_case,
            "min_first_divergence": min([c["first_divergence"] for c in per_case
                                         if c["first_divergence"] is not None], default=None),
            "all_prefix_ok": all(c["prefix_match"]["ok"] for c in per_case),
            "any_exact": any(c["exact"] for c in per_case),
            "mean_match_rate": sum(c["match_rate"] for c in per_case) / len(per_case),
        },
        "latency": latency,
        "wire": None if hidden is None else {
            "hidden_size": hidden,
            "decode_bytes_per_token_per_hop": hidden * (2 if dtype != "float32" else 4),
        },
    }

    fr = results["fidelity_relaxed"]
    print("\n===================== M4 SUMMARY =====================", flush=True)
    print(f"Path      : head {head_device} + nodes {node_devices}"
          f"{'  (cross-vendor)' if is_cross else ''}", flush=True)
    print(f"Fidelity  : prefix[{K}] {'all OK' if fr['all_prefix_ok'] else 'SOME FAIL'} · "
          f"first divergence ≥ {fr['min_first_divergence']} · "
          f"mean match {fr['mean_match_rate']*100:.1f}%", flush=True)
    if "tpot_ms" in latency:
        print(f"Latency   : {latency['tpot_ms']:.1f} ms/token ({latency['tok_per_s']:.2f} tok/s), cross-machine",
              flush=True)
    print("=====================================================", flush=True)
    if is_cross and fr["all_prefix_ok"]:
        print("[m4] cross-vendor split tracks the single machine for the prefix, then drifts on "
              "fp16 noise — the expected M4 result.", flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[m4] wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
