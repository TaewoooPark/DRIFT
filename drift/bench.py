"""DRIFT benchmark harness (see docs/08-benchmark-plan.md).

Measures the axes where a *correct* split genuinely wins — all reproducible on
ONE machine, so no second node or competitor install is required to produce the
headline numbers:

  1. Fidelity   — split path vs the single-machine reference oracle:
                  exact-match token rate, first-step logit max-abs-diff (fp32),
                  KL divergence. DRIFT's monopoly axis (nobody else measures or
                  guarantees this).
  2. Footprint  — decoder-layer parameter bytes per node. Pipeline splitting's
                  raison d'etre: no single node holds the whole model.
  3. Overhead   — TPOT delta between M2 (in-process) and M3 (TCP localhost).
                  The same decode loop runs over both transports, so the delta
                  is the *pure* cost of the neutral protocol, nothing else.
  4. Wire       — bytes on the wire per token per hop (msgpack-framed request).

This harness deliberately does NOT fabricate competitor numbers. docs/08 gives
the fair head-to-head protocol for running Exo / llama.cpp RPC on identical
hardware; until that is run, the honest comparative claim is the capability
matrix (README) plus DRIFT's absolute fidelity being provably perfect.

Usage:
    python -m drift.bench                 # full run
    python -m drift.bench --quick         # drop the long n=180 fidelity case
    python -m drift.bench --no-socket     # skip the M3 (server-spawn) overhead
    python -m drift.bench --json out.json # also write raw results
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np

from .common import build_input_ids, load_config


# Mirrors parity_test._SELFTEST_CASES so the benchmark reports on the exact
# prompt set the correctness gate already publishes.
_CASES = [
    ("Explain quantum entanglement simply.", 80),
    ("Write a haiku about the sea.", 40),
    ("def fibonacci(n):", 60),
    ("한국어로 인공지능을 한 문장으로 설명해줘.", 50),
    ("Hi.", 1),
    ("Count from one to forty in words.", 180),
]


def _sync(device: str) -> None:
    """Flush the async accelerator queue so wall-clock timing is honest."""
    import torch

    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _free() -> None:
    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _device_allocated(device: str):
    """Current accelerator bytes allocated, or None on CPU (no allocator counter)."""
    import torch

    if device == "mps" and torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory()
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    return None


def _kl_divergence(ref_logits: np.ndarray, split_logits: np.ndarray) -> float:
    """KL(P_ref || P_split) over the first-step vocab distribution, in nats.

    Bit-identical logits -> exactly 0.0. Numerically stable log-softmax.
    """
    a = ref_logits.astype(np.float64)
    b = split_logits.astype(np.float64)
    a = a - a.max()
    b = b - b.max()
    log_p = a - np.log(np.exp(a).sum())
    log_q = b - np.log(np.exp(b).sum())
    p = np.exp(log_p)
    return float(np.sum(p * (log_p - log_q)))


# --------------------------------------------------------------------- fidelity
def _reference_generator(cfg: dict):
    """A clean, unmodified full model for greedy oracle generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    dev = cfg.get("device", "cpu")
    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[cfg.get("dtype", "float16")]
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=dtype).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])

    @torch.no_grad()
    def gen(prompt: str, n: int):
        ids = build_input_ids(tok, prompt).to(dev)
        cache = DynamicCache(config=model.config)
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1, :]
        first = logits[0].detach().float().cpu().numpy()
        nxt = int(logits.argmax(-1))
        seq = [nxt]
        for _ in range(n - 1):
            out = model(input_ids=torch.tensor([[nxt]], device=dev),
                        past_key_values=cache, use_cache=True)
            nxt = int(out.logits[:, -1, :].argmax(-1))
            seq.append(nxt)
        return seq, first

    return gen, model


def measure_fidelity(cfg: dict, orch, cases) -> dict:
    """Compare the in-process split path to the single-machine reference."""
    ref_gen, ref_model = _reference_generator(cfg)

    per_case = []
    total_tokens = 0
    total_match = 0
    max_logit_diff = 0.0
    max_kl = 0.0
    curve = None  # (positions, agreement) for the longest case

    for prompt, n in cases:
        ref_ids, ref_first = ref_gen(prompt, n)
        out = orch.generate(prompt, n, stop_on_eos=False)
        got, got_first = out["token_ids"], out["first_logits"]

        matches = sum(1 for r, g in zip(ref_ids, got) if r == g)
        exact = ref_ids == got
        ldiff = float(np.abs(ref_first - got_first).max())
        kl = _kl_divergence(ref_first, got_first)

        total_tokens += len(ref_ids)
        total_match += matches
        max_logit_diff = max(max_logit_diff, ldiff)
        max_kl = max(max_kl, kl)
        per_case.append({
            "prompt": prompt, "n": n, "exact": exact,
            "match_rate": matches / len(ref_ids), "logit_max_abs_diff": ldiff, "kl": kl,
        })
        if n == max(c[1] for c in cases):
            # per-position agreement for the longest decode (flat 1.0 when bitwise)
            agree = [1.0 if (i < len(got) and ref_ids[i] == got[i]) else 0.0
                     for i in range(len(ref_ids))]
            curve = {"n": n, "agreement": agree}
        print(f"[fidelity] {'PASS' if exact else 'FAIL'} n={n:>3} "
              f"match={matches}/{len(ref_ids)} logit_diff={ldiff:.3e} kl={kl:.3e} "
              f"prompt={prompt[:32]!r}", flush=True)

    del ref_model
    _free()
    return {
        "exact_match_rate": total_match / total_tokens,
        "tokens_compared": total_tokens,
        "cases_bitwise_equal": sum(1 for c in per_case if c["exact"]),
        "cases_total": len(per_case),
        "max_logit_abs_diff_fp32": max_logit_diff,
        "max_kl_nats": max_kl,
        "per_case": per_case,
        "curve": curve,
    }


# -------------------------------------------------------------------- footprint
def measure_footprint(cfg: dict) -> dict:
    """Per-node parameter bytes (the theoretical split) plus, on an accelerator,
    the MEASURED on-device allocation from actually loading each node's slice."""
    import torch
    from transformers import AutoModelForCausalLM

    device = cfg.get("device", "cpu")
    dtype_name = cfg.get("dtype", "float16")
    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[dtype_name]
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=dtype).eval()
    inner = model.model

    # Unique parameter bytes (dedupe tied weights by storage pointer).
    seen: dict[int, int] = {}
    for p in model.parameters():
        seen[p.data_ptr()] = p.numel() * p.element_size()
    total_bytes = sum(seen.values())

    def layer_bytes(i: int) -> int:
        return sum(p.numel() * p.element_size() for p in inner.layers[i].parameters())

    shards = []
    layer_total = 0
    for s in cfg["shards"]:
        b = sum(layer_bytes(i) for i in range(s["start_layer"], s["end_layer"]))
        layer_total += b
        shards.append({"name": s["name"],
                       "layers": [s["start_layer"], s["end_layer"]], "bytes": b})
    orch_bytes = total_bytes - layer_total  # embed + norm + (untied) head

    nodes = [{"role": "orchestrator", "bytes": orch_bytes}] + \
            [{"role": f"shard:{s['name']}", "bytes": s["bytes"]} for s in shards]
    for nd in nodes:
        nd["pct_of_full"] = nd["bytes"] / total_bytes
    heaviest = max(nodes, key=lambda x: x["bytes"])
    tied = bool(getattr(model.config, "tie_word_embeddings", False))

    del model
    _free()

    # Measured on-device allocation: actually load each node's slice with the
    # sliced loader and read the accelerator's allocated bytes. Turns the
    # footprint from a theoretical split into a reproducible measurement, and
    # cross-checks that a node really holds only its fraction in memory.
    heaviest_meas = None
    if _device_allocated(device) is not None:
        from transformers import AutoConfig

        from .loader import build_sliced

        tie = bool(getattr(AutoConfig.from_pretrained(cfg["model_id"]),
                           "tie_word_embeddings", False))
        head_keep = ["model.embed_tokens.", "model.norm."] + ([] if tie else ["lm_head."])
        plan = {"orchestrator": (head_keep, False, tie)}
        for s in cfg["shards"]:
            plan[f"shard:{s['name']}"] = (
                [f"model.layers.{i}." for i in range(s["start_layer"], s["end_layer"])],
                True, False)
        for nd in nodes:
            keep, need_rot, tie_n = plan[nd["role"]]
            _free()
            base = _device_allocated(device)
            lm, _ = build_sliced(cfg["model_id"], dtype_name, device, keep,
                                 need_rotary=need_rot, tie=tie_n)
            _sync(device)
            nd["measured_device_bytes"] = _device_allocated(device) - base
            del lm
            _free()
        heaviest_meas = max(nd["measured_device_bytes"] for nd in nodes)

    msg = (f"[footprint] full={total_bytes/1e9:.2f} GB · heaviest node "
           f"{heaviest['role']}={heaviest['bytes']/1e9:.2f} GB "
           f"({heaviest['pct_of_full']*100:.1f}% of full)")
    if heaviest_meas is not None:
        msg += f" · measured heaviest {heaviest_meas/1e9:.2f} GB on {device}"
    print(msg, flush=True)
    return {
        "full_model_bytes": total_bytes,
        "tie_word_embeddings": tied,
        "nodes": nodes,
        "heaviest_node_pct": heaviest["pct_of_full"],
        "measured_device": device if heaviest_meas is not None else None,
        "measured_heaviest_bytes": heaviest_meas,
        "measured_heaviest_pct": (heaviest_meas / total_bytes) if heaviest_meas else None,
    }


# ------------------------------------------------------------------------- wire
def measure_wire(cfg: dict) -> dict:
    """Exact msgpack-framed request bytes per hop for prefill and decode."""
    import msgpack
    from transformers import AutoConfig

    hcfg = AutoConfig.from_pretrained(cfg["model_id"])
    D = getattr(hcfg, "hidden_size", None) or hcfg.text_config.hidden_size
    dtype = cfg.get("dtype", "float16")
    esize = {"float16": 2, "float32": 4, "bfloat16": 2}[dtype]

    def frame_bytes(S: int, mode: str) -> int:
        msg = {
            "type": mode, "session_id": "s0", "seq_id": 1,
            "shape": [1, S, D], "dtype": dtype,
            "position_ids": list(range(S)), "input_ids": [0] * S,
            "tensor": b"\x00" * (S * D * esize),
        }
        return 4 + len(msgpack.packb(msg, use_bin_type=True))  # 4B length prefix

    decode = frame_bytes(1, "decode")
    print(f"[wire] hidden_size={D} dtype={dtype} · decode request "
          f"{decode} B/token/hop ({decode/1024:.2f} KB)", flush=True)
    return {
        "hidden_size": D, "dtype": dtype,
        "decode_bytes_per_token_per_hop": decode,
        "decode_kb_per_token_per_hop": decode / 1024,
        "prefill_bytes_per_token_per_hop_at_S32": frame_bytes(32, "prefill") / 32,
    }


# --------------------------------------------------------------------- overhead
def _tpot(orch, prompt: str, n1: int, n2: int, device: str, warmup: int = 1) -> float:
    """Isolate per-decode-token latency: (T(n2) - T(n1)) / (n2 - n1) ms.

    Subtracting the two runs cancels prefill + fixed per-call overhead, leaving
    the marginal cost of one decoded token.
    """
    for _ in range(warmup):
        orch.generate(prompt, n1, stop_on_eos=False)
    _sync(device)
    t0 = time.perf_counter()
    orch.generate(prompt, n1, stop_on_eos=False)
    _sync(device)
    t1 = time.perf_counter()
    orch.generate(prompt, n2, stop_on_eos=False)
    _sync(device)
    t2 = time.perf_counter()
    return ((t2 - t1) - (t1 - t0)) / (n2 - n1) * 1000.0


def measure_overhead(cfg: dict, orch_inproc, do_socket: bool) -> dict:
    device = cfg.get("device", "cpu")
    prompt = "Explain the theory of relativity in one paragraph."
    n1, n2 = 8, 40

    tpot_m2 = _tpot(orch_inproc, prompt, n1, n2, device)
    print(f"[overhead] M2 in-process TPOT = {tpot_m2:.2f} ms/token", flush=True)
    result = {"tpot_m2_ms": tpot_m2, "n_hops": len(cfg["shards"])}

    if not do_socket:
        return result

    # Spawn the shard servers (each loads only its own slice via the sliced
    # loader; the in-process orchestrator's model is freed by the caller).
    procs = []
    try:
        ports = [s["port"] for s in cfg["shards"]]
        for s in cfg["shards"]:
            env = dict(os.environ, DRIFT_PORT=str(s["port"]))
            p = subprocess.Popen(
                [sys.executable, "-m", "drift.shard_server",
                 "--name", s["name"], "--start", str(s["start_layer"]),
                 "--end", str(s["end_layer"]), "--device", s.get("device", device),
                 "--preload"],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            procs.append(p)

        # Build the real orchestrator first and reuse ITS transport for the
        # readiness ping. The shard server serves ONE connection at a time, so a
        # separate ping connection would occupy that loop and the orchestrator's
        # own connection would never be accepted -> deadlock. One persistent
        # connection per shard, used for both ping and timing, is the contract.
        from .orchestrator import build_socket

        orch_socket = build_socket(cfg, ports)
        transport = orch_socket.transport
        deadline = time.time() + 180
        ready = False
        while time.time() < deadline:
            try:
                if all(transport.ping(s["name"]).get("ok") for s in cfg["shards"]):
                    ready = True
                    break
            except Exception:
                for sk in transport.socks.values():
                    try:
                        sk.close()
                    except Exception:
                        pass
                transport.socks.clear()
                time.sleep(1.0)
        if not ready:
            raise RuntimeError("shard servers did not become ready in time")

        tpot_m3 = _tpot(orch_socket, prompt, n1, n2, device)
        delta = tpot_m3 - tpot_m2
        result.update({
            "tpot_m3_ms": tpot_m3,
            "protocol_overhead_ms_per_token": delta,
            "protocol_overhead_ms_per_hop": delta / len(cfg["shards"]),
            "tok_per_s_m2": 1000.0 / tpot_m2,
            "tok_per_s_m3": 1000.0 / tpot_m3,
        })
        print(f"[overhead] M3 TCP TPOT = {tpot_m3:.2f} ms/token · "
              f"protocol overhead = +{delta:.2f} ms/token "
              f"(+{delta/len(cfg['shards']):.2f}/hop)", flush=True)
    except Exception as e:  # degrade gracefully on limited-memory machines
        result["socket_error"] = f"{type(e).__name__}: {e}"
        print(f"[overhead] M3 socket measurement unavailable: {e}", flush=True)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
    return result


# ------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DRIFT benchmark harness")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true", help="drop the long n=180 case")
    ap.add_argument("--no-socket", action="store_true", help="skip M3 server spawn")
    ap.add_argument("--json", help="write raw results to this path")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cases = [c for c in _CASES if not (args.quick and c[1] == 180)]
    device = cfg.get("device", "cpu")
    print(f"=== DRIFT bench · model={cfg['model_id']} dtype={cfg.get('dtype')} "
          f"device={device} shards={len(cfg['shards'])} ===", flush=True)

    results: dict = {"model_id": cfg["model_id"], "dtype": cfg.get("dtype"),
                     "device": device, "shards": cfg["shards"]}

    # 2 · footprint (cheap; do first, frees its model)
    results["footprint"] = measure_footprint(cfg)
    # 4 · wire (no weights loaded)
    results["wire"] = measure_wire(cfg)

    # 1 + 3 · fidelity and M2 overhead share one in-process orchestrator
    from .orchestrator import build_inprocess

    orch = build_inprocess(cfg)
    results["fidelity"] = measure_fidelity(cfg, orch, cases)
    results["overhead"] = measure_overhead(cfg, orch, do_socket=not args.no_socket)
    del orch
    _free()

    # ---- summary
    f = results["fidelity"]
    fp = results["footprint"]
    w = results["wire"]
    o = results["overhead"]
    print("\n===================== SUMMARY =====================", flush=True)
    print(f"Fidelity   : {f['exact_match_rate']*100:.2f}% exact-match "
          f"({f['cases_bitwise_equal']}/{f['cases_total']} cases bitwise) · "
          f"logit diff {f['max_logit_abs_diff_fp32']:.2e} · KL {f['max_kl_nats']:.2e}", flush=True)
    fp_line = (f"Footprint  : heaviest node = {fp['heaviest_node_pct']*100:.1f}% of the "
               f"full {fp['full_model_bytes']/1e9:.2f} GB model")
    if fp.get("measured_heaviest_bytes"):
        fp_line += (f" · measured {fp['measured_heaviest_bytes']/1e9:.2f} GB "
                    f"on {fp['measured_device']}")
    print(fp_line, flush=True)
    print(f"Wire       : {w['decode_kb_per_token_per_hop']:.2f} KB/token/hop", flush=True)
    if "tpot_m3_ms" in o:
        print(f"Overhead   : M2 {o['tpot_m2_ms']:.2f} → M3 {o['tpot_m3_ms']:.2f} ms/token "
              f"(protocol +{o['protocol_overhead_ms_per_token']:.2f} ms/token)", flush=True)
    else:
        print(f"Overhead   : M2 {o['tpot_m2_ms']:.2f} ms/token (M3 skipped)", flush=True)
    print("==================================================", flush=True)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"[bench] wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
