# Benchmarks — how DRIFT is measured

**How to benchmark DRIFT against similar tools, and what it scores.** Harness:
[`../drift/bench.py`](../drift/bench.py) · reproduce every number with `python -m drift.bench`.

---

## Why speed is the wrong axis to lead with

The obvious benchmark — `tokens/sec` against [Exo](https://github.com/exo-explore/exo) or
[llama.cpp RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) — is the one
DRIFT should *not* lead with, for two reasons:

1. **It is not where DRIFT wins.** DRIFT is correctness-first: an fp16 CPU round-trip over
   TCP + msgpack. On an Apple-only cluster, Exo's native `mx.distributed` path will usually
   post higher raw throughput. Leading with speed argues on the opponent's ground.
2. **On the axis DRIFT *does* win — heterogeneity — there is no competitor number to beat.**
   Exo cannot run Mac (MPS) ↔ Windows (CUDA) at all; vLLM / Megatron need NCCL. The
   comparison there is `✅ vs ❌`, a capability, not a faster/slower number.

So the numeric story lives on the **single-machine axis**, where (a) competitors *can* be run
for a fair head-to-head, (b) the whole thing fits on one modest Mac, and (c) DRIFT genuinely
leads. Four metrics, in priority order.

---

## The four metrics

**1 · Fidelity — the monopoly axis.** "Does distributing the model change the output?" No
other tool in this space measures or guarantees an answer. DRIFT does, against the
single-machine reference oracle:

| Sub-metric | Definition |
|---|---|
| Exact-match rate | fraction of greedy token ids equal to the reference, over all (prompt, position) pairs |
| First-step logit max-abs-diff (fp32) | `max\|logit_ref − logit_split\|` — the decision margin the argmax survives |
| KL divergence (nats) | `KL(softmax(ref) ‖ softmax(split))` — precision-independent distance |

The honest claim is precise: **token ids are bitwise-identical**, while the underlying logits
agree only to the **fp16 ULP** (a batched `lm_head` GEMM rounds the last row slightly
differently than a full-sequence forward). That the *decision* is invariant under that noise
is the robustness result — not a hand-wave that "everything is exactly zero."

**2 · Footprint — why you split at all.** Parameter bytes held per node, and — on an
accelerator — the **actual on-device allocation** measured by loading each node's slice
(`init_empty_weights` + selective safetensors). Pipeline splitting exists so a model too big
for one machine runs anyway; the number that proves it is the heaviest single node as a % of
the full model — now confirmed in real memory, not just parameter accounting.

**3 · Overhead — the price of a neutral wire.** DRIFT runs the **same decode loop** over two
transports: an in-process callable and the TCP protocol. So `TPOT(TCP) − TPOT(in-process)`
isolates the pure cost of the framework-neutral protocol — serialization + framing + loopback
— with the compute held identical. No competitor can produce this decomposition, because none
has a bit-identical in-process reference to subtract. TPOT is measured by subtraction,
`(T(n₂) − T(n₁)) / (n₂ − n₁)`, which cancels prefill and fixed per-call cost.

**4 · Wire — how thin the boundary is.** Exact msgpack-framed request bytes per token per hop
(4-byte length prefix + body). For a plain decoder this is `hidden_size × 2` bytes plus a few
ints. The headline is bytes-on-wire vs the model size it represents.

---

## Controls (so the numbers survive review)

- **One model, one precision.** Qwen2.5-1.5B-Instruct, fp16, everywhere. Never compare a
  quantized competitor's speed against fp16 fidelity without labelling both — quantization
  drift and distribution drift are **separate axes** and must not be conflated.
- **A published prompt set** — English, code, Korean; `n = 1, 40, 50, 60, 80, 180`.
- **Warmup + subtraction** for all timings (MPS compiles kernels on first use).
- **Depth over breadth** — one model measured exhaustively rather than many measured shallowly.

---

## Measured results

**Config:** Qwen2.5-1.5B-Instruct · fp16 · Apple MPS · 28 layers → `[0,14) / [14,28)` (both
shards on MPS for exactness). Single Mac. Timings are machine-specific; fidelity, footprint,
and wire are not.

### Fidelity — bitwise token parity, ULP-level logits

| Case | n | Exact match | logit max-abs-diff (fp32) | KL (nats) |
|---|---:|:---:|---:|---:|
| Explain quantum entanglement simply. | 80 | 80/80 | 7.81e-03 | 3.66e-13 |
| Write a haiku about the sea. | 40 | 40/40 | 7.81e-03 | 1.24e-11 |
| `def fibonacci(n):` | 60 | 60/60 | 3.91e-03 | 4.80e-12 |
| 한국어로 인공지능을 한 문장으로 설명해줘. | 50 | 50/50 | 7.81e-03 | 2.83e-11 |
| Hi. | 1 | 1/1 | 7.81e-03 | 9.43e-16 |
| Count from one to forty in words. | 180 | 180/180 | 7.81e-03 | 2.82e-10 |
| **Total** | **411** | **411/411 = 100.00%** | **≤ 7.81e-03** | **≤ 2.82e-10** |

**6/6 cases bitwise-identical** token ids to the single-machine oracle; logits agree to fp16
ULP; KL is numerically zero. Verified in-process; the TCP path is separately proven bitwise by
the socket parity test (the fp16 CPU round-trip is lossless).

### Footprint — no node holds the whole model

| Node | Holds | fp16 | % of full |
|---|---|---:|---:|
| orchestrator | embed + norm + lm_head (tied) | 0.47 GB | 15.1% |
| shard · mac | decoder layers [0, 14) | 1.31 GB | 42.4% |
| shard · windows | decoder layers [14, 28) | 1.31 GB | 42.4% |
| **full model** | — | **3.09 GB** | 100% |

The heaviest single node carries **42.4%** of the model — one 2× too large for either machine
alone runs across the pair. Since v0.10 these are also the **measured on-device allocations**:
`python -m drift.bench` loads each slice and reads the accelerator's allocated bytes, and the
heaviest node measures **1.31 GB on MPS**, matching the parameter-byte split exactly. Each node
loads only its slice, so the whole model is never resident on any one machine — and the parity
gate proves the sliced load stays bitwise-identical to the full load.

### Wire — a few KB against a few GB

| Quantity | Value |
|---|---|
| hidden_size | 1536 |
| decode request per token per hop | **3174 B (3.10 KB)** |
| full model weights | 3.09 GB |
| ratio (weights : per-token wire) | **≈ 970,000 ×** |

Only `hidden_states` (fp16) + `position_ids` + `input_ids` cross a boundary — the weights are
~10⁶× the per-token traffic.

### Overhead — the neutral protocol is nearly free

| Transport | TPOT | tok/s |
|---|---:|---:|
| in-process callable | 40.68 ms/token | 24.6 |
| TCP + msgpack (localhost, 2 hops) | 43.13 ms/token | 23.2 |
| **protocol overhead** | **+2.45 ms/token (~1.2 ms/hop) — within run-to-run noise** | — |

Both TPOTs come from the *same* run. The delta is **+2.45 ms/token (~1.2 ms/hop)** — a small,
real localhost round-trip cost, ~6% of the ~41 ms/token of compute. It is noisy at this scale:
an earlier run measured it slightly *negative* (the TCP path came out marginally faster), so
run-to-run the delta straddles a few ms either side of zero. The honest reading: at localhost
the neutral protocol costs about **a millisecond per hop** — dwarfed by compute. (A real LAN
adds RTT on top, unchanged by DRIFT. Average several runs for a robust point estimate.)

---

## Fair head-to-head protocol (competitors)

This harness **does not fabricate competitor numbers.** To run an honest head-to-head on
identical hardware:

1. **Same model, same precision** for every system, or label the precision on every row.
2. **Fidelity, measured per-system against its own single-machine baseline** — never compare
   raw outputs across frameworks (different kernels ≠ a bug). Report each system's
   `split-vs-single` gap and compare the *gaps*.
3. **Same prompt set, warmup, N repeats, p50/p90.**

What is known by construction, verifiable from each project's source (no run needed):

| System | Heterogeneous (MPS↔CUDA) | Split-vs-single fidelity | Node-to-node plane |
|---|:---:|---|---|
| **DRIFT** | ✅ | **measured: 100% token-bitwise** | neutral TCP + msgpack |
| Exo | ❌ (Apple-only; `mx.distributed`) | not measured/guaranteed; MLX numerics | MLX |
| llama.cpp RPC | ✅ | not guaranteed; quantized by default | ggml RPC |
| vLLM / Megatron PP | ❌ (NCCL) | N/A (datacenter) | torch.distributed / NCCL |

The comparative claim DRIFT can make **today**, without a competitor run: *it is the only
system here that both spans GPU vendors and proves its distributed output equals the
single-machine output, with a number.* A speed head-to-head is future work (needs the
competitor installed on the same box).

---

## Topology & wire (v1.0)

The v1 decentralization layer changes the *shape* of the traffic, not the fidelity. Measured on
Qwen2.5-1.5B (`hidden = 1536`), per token per hop:

| Axis | Star (default) | Chain (`--chain`) |
|---|---:|---:|
| tensor crossings / token | `2N` | `N + 1` |
| head data-plane bandwidth | `O(N)` | `O(1)` |
| fp16 bytes / hop | 3 072 | 3 072 |
| **int8** bytes / hop (`--int8`) | — | **1 560 (51 %)** |

int8 is lossy (group-wise, 128-dim blocks): measured token match-rate vs the fp16 reference was
~67 % across two prompts (one 100 %, one drifting at token 11) — it runs under the *relaxed* gate,
never the bitwise one. The fp16 chain and thin-head paths remain **bitwise == the single machine**.
Reproduce the topology/wire gates with `python -m drift.itest --nodes N [--chain|--thin|--int8]`.

---

## Reproduce

```bash
python -m drift.bench                 # full run (fidelity, footprint, wire, overhead)
python -m drift.bench --quick         # drop the n=180 fidelity case
python -m drift.bench --no-socket     # skip the TCP overhead (no server spawn) on low RAM
python -m drift.bench --json out.json # also dump raw results

python -m drift.itest --nodes 2 --chain          # real-node bitwise gate (peer-to-peer)
python -m drift.itest --nodes 2 --int8           # measured int8 wire savings + fidelity
python -m drift.itest --nodes 2 --kill 1 --chain # bitwise failover after a mid-run kill
```

Footprint, wire, and fidelity are deterministic and machine-independent. Overhead / TPOT depend
on the host; re-run locally for your own numbers.
