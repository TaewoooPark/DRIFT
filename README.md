<h1 align="center">DRIFT</h1>

<p align="center"><b>Decentralized Routed Inference For Tokens — one model, split across your machines, no datacenter.</b></p>

<p align="center">
  <b>English</b> ·
  <a href="./README.ko.md">한국어</a> ·
  <a href="./README.zh.md">中文</a> ·
  <a href="./README.ja.md">日本語</a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/TaewoooPark/DRIFT?style=flat-square&logo=github&logoColor=white&labelColor=000000&color=333333" alt="GitHub stars">
  <img src="https://img.shields.io/github/v/release/TaewoooPark/DRIFT?style=flat-square&labelColor=000000&color=333333" alt="Release">
  <img src="https://img.shields.io/github/last-commit/TaewoooPark/DRIFT?style=flat-square&labelColor=000000&color=333333" alt="Last commit">
  <img src="https://img.shields.io/badge/License-MIT-000000?style=flat-square&labelColor=000000&color=333333" alt="License MIT">
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.12-000000?style=flat-square&logo=python&logoColor=white&labelColor=000000" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.12-000000?style=flat-square&logo=pytorch&logoColor=white&labelColor=000000" alt="PyTorch">
  <img src="https://img.shields.io/badge/Apple%20MPS-000000?style=flat-square&logo=apple&logoColor=white&labelColor=000000" alt="Apple MPS">
  <img src="https://img.shields.io/badge/CUDA-000000?style=flat-square&logo=nvidia&logoColor=white&labelColor=000000" alt="CUDA">
  &nbsp;
  <img src="https://img.shields.io/badge/Peer--to--peer%20chain-000000?style=flat-square&labelColor=000000" alt="Peer to peer chain">
  <img src="https://img.shields.io/badge/Encrypted%20wire-000000?style=flat-square&labelColor=000000" alt="Encrypted wire">
  <img src="https://img.shields.io/badge/Signed%20receipts-000000?style=flat-square&labelColor=000000" alt="Signed receipts">
  <img src="https://img.shields.io/badge/Bitwise%20failover-000000?style=flat-square&labelColor=000000" alt="Bitwise failover">
  <img src="https://img.shields.io/badge/Bitwise%20parity-000000?style=flat-square&labelColor=000000" alt="Bitwise parity">
  <img src="https://img.shields.io/badge/MPS%20%E2%86%94%20CUDA-000000?style=flat-square&labelColor=000000" alt="MPS to CUDA">
</p>

<p align="center">
  <img src="docs/img/hero.png" alt="One model, split across the Earth — New York and Seoul running one model together, no datacenter" width="900">
</p>

<p align="center"><sub>A friend in New York leaves a node running while they sleep; you're in Seoul. DRIFT splits <b>one</b> model across both machines — their GPU computes the front layers, yours the back, the hidden state streams <b>node-to-node</b> over an <b>encrypted</b> wire, and every hop <b>signs a receipt</b> — so together you run a model neither could hold alone, provably the same answer as one machine.</sub></p>

**DRIFT** runs **one** large language model across **heterogeneous personal machines** — a Mac (Apple GPU, PyTorch **MPS**) and a Windows/Linux PC (NVIDIA GPU, PyTorch **CUDA**) — by splitting the model **layer by layer** (pipeline parallelism) and streaming only the **hidden state** between nodes over a **framework-neutral byte protocol** (TCP + msgpack). No datacenter, no `torch.distributed`, no NCCL, no vendor lock. The data plane is bound to *no* framework, so runtimes that could never talk to each other — an Apple Metal graph and an NVIDIA CUDA graph — now run one model together, and the output is **bit-for-bit identical** to running the whole model on a single machine.

On top of that exact core, DRIFT has grown a real **decentralization layer**: the hidden state now streams **peer-to-peer** (the head is no longer a bandwidth hub), the wire is **encrypted and membership-authenticated**, a dropped node is recovered **bitwise**, the head can be **weightless**, every hop **signs a receipt** the head verifies on live traffic, nodes **gossip-discover** each other, and their contribution is tallied in a **ledger**.

**The differentiator in one line:** [Exo](https://github.com/exo-explore/exo) binds node-to-node communication to MLX (`mx.distributed`), so it is *Apple-silicon-to-Apple-silicon only*. DRIFT lifts the boundary into a **neutral, encrypted wire protocol** — *different runtimes, different GPU vendors, one model* — proves the split is exact with a **bitwise parity gate**, and makes it **self-verifying** with signed per-hop receipts. A data plane bound to no framework, provably exact, and checkable without trusting the nodes — that is the core contribution.

**Scale.** One node per decoder layer — split one model across up to **28** machines on the default Qwen (**35** on Gemma), streaming across all of them. Two to four is today's sweet spot.

> *"The transcript is the model's output. The interesting part is **where** the computation actually ran — that it added up bit for bit, that the wire was encrypted, and that every hop signed for its work."*

[**taewoopark.com** — author site](https://taewoopark.com)

---

## Table of contents

- [Why this is different](#why-this-is-different) — the comparison table engineers came for
- [What is DRIFT](#what-is-drift) — the name, the vision, the scope
- [The five planes](#the-five-planes) — control / data / KV / security / trust
- [The wire contract](#the-wire-contract-what-actually-crosses-the-boundary) — schema + bytes-per-token
- [Three correctness problems](#three-problems-a-correct-split-must-solve) — KV re-index, RoPE, mask
- [Peer-to-peer, weightless head](#peer-to-peer-and-a-weightless-head) — the chain + thin head
- [Trust without trusting the nodes](#trust-without-trusting-the-nodes) — encryption, signed receipts, failover
- [Correctness & parity](#correctness--the-parity-gate) — the bitwise gate + measured results
- [Benchmarks](#benchmarks) — fidelity 100% · ½ the wire on int8 · O(1) head bandwidth
- [Model-agnostic by introspection](#model-agnostic-by-introspection) — Qwen, Gemma, no hardcoding
- [Design rationale (why-not)](#design-rationale-why-not) — the decisions and their reasons
- [Milestones](#milestones) · [Quickstart](#quickstart) · [Repo map](#repository-map--where-to-look) · [FAQ](#faq) · [What's still the vision](#whats-shipped-vs-still-the-vision)

---

## Why this is different

The whole point of DRIFT lives at the **boundary between nodes.** Here is how that boundary compares to the prior art:

| | **DRIFT** | Exo | Petals | llama.cpp RPC | vLLM / Megatron PP |
|---|---|---|---|---|---|
| **Split unit** | decoder layers | layers | transformer blocks | layers / tensors | layers (stages) |
| **Node↔node transport** | **TCP + msgpack** | MLX `mx.distributed` | gRPC (torch tensors) | custom RPC (ggml) | `torch.distributed` + NCCL |
| **Framework-neutral wire** | **✅ yes** | ❌ MLX-bound | ❌ torch-bound | ggml-bound | ❌ torch/NCCL-bound |
| **Heterogeneous GPU vendors** | **✅ MPS + CUDA at once** | ❌ Apple only | partial | ✅ (ggml backends) | ❌ NCCL can't bridge |
| **Data plane topology** | **✅ peer-to-peer chain** | activations | activations | activations | activations |
| **Wire encryption + node auth** | **✅ X25519 + ChaCha20 + PSK** | ❌ | ❌ | ❌ | ❌ |
| **Self-verifying (per-hop signed)** | **✅ Ed25519 receipts, live** | ❌ | ❌ | ❌ | ❌ |
| **Bitwise-exact failover** | **✅ re-split + replay** | ❌ | ~ (re-route) | ❌ | ❌ |
| **What crosses per token** | **~1.5–3 KB (hidden only)** | activations | activations | activations | activations |
| **Correctness contract** | **bitwise parity vs 1-machine** | — | — | — | — |

Read the table top-to-bottom and the thesis falls out: **everyone passes activations; only DRIFT makes the passing framework-neutral, encrypted, peer-to-peer, *and* provably bitwise-exact — then lets you check a node isn't lying without re-running the model.** NCCL cannot put an Apple GPU and an NVIDIA GPU in one process group. MLX cannot leave the Apple ecosystem. DRIFT's answer is a wire that carries *nothing but bytes* — no torch object, no MLX array, no CUDA handle — so the two worlds meet at a contract they can both implement, and then hardens that contract.

---

## What is DRIFT

A server-less, peer-to-peer inference network: heterogeneous personal devices split **one** model by layer and run it **together.** Instead of routing through a hyperscaler's datacenter, *your machine and someone else's* converge to run a single AI.

The name is the system:

| letter | meaning |
|---|---|
| **D** — Decentralized | no datacenter; the hidden state streams **peer-to-peer** node→node, the wire is encrypted + membership-authenticated, and a dropped node is recovered. An orchestrator still starts the run and the head can be made weightless — full leaderless consensus is still the vision (see [what's still the vision](#whats-shipped-vs-still-the-vision)). |
| **R** — Routed | an orchestrator *routes* hidden state through the nodes to carry inference forward |
| **I** — Inference | the workload is LLM inference (extensible to training) |
| **For T** — For Tokens | the double meaning of "token": the **inference** token (the atom of machine thought) **and** the **value** token (earned by contributing, spent on inference). Every hop now signs a receipt and `drift ledger` tallies contribution — the input a payout layer consumes. DRIFT's vision is to make the unit of thought and the unit of value one. |

> **Scope of this repository.** The hard technical core — *does a model split across a Mac and a Windows box produce the right answer?* — ships and is proven **bitwise**. On top of it, the **"For Tokens"** substrate is no longer only a diagram: a **peer-to-peer encrypted data plane**, **bitwise failover**, a **weightless head**, **signed-receipt verification on live traffic**, **gossip membership**, and a **contribution ledger** are all implemented and gated. A full token economy, on-chain settlement, and leaderless consensus remain the vision.

---

## The five planes

<p align="center"><img src="docs/img/arch.png" alt="DRIFT architecture — orchestrator head, per-layer shards, neutral wire" width="900"></p>

DRIFT separates cleanly into planes:

- **Control plane** — an orchestrator assigns each node a layer range (`configure`) and drives the decode loop. Nodes are found four ways: zero-config LAN discovery (mDNS), an explicit `--nodes host:port` list, a public `bore.pub` tunnel a NAT'd node opens with `drift node --tunnel`, or **gossip** — a node `--join`s one seed and the network learns its own membership, which `drift run --expand` then splits across.
- **Data plane** — the only things that cross a stage boundary are `hidden_states` (floats) + `position_ids` + `input_ids` (ints). Framework-agnostic, and — crucially — its size depends on `hidden_size`, not on the parameter count. **It now flows peer-to-peer** (`--chain`): head → n0 → n1 → … → tail → head, so tensor crossings/token drop from 2N to **N+1** and the head's bandwidth becomes **O(1)** in the node count instead of O(N). Optionally **int8** (`--int8`) halves the bytes.
- **KV cache plane** — each shard keeps the KV for *its own* layer range, per session, on its own device. The cache never crosses the wire (that would be megabytes/token and defeat the design). Only the residual stream travels.
- **Security plane** — a network shares one pre-shared key (`drift keygen`). Every connection then runs an X25519 ECDH → HKDF(mix PSK) → ChaCha20-Poly1305 channel, so the stream is confidential and a dialer without the key is dropped. `drift node --tunnel` refuses to run unkeyed (no open public compute), and the length prefix is capped (no alloc-DoS).
- **Trust plane** — every hop signs an **Ed25519 receipt** over `(in_hash, out_hash, layer range)`. The head verifies signatures + adjacency + end-anchors on **every token** of real traffic (not a separate challenge), so wire corruption, dropped/forged hops, and a node lying about what it computed vs. sent are caught live. A dropped node is recovered **bitwise** by re-splitting over the survivors and replaying.

**The split scales past two** — one node per decoder layer, up to 28 (35 on Gemma), with the head and the wire unchanged:

<p align="center"><img src="docs/img/scale.png" alt="DRIFT scales one model across 2 to 28 nodes, one decoder layer per node" width="900"></p>

---

## The wire contract (what actually crosses the boundary)

The contract (`drift/protocol.py`) is **frozen**: every message is a **4-byte big-endian length prefix + a msgpack dict** (encrypted as one ChaCha20-Poly1305 frame when a network key is set). Any future runtime — MLX, ggml, JAX, a Rust node — only has to implement this framing to join. There is no PyTorch on the wire.

```jsonc
// request  (orchestrator → shard, or shard → shard in chain mode)
{
  "type":         "prefill" | "decode" | "reset" | "ping" | "configure",
  "session_id":   "s0",               // one generation sequence
  "seq_id":       42,                 // monotonic, for ordering / debug
  "shape":        [1, 1, 1536],       // hidden_states shape (decode: S=1)
  "dtype":        "float16" | "int8",  // int8 → half the wire (lossy)
  "scale":        "<per-group fp16>",  // int8 dequant scales (absent for fp16)
  "position_ids": [37],               // absolute positions  → RoPE, computed on-shard
  "input_ids":    [785],              // token ids → per-layer embeddings (PLE) / thin-head embed
  "tensor":       "<raw bytes>",       // row-major hidden_states
  "route":        [["10.0.0.2", 52601]], // chain mode: the downstream nodes
  "collect":      ["10.0.0.9", 6000]     // chain mode: the head's sink
}

// response (shard → next hop / head)
{ "ok": true, "shape": [1,1,1536], "dtype": "float16", "tensor": "<bytes>",
  "receipt": { "node": "<pubkey>", "in_hash", "out_hash", "start", "end", "sig" },
  "token":  785 }   // thin-head tail returns a token id instead of a tensor
```

`route` / `collect` are **additive and optional** — a node without them behaves exactly like the classic star. `configure` assigns a layer range (and thin-head edge duties) to a **fungible** node, so users never hand-write ranges.

**Bytes per token.** During decode the activation is `[1, 1, hidden]`. For Qwen's `hidden = 1536` that is **3 072 bytes** in fp16, or **1 560 bytes** in int8 (H int8 + per-group fp16 scales ≈ 0.51×). A chain does `N+1` such crossings per token; a star does `2N`. On a LAN, trivial next to the compute.

**Why fp16 on the wire is safe (bitwise).** Serialization is a CPU fp16 round-trip. If the compute dtype is fp16, that round-trip is **bit-lossless** — the premise that lets the split path reproduce a single machine *exactly*, not approximately. int8 is *not* lossless and is opt-in; it runs under a relaxed gate, never the bitwise one.

---

## Three problems a correct split must solve

Splitting layers across processes sounds trivial until you try to make the output *identical* to the unsplit model. Three things bite, and DRIFT handles each explicitly.

### 1 · KV cache indexing — the subtle one

Hugging Face's `DynamicCache` reports "past length" from **layer 0's** slot. A shard that keeps global layers `[14, 28)` and reuses their global indices leaves cache slot 0 **empty** — so during decode the causal mask is built as if there were *no past*, and parity silently breaks after the very first token.

<p align="center"><img src="docs/img/kv-reindex.png" alt="KV-cache local re-indexing — the fix that keeps decode parity" width="900"></p>

DRIFT re-indexes each shard's kept layers to **local, 0-based** cache slots at load time, and sizes the per-session `DynamicCache` to the shard's local layer count.

### 2 · RoPE self-computation — keep the wire tiny

Rotary position embeddings depend only on `position_ids`. So each shard computes its own `cos/sin` from **absolute** positions via the model's own `rotary_emb`. The boundary carries a handful of integers instead of a full `cos/sin` tensor, and every node stays self-sufficient.

### 3 · Attention mask per stage

For prefill the mask is causal-full; for decode it is KV-length-aware. DRIFT rebuilds the mask per shard with the installed Transformers masking utilities, chosen **per layer** by the layer's own attention type (Gemma alternates local/global) — nothing hardcoded.

---

## Peer-to-peer, and a weightless head

**Chain streaming (`--chain`).** Instead of star-routing every hop back through the head, the hidden state flows node→node along the route and the tail delivers the final state to the head's collect sink. Two wins: tensor crossings/token drop from **2N to N+1**, and — the point — the head's data-plane bandwidth becomes **O(1)** in the node count, not O(N). The head stops being the hub every activation passes through.

**Thin head (`--thin`).** The head can hold **zero model weights**: `embed_tokens` moves to the first node's duty, `norm` + `lm_head` + `argmax` to the last node's. Combined with the chain, the head sends **one integer token id** into the pipeline and gets **one integer token id** back — it does no tensor math and materializes no parameters. Parity holds because `norm`+`lm_head`+`argmax` runs on the same device with the same (tied) weights over the bitwise-identical hidden state — the argmax is invariant to whether the head or the tail computes it.

The decode loop is written **once** over an injectable transport; only the transport (in-process / star / chain) is swapped, so the network is the only variable between milestones and any regression is *provably* a transport bug, never a logic bug.

<p align="center"><img src="docs/img/decode-loop.png" alt="The decode loop over an injectable transport (in-process / TCP / chain)" width="900"></p>

---

## Trust without trusting the nodes

**Encrypted, authenticated wire (`drift keygen`).** A network shares one 32-byte pre-shared key. Keyed connections do an X25519 ECDH (ephemeral → forward secrecy) → HKDF-SHA256 with the PSK mixed in → ChaCha20-Poly1305 with a per-direction counter nonce. Mixing the PSK into the KDF is the membership check: a peer without it derives a different key and its first frame fails to decrypt. Unkeyed stays plaintext for local dev; keying is network-wide.

**Signed receipts on live traffic.** Every hop signs an Ed25519 receipt over `(session, seq, mode, layer range, in_hash, out_hash)`. On **each token** the head checks signatures, adjacency (hop *i*'s `out_hash` == hop *i+1*'s `in_hash`), and end-anchors (the first hop's input matches what the head sent, the last hop's output matches what it received). A tampering node is caught on ordinary generation — no separate challenge to be honest on — and marked SUSPECT in a local reputation table. *What this catches:* wire corruption, dropped/reordered/forged hops, a node lying about what it computed vs. sent. *What it doesn't* (a node that consistently miscomputes and signs the result) is the job of the recompute audit (`drift verify`) or redundant N-of-M execution (future).

<p align="center"><img src="docs/img/parity-gate.png" alt="The parity gate — strict bitwise on one device, relaxed across GPU vendors" width="900"></p>

**Bitwise failover.** A node dying mid-generation no longer kills the session. The orchestrator re-splits the model across the survivors (plus any spare), re-prefills the sequence-so-far to rebuild every node's KV, and resumes. Because greedy decoding is deterministic over a fixed prefix, the resumed continuation is **bitwise-identical** to never having dropped — verified by killing a node mid-decode and checking the finished sequence against an uninterrupted reference.

**Contribution ledger.** The head journals every verified receipt; `drift ledger` folds it into a per-node tally (tokens carried, layer-tokens served, sessions) with `--verify` re-checking every signature and `--csv` export. That is the settlement layer's input substrate.

---

## Correctness — the parity gate

DRIFT is **correctness-first**: every networked step must reproduce the single-machine reference **bitwise** before any performance or decentralization work. Speed is not the point — *heterogeneous split inference being exact* is, and every feature above is gated against that.

**Measured results** — Qwen2.5-1.5B-Instruct, Apple MPS, fp16:

| Gate | What it isolates | Result |
|---|---|---|
| **in-process** 2-shard | sharding · RoPE · KV · mask | ✅ **6 / 6 prompts bitwise** (`n = 1…180`) |
| **TCP star** 2-process | serialization / framing | ✅ **bitwise == reference** |
| **chain** 2 & 3 nodes | peer-to-peer relay | ✅ **bitwise == reference** |
| **chain + encrypted** | AEAD channel transparency | ✅ **bitwise** (encryption doesn't perturb tokens) |
| **thin head** 2 & 3 nodes | weightless head, edge embed/lm_head | ✅ **bitwise** |
| **kill mid-decode** (chain / star, entry / middle / tail) | failover replay | ✅ **48 / 48 bitwise**, recovery triggered |
| **tamper a node** | live receipt verification | ✅ **caught on live traffic**, honest run 0 suspects |
| **MPS ↔ CUDA (M4)** | cross-vendor fp16 rounding | ✅ **130 / 130 tokens** match across 3 prompts |

**MPS ↔ CUDA (M4).** Running the front half on a Mac (Apple MPS) and the back half on a Colab NVIDIA T4 (CUDA), the split reproduced the single machine **exactly (130/130 tokens)** — even though the two vendors' fp16 kernels widened the first-step logit gap to ~2×10⁻² (vs ~8×10⁻³ same-device), not enough to flip an argmax here. At larger scale that gap can flip a late token; the **relaxed gate** `python -m drift.parity_test --prefix-match K` is there for that.

<p align="center"><img src="docs/img/m4-result.png" alt="M4 measured — Mac Apple MPS + Colab NVIDIA T4 CUDA, 130/130 token match vs one machine" width="900"></p>

---

## Benchmarks

*Reproduce the single-machine numbers with `python -m drift.bench`; the integration gates with `python -m drift.itest …`.*

**Fidelity — does splitting change the output?** *(split path vs the single-machine oracle, greedy)*

| Metric | Result |
|---|---|
| token exact-match — 6 prompts, `n = 1…180` | **411 / 411 = 100.00 %** |
| first-step logit max-abs-diff (fp32) | 7.81 × 10⁻³ *(fp16 ULP)* |
| KL divergence (nats) | ≤ 2.82 × 10⁻¹⁰ |

**Footprint — no single node holds the whole model** (heaviest node = **42 %** of the weights, measured on-device, not just compute share). Each node materializes only its slice (`init_empty_weights` + a selective safetensors read), so the whole model is never resident on any one machine.

**The wire is thin, peer-to-peer, and optionally half-size**

| Metric | Value |
|---|---|
| on the wire per token per hop (fp16) | **3.10 KB** — only the hidden state |
| on the wire per token per hop (**int8**) | **1.52 KB** — 51 % of fp16 (measured fidelity ~67 %, relaxed) |
| tensor crossings/token — star → **chain** | 2N → **N+1** |
| head data-plane bandwidth — star → **chain** | O(N) → **O(1)** |
| protocol overhead (localhost, fp16 star) | ~1.2 ms/hop, dwarfed by ~41 ms/token compute |

> Absolute, reproducible numbers — not a cherry-picked win. On an Apple-only cluster Exo's native MLX path wins raw throughput; DRIFT's axis is *heterogeneous, exact, and verifiable* — where no competitor even runs.

---

## Model-agnostic by introspection

The engine never hardcodes a model architecture. At load it **introspects** the loaded model and adapts — the loaded model is the source of truth, not a fixed class. Two very different families drop into the *same* engine:

| Model | Layers → split | Quirks DRIFT handles (introspected, never hardcoded) |
|---|---|---|
| **Qwen/Qwen2.5-1.5B-Instruct** *(primary)* | 28 → `0–14 / 14–28` | plain decoder, single RoPE θ, `DynamicCache`, tied `lm_head` — the correctness baseline |
| **google/gemma-4-E2B-it** *(secondary)* | 35 → `0–18 / 18–35` | **Per-Layer Embeddings** · sqrt(hidden) embed scaling · **dual RoPE θ** · **hybrid** sliding/global attention · `HybridCache` + KV-sharing groups; needs `transformers ≥ 5.5` |

Every quirk maps cleanly onto a plane and is discovered from `config`/signature at load, so the code that runs Qwen runs Gemma unchanged: *depend on what you can observe, hardcode nothing.*

**And beyond these two:** the table lists the *gated* families — the ones the parity suite has proven bitwise. The same introspection is designed to carry **any decoder-only Hugging Face causal LM** within DRIFT's constraints (an architecture the installed `transformers` supports; fp16 weights that fit across the nodes' combined memory). Point `drift run --model <hf-id>` (or `model_id` in `config.yaml`) at one — the split, the wire size, and the layer plan re-derive themselves — then hold it to the same standard with `python -m drift.parity_test`. Details in the [operations manual §6](docs/manual.md).

---

## Design rationale (why-not)

- **Why not `torch.distributed` / NCCL across nodes?** NCCL cannot place an Apple Metal device and an NVIDIA CUDA device in one process group — full stop. And it couples the data plane to a backend, which is exactly what DRIFT refuses.
- **Why peer-to-peer chain, not a star?** The star makes the head an O(N) bandwidth hub — a single point every activation passes through. The chain drops it to O(1) and is the prerequisite for a de-privileged head.
- **Why sign a receipt on every hop instead of a spot-check?** A fixed challenge is escaped by a node honest only on the challenge. Binding verification to the real traffic means there is nothing to be selectively honest on.
- **Why re-prefill on failover instead of replicating KV?** Replication is bandwidth the design refuses; re-prefill is O(sequence) once and — because greedy is deterministic — bitwise-exact. Correct and cheap beats seamless and heavy for a v1.
- **Why group-wise int8, not per-tensor?** The residual stream has outlier channels; one scale per tensor crushes everything else (measured: 0% match). A scale per 128-dim block keeps fidelity usable while still ~halving the wire.
- **Why freeze the wire at M0?** So node internals change forever without a flag day. `route` / `collect` / `scale` were all added as *optional* fields — never a breaking change.

---

## Milestones

| # | Milestone | Status |
|---|---|---|
| **M0–M3** | env · reference oracle · in-process + TCP 2-shard parity | ✅ **bitwise** |
| **M4** | cross-machine — Mac MPS + NVIDIA CUDA | ✅ **measured** — 130/130 tokens |
| **M6** | graceful kill-node detection | ✅ clean `NodeUnavailable` |
| **M7** | peer-to-peer chain data plane | ✅ **bitwise** · 2N→N+1 crossings, O(1) head |
| **M8** | encrypted + authenticated wire (PSK + X25519 + ChaCha20) | ✅ **bitwise** · tamper-tunnel closed |
| **M9** | bitwise failover — re-split + replay | ✅ **48/48 bitwise** after a mid-run kill |
| **M10** | thin head — zero-weight orchestrator | ✅ **bitwise** |
| **M11** | signed-receipt verification on live traffic | ✅ tamper **caught live**, honest run clean |
| **M12** | gossip membership + dynamic join | ✅ seed learns all, head expands + splits |
| **M13** | contribution ledger (`drift ledger`) | ✅ tally reconciles, forged line rejected |
| **M14** | WAN performance — group-wise int8 wire | ✅ **½ the wire**, ~67% measured fidelity |
| **M15** | docs overhaul — this README | ✅ |

Everything above is exercised by `drift itest` (spawns real local nodes and gates the split against the in-process reference). Speculative decoding, leaderless consensus, and a token economy are the vision, not shipped — see below.

---

## Quickstart

Requires Python **3.12** and [`uv`](https://github.com/astral-sh/uv). Both default models are **ungated** — no Hugging Face login.

**1 · Install** — on each machine:

```bash
git clone https://github.com/TaewoooPark/DRIFT && cd DRIFT
bash scripts/install.sh          # macOS / Linux   ·   Windows: powershell -File scripts\install.ps1
drift doctor                     # checks Python, torch, device, config, ports
```

**2 · Try it on one machine:**

```bash
drift up 2                       # 2 local nodes, auto-split, open a chat
drift up 3 --chain               # peer-to-peer: nodes stream to each other
drift up 2 --thin                # weightless head (embed + lm_head on the nodes)
drift up 2 --int8                # half-size wire (lossy, opt-in)
```

**3 · Run one model across your Mac + a CUDA PC** — the real thing:

```bash
# Windows/Linux PC (NVIDIA)     — one terminal
drift node --port 52601        # device = cuda, announced on the LAN

# Mac (Apple)                  — terminal 1: a worker
drift node --port 52600        # device = mps

# Mac                          — terminal 2: the head (type the prompt)
drift run --chain --prompt "hello world"
```

**Encrypt the wire** (share one key across machines):

```bash
drift keygen                     # prints DRIFT_NETWORK_KEY=<hex>
export DRIFT_NETWORK_KEY=<hex>   # on every machine — the wire is now encrypted + authenticated
```

**Join from anywhere** — a NAT'd node opens a tunnel and gossips into the network:

```bash
drift node --tunnel --join bore.pub:PORT      # needs a network key (no open compute)
drift run --expand --nodes bore.pub:PORT      # discover the whole membership, split across it
```

**Who computed what:**

```bash
export DRIFT_JOURNAL=~/drift.jsonl && drift run --chain --prompt "…"
drift ledger ~/drift.jsonl --verify           # per-node contribution, signatures re-checked
```

**Serve it like an OpenAI-compatible local backend** — the model still runs across
DRIFT nodes; only the client-facing API is HTTP/SSE:

```bash
drift serve --nodes 127.0.0.1:52600,127.0.0.1:52601 --api-key local-dev
```

Supported text-generation surfaces include `/v1/models`, `/v1/chat/completions`,
`/v1/completions`, `/v1/responses`, `/v1/embeddings` where the mode can expose
hidden states, tokenizer helpers, health/readiness, and metrics. Multiple choices
(`n`) and OpenAI-shaped logprobs are accepted; logits-backed DRIFT runs return
exact selected-token/top-k logprobs. Tool-call and JSON response-format
compatibility are exposed as API-shape layers, and Responses streaming emits
semantic SSE events. DRIFT does not execute tools or guarantee strict
schema-constrained decoding. Multimodal/audio and thin-mode sampling/embeddings
return explicit OpenAI-shaped unsupported errors. See
[docs/openai-compatibility.md](docs/openai-compatibility.md).

**Customize & fine-tune** — models, split points, devices, troubleshooting — is all in the **operations manual → [docs/manual.md](docs/manual.md)** ([한국어](docs/manual.ko.md) · [中文](docs/manual.zh.md) · [日本語](docs/manual.ja.md)).

**See it live** — [**DRIFT-Demo**](https://github.com/TaewoooPark/DRIFT-Demo): a two-screen visual demo of a real run — the residual stream crossing the wire, per-layer ‖Δh‖, the tail's own top-k, signed receipts, and the contribution tally — every pixel drawn from live traffic, the DRIFT sources untouched.

---

## Repository map — where to look

```text
drift/
  protocol.py       # THE CONTRACT — 4B length prefix + msgpack; fp16/int8 tensor ser/deser
  crypto.py         # network key + node identity; X25519+ChaCha20 channel; keygen
  engine_torch.py   # PyTorch shard: introspected layer calls, local KV re-index, self-RoPE  ← the crux
  loader.py         # sliced weights — init_empty_weights + only the shards a node runs
  shard_server.py   # concurrent TCP server: ping / configure / prefill / decode / relay / gossip
  orchestrator.py   # head + injectable transport (in-process / star / chain) + decode loop + verifier
  run.py, node.py   # `drift run` head + `drift node` worker (auto-split, discovery, tunnel, --join)
  openai_api.py     # `drift serve`: OpenAI-compatible HTTP/SSE adapter over the orchestrator
  receipts.py       # signed per-hop receipts + live verifier + journal (the ledger source)
  membership.py     # gossip peer table — signed entries, anti-entropy, --expand
  ledger.py         # `drift ledger` — per-node contribution from the receipt journal
  verify.py         # trustless spot-check (recompute audit — complements the live receipts)
  parity_test.py    # in-process / TCP bitwise gate + multi-prompt --selftest
  itest.py          # integration gate over REAL nodes: chain / secure / thin / kill / tamper / expand / ledger / int8
  bench.py, bench_m4.py   # single-machine + cross-machine (M4) benchmarks
config.yaml         # model, dtype, port, shard table
```

**Reviewer's shortlist:** `engine_torch.py` (KV re-index + introspection), `protocol.py` (the frozen wire), `orchestrator.py` (injectable transport + chain + verifier), `receipts.py` (the trust layer).

---

## FAQ

**Is this just pipeline parallelism?** The *idea* is, but the contribution is the **boundary**: PP in vLLM/Megatron is welded to `torch.distributed`+NCCL and can't bridge MPS↔CUDA. DRIFT's boundary is neutral, encrypted bytes flowing peer-to-peer — proven bitwise-exact and self-verifying.

**Does the network see my tokens?** Be clear-eyed: `input_ids` are integer token ids, but that is a *reversible* encoding — anyone with the (public) tokenizer turns them back into your text, and a downstream shard needs them. So **a node operator can read your prompt** unless you encrypt the wire. `drift keygen` + `DRIFT_NETWORK_KEY` makes the stream confidential to nodes that share the key; without it, DRIFT is plaintext (fine for a LAN you own). And you can check a node isn't lying about its compute — every hop signs a receipt the head verifies live, and `drift verify` recompute-audits a node you don't own.

**What happens if a node dies mid-generation?** The session survives: DRIFT re-splits over the survivors (plus any spare), replays the sequence-so-far, and continues — bitwise-identical to never having dropped. No seamless (zero-replay) failover yet; that needs replication.

**Can I add a third node?** Yes — `drift up 3`, or `drift run --expand` to gossip-discover every member and split across all of them. The wire contract doesn't change.

**Why is the reference on MPS, not CPU?** The compute dtype is fp16 and CPU fp16 kernels in PyTorch are unreliable; MPS runs fp16 correctly and deterministically, so the parity baseline is on MPS. CPU/CUDA are configurable.

---

## What's shipped vs. still the vision

The hard core — a correct, **bitwise-verified** heterogeneous split — is done. The decentralization layer on top is **implemented and gated**, not a diagram:

| capability | shipped | milestone |
|---|---|---|
| Run a model too big for one machine (per-shard load) | ✅ | v0.10–0.16 |
| Peer-to-peer data plane (no head hub) | ✅ | M7 |
| Encrypted + authenticated wire | ✅ | M8 |
| Bitwise failover on a dropped node | ✅ | M9 |
| Weightless head | ✅ | M10 |
| Self-verifying — signed receipts on live traffic | ✅ | M11 |
| Gossip membership + join-from-anywhere | ✅ | M12 |
| Contribution ledger | ✅ | M13 |
| Half-size int8 wire | ✅ | M14 |

**Still the vision** (honestly): leaderless **consensus** (an orchestrator still starts each run), **Sybil resistance** (gossip entries are self-asserted; no admission control), a **token economy** with pricing / payout / on-chain settlement (the ledger is the input, not the settlement), **seamless failover** (replication, so no replay), **speculative decoding** (needs per-shard KV rollback), and **N-of-M redundant execution** (to catch a consistently-miscomputing node, which live receipts alone can't). These are the roadmap — the difference between *"a P2P, encrypted, self-verifying, fault-tolerant heterogeneous inference network, every step provably identical to one machine"* (true today) and *"a finished decentralized token economy"* (not yet).

---

## Contact

<p align="center">
  <a href="https://github.com/TaewoooPark"><img src="https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white&cacheSeconds=3600" alt="GitHub"></a>
  <a href="https://x.com/theoverstrcture"><img src="https://img.shields.io/badge/-X-000000?style=for-the-badge&logo=x&logoColor=white&cacheSeconds=3600" alt="X (Twitter)"></a>
  <a href="https://www.linkedin.com/in/taewoo-park-427a05352"><img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&cacheSeconds=3600" alt="LinkedIn"></a>
  <a href="https://www.instagram.com/t.wo0_x/"><img src="https://img.shields.io/badge/-Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white&cacheSeconds=3600" alt="Instagram"></a>
  <a href="https://taewoopark.com"><img src="https://img.shields.io/badge/-taewoopark.com-000000?style=for-the-badge&logo=safari&logoColor=white&cacheSeconds=3600" alt="Personal site"></a>
  <a href="mailto:ptw151125@kaist.ac.kr"><img src="https://img.shields.io/badge/-Email-D14836?style=for-the-badge&logo=gmail&logoColor=white&cacheSeconds=3600" alt="Email"></a>
</p>

<p align="center"><sub>No datacenter. No torch.distributed. Your machine and someone else's, running one mind — peer-to-peer, encrypted, and signed for, bit for bit.</sub></p>
