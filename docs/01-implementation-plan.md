# 01 — Phase-by-Phase Implementation Plan (M0–M6)

Operationalizes spec §9 milestones into concrete tasks. Canonical values (model, split,
port, versions) live in [`03`](03-goal-execution-plan.md); parity-debugging procedure in
[`05`](05-parity-debugging-playbook.md); environment setup in [`06`](06-m0-setup-runbook.md).

> **Hard constraints (spec §1) that bind every milestone:** no `torch.distributed`/NCCL/
> gloo/RPC across nodes; the §6 wire contract is immutable once set; correctness before
> performance; the node engine stays behind the `ShardEngine` interface (§7).

## Cross-cutting design decision — injectable transport

`orchestrator.py` routes hidden states through shards via an **injectable transport** with
one signature:

```python
# transport(shard, session_id, hidden, position_ids, mode) -> hidden
```

- **M2** injects an *in-process callable* that calls `engine.forward(...)` directly (no socket).
- **M3+** injects a *socket client* speaking the §6 protocol.

The decode loop is written **once** and never changes between M2 and M3 — so the network
becomes the *only* variable, and any M3 regression is provably a serialization/framing bug,
not a logic bug. This is the single most important structural choice in the build.

---

## M0 — Environment + neutral protocol framing

**Goal:** both shards answer `ping` over the neutral protocol.
**Executable on Mac now:** ✅ (localhost, two ports). Split into **M0a** (Mac, now) / **M0b** (Windows, defer to M4).

**Tasks**
1. Env per [`06`](06-m0-setup-runbook.md): `uv venv --python 3.12`, install torch (MPS)/transformers/safetensors/msgpack/numpy, `huggingface-cli login`, `pip freeze > requirements.lock`.
2. `config.yaml` (spec §5) — model id, dtype, port, shard table with `host/start_layer/end_layer/device`.
3. `protocol.py` — `send_msg`/`recv_msg`/`_recvn` exactly per spec §6 (4-byte big-endian length prefix + msgpack dict). **This is the wire contract — freeze it.** Decide the schema fields *now* (spec §1.2): `hidden_states` + `position_ids` **+ `input_ids`**. The extra `input_ids` (small ints) lets PLE models (Gemma 4) self-compute per-layer embeddings on the downstream shard without a re-freeze; Qwen simply ignores it. The boundary stays neutral and small.
4. `engine_base.py` — the `ShardEngine` ABC (spec §7): `load`, `forward`, `reset`.
5. `engine_torch.py` — `load()` only for now (full-model load, keep `layers[start:end]`, capture `embed_tokens`/`norm`/`lm_head`/`rotary_emb` refs); plus a `ping`-info method.
6. `shard_server.py` — TCP listen → `recv_msg` → handle `ping`/`reset` → `send_msg`. Sequential, single-session. Reads its identity from `config.yaml`, but accepts CLI overrides (`--name/--start/--end/--device`) and a `DRIFT_PORT` env var for localhost multi-port runs (see [`06`](06-m0-setup-runbook.md)).
7. `orchestrator.py` — `ping` client against the shard list; accepts a `--ports` override to target multiple localhost ports (config supplies the single shared port for the cross-host M4 case).

**Acceptance (§9 M0):** orchestrator receives a valid `ping` reply `{ok, name, start_layer, end_layer, device}` from both shards.

**Risks:** Python 3.14 has no torch wheel → pin 3.12 (`06`). Gemma 4 needs `transformers >= 5.5` (Qwen ≥ 4.44) → pin the newer and match both nodes. Default models are ungated, so no login needed. Localhost port reuse → run the two shards on `52600` and `52601`.

**Effort:** 0.5–1 day.

---

## M1 — Single-machine reference oracle

**Goal:** a deterministic ground truth for parity.
**Executable on Mac now:** ✅

**Tasks**
1. `reference.py` — load the full model normally; **greedy** (`do_sample=False`) generate **50 tokens** from a fixed prompt.
2. Save the **token-id sequence** + **first-step logits** to `reference_out.npz`.
3. `AutoConfig`-verify `num_hidden_layers` (Qwen 28 / Gemma 4 E2B 35) → this fixes the split point (`03` decision #2). For Gemma 4 also inspect the config for KV-sharing layer groups — don't split inside one.
4. Compare logits in **fp32** when diffing later.

**Acceptance (§9 M1):** deterministic output saved; re-running yields identical token ids.

**Risks:** greedy must be deterministic on a single device (it is). Wrong layer count silently breaks the split → the `AutoConfig` check is mandatory.

**Effort:** 0.5 day.

---

## M2 — In-process 2-shard parity (no network) — ⚠️ correctness core

**Goal:** prove sharding/RoPE/KV logic, isolated from the network.
**Executable on Mac now:** ✅ (highest-risk milestone)

Do M2 on **Qwen first** (plain decoder — isolates the split logic from model quirks), then repeat for Gemma 4 (callout below).

**Tasks**
1. Finish `engine_torch.forward()`:
   - Run `model.model.layers[start:end]` over the incoming hidden state.
   - Compute RoPE **locally** from `position_ids` via `model.model.rotary_emb` (layer-agnostic — a shard holding `[k,N)` still computes correct cos/sin). Pass `position_embeddings=(cos,sin)` to layers. **Introspect** the *loaded model's* decoder-layer `forward` signature (`type(model.model.layers[0])`, spec §7.2) — never hardcode arg lists (it's `Qwen2DecoderLayer` for Qwen, `Gemma4DecoderLayer` for Gemma 4).
   - Per-`session_id` cache, **type chosen by model**: `DynamicCache` (Qwen) vs `HybridCache` (Gemma — sliding-window layers). Use `cache_position` (not deprecated `_seen_tokens`). `prefill` fills KV; `decode` appends one position.
   - `attention_mask`: causal-full for prefill, KV-length-aware for decode (use HF utilities). For Gemma's hybrid attention, build **both** a full and a sliding-window mask and pass the right one per layer by its type.
2. Finalize `engine_base.py`.
3. `orchestrator.py` core decode loop (spec §8) with the **in-process transport**: `embed_tokens` → route through both shards → `final_norm` → `lm_head` → argmax. (Gemma: also apply embedding scaling — callout.)
4. `parity_test.py` — run the split path greedy 50 tokens, assert token-id sequence equals `reference_out.npz`.

**Acceptance (§9 M2, strict):** split token-id sequence is **exactly** equal to M1 (per model).

**Risks (spec §13 suspect list):** (a) RoPE/`position_ids` wiring, (b) KV position accumulation across prefill→decode, (c) attention-mask length, (d) `embed_tokens`/`norm`/`lm_head` applied inside a shard by mistake. Use [`05`](05-parity-debugging-playbook.md) continuously here.

> **Gemma 4 second-model bring-up (after Qwen parity).** Same engine, extra model-aware handling — all introspected, never hardcoded (see [`05`](05-parity-debugging-playbook.md) model-specific suspects):
> - **PLE (per-layer embeddings)** — each shard holds `embed_tokens_per_layer` for *its* layers and computes them from `input_ids` (carried on the wire) — like RoPE, self-computed locally so the boundary stays small. **ORCHESTRATOR/SHARD.**
> - **Embedding sqrt(hidden) scaling** at the embed step. **ORCHESTRATOR.**
> - **Dual RoPE theta** (≈10k sliding / 1M global) — shard uses the correct base per layer type. **SHARD.**
> - **Hybrid per-layer attention** (512 sliding vs global) + **HybridCache** + possible **KV-sharing layer groups**. **SHARD/MASK.**
> - Tied `lm_head ↔ embed_tokens` (orchestrator holds both — already so).

**Effort:** 1.5–2 days (Qwen) + ~1–1.5 days (Gemma 4 quirks).

---

## M3 — Localhost 2-process parity (TCP)

**Goal:** prove serialization/framing; the only new variable vs M2 is the socket.
**Executable on Mac now:** ✅

**Tasks**
1. `shard_server.py` — wire the `prefill`/`decode` path to `engine.forward`.
2. `orchestrator.py` — swap in the **socket transport** (same decode loop).
3. `protocol.py` — tensor ser/deser: `tensor.detach().to("cpu", torch.float16).contiguous().numpy().tobytes()`; restore via `np.frombuffer(...).reshape(shape).copy()` (`.copy()` because `frombuffer` is read-only).

**Acceptance (§9 M3, strict):** token-id sequence is **exactly** equal to M1. (fp16 CPU round-trip is bitwise-lossless, so parity *must* stay exact — any drift is a framing bug: length prefix, partial `recv`, missing `.copy()`, or shape/dtype mismatch.)

**Risks:** partial reads (`_recvn` must loop), endian/length-prefix errors, forgetting `.copy()`.

**Effort:** 1 day.

---

## M4 — Cross-machine (Mac MPS + Windows CUDA)

**Goal:** the differentiator — two GPU vendors, one model.
**Executable on Mac now:** ❌ needs Windows + LAN.

**Tasks**
1. M0b on Windows (CUDA torch; **identical** transformers via `requirements.win.lock` matching the Mac lock).
2. `config.yaml` real LAN IPs; same model files on both.
3. Run the same generation across machines.

**Acceptance (§9 M4, relaxed):** coherent output + early ~10 greedy tokens match the reference; late divergence tolerated.

**Risks:** MPS↔CUDA float differences cause *late* divergence (expected). **Early** divergence (token 1–2) is a bug → bisect with [`05`](05-parity-debugging-playbook.md). Version skew between nodes silently breaks parity → `drift-env-introspect` validates the locks (`04`).

**Effort:** 1 day + Windows setup.

---

## M5 — Booth display + interactive

**Goal:** the demo experience.
**Executable on Mac now:** 🟡 localhost dry-run; real run needs Windows.

**Tasks**
1. `display.py` — each shard shows `layer a–b · <host>(<device>)` + activity counter; orchestrator shows prompt box + streaming output + the route ("front half = Apple GPU, back half = NVIDIA").
2. `orchestrator.py` — token streaming to stdout/websocket as generated; interactive prompt input.
3. `shard_server.py` — activity hooks for the display.
4. Decision #5 (`03`): `rich` terminal vs local webpage.

**Acceptance (§9 M5):** an audience member enters a prompt and watches a response generated across both machines in real time, each node showing its layers.

**Risks:** `rich`-terminal vs local-webpage rendering differences (`03` decision #5); MPS op fallback during interactive runs; localhost dry-run hides real cross-machine streaming latency.

**Effort:** 1–1.5 days.

---

## M6 (optional) — Graceful kill-node recovery

**Goal:** resilience polish (not seamless failover — that needs replication, out of scope).
**Executable on Mac now:** ❌ needs two nodes.

**Tasks:** orchestrator detects a dropped shard mid-decode → notifies the user → graceful restart/reconfigure.

**Acceptance (§9 M6):** kill a shard during decode → detected → user notified → graceful restart.

**Risks:** half-open/dropped TCP sockets are slow to detect mid-decode; must cleanly reset per-session KV/`DynamicCache` + shard state on reconfigure; guard against double-generation after restart.

**Effort:** 1 day.

---

## File-creation summary (spec §4)

| File | First touched | Finalized |
|---|---|---|
| `config.yaml` | M0 | M4 (real IPs) |
| `protocol.py` | M0 (framing) | M3 (tensor ser/deser) |
| `engine_base.py` | M0 | M2 |
| `engine_torch.py` | M0 (`load`) | M2 (`forward`) |
| `shard_server.py` | M0 (ping/reset) | M3 (forward), M5 (hooks), M6 (shutdown) |
| `orchestrator.py` | M0 (ping) | M2 (loop), M3 (socket), M5 (stream), M6 (detect) |
| `reference.py` | M1 | M1 |
| `parity_test.py` | M2 | M3 |
| `display.py` | M5 | M5 |
| `requirements.lock` / `.win.lock` | M0 | M4 |

> **Out of M0–M6 scope (booth safety net):** spec §11's llama.cpp RPC fallback (deliverable `fallback_llamacpp.md`, the spec's "Plan 3") is an intentionally-separate demo-day backstop — each machine runs llama.cpp on its own backend (Mac=Metal, Windows=CUDA) over the RPC transport — **not** part of the M0–M6 main path. It is deferred for the same reason spec §12 (v2 MLX, `engine_mlx.py`) is flagged post-DoD in [`03`](03-goal-execution-plan.md) decision #4: prove the unified PyTorch cross-vendor path first.
