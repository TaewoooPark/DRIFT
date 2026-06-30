# 03 — Goal Execution Plan

Maps the spec's goals (§1) to milestones (§9), fixes the canonical values every other doc
cites, sets the dependency order and the now/later split, records the open decisions, and
states the Definition of Done (§14).

---

## Canonical values (defined here — cite, never redefine)

| Key | Value | Source / note |
|---|---|---|
| Primary model | `Qwen/Qwen2.5-1.5B-Instruct` | **ungated**; simple plain decoder — the first bring-up |
| Primary layers / split | 28 layers → `0–14` (Mac) / `14–28` (Windows) | verify via `AutoConfig` at M1 |
| Secondary model | `google/gemma-4-E2B-it` | **ungated** (Apache 2.0); advanced — second bring-up |
| Secondary layers / split | 35 layers → `0–18` / `18–35` | verify layer count + KV-sharing groups via config at M1 |
| Boundary dtype | `float16` | spec §6; CPU round-trip is lossless; **MPS-safe (avoid bf16 on MPS)** |
| TCP port | `52600` | spec §5; localhost uses two ports (see `06`) |
| Python | `3.12` (via `uv venv --python 3.12`) | **not 3.14** — no torch wheel |
| `transformers` | pinned in `requirements.lock`, **identical on both nodes** | Qwen needs ≥ 4.44; **Gemma 4 needs ≥ 5.5**; parity-critical |
| `torch` | per-machine (Mac MPS wheel / Windows CUDA wheel) | versions differ by device — that's expected |
| Wire schema fields | `hidden_states` + `position_ids` + **`input_ids`** | `input_ids` (ints, small) added so **PLE models (Gemma 4) work** — shards self-compute per-layer embeddings; decide at M0 before freezing (spec §1.2) |
| Code dir | `drift/` | spec §4 |
| Repo root | `/Users/taewoopark/personal/DRIFT` | — |

> **Both default models are ungated** → `huggingface-cli login` is **optional** (only for rate limits / private repos). Llama-3.2-1B (spec §5's placeholder example — §5 values are user-filled) and the gated `google/gemma-3-1b-it` are *not* used here; Gemma 3 is the simpler-but-gated fallback only if Gemma 4's PLE proves too fiddly (decision #1).

---

## Goal ↔ milestone matrix

| Goal (spec §1) | Proven by |
|---|---|
| Neutral data plane (no `torch.distributed`/NCCL/gloo) | **M0** (ping over msgpack/TCP), **M3** (lossless tensor transport) |
| Correctness-first / parity | **M1** (oracle), **M2** (sharding/RoPE/KV), **M3** (serialization) |
| Heterogeneous cooperation — *the differentiator* | **M4** (MPS + CUDA on one model) |
| Swappable engine behind an interface | **M2** (`ShardEngine` ABC in use); fully realized in v2 (§12) |
| Model-family-agnostic engine | running **both** Qwen (plain decoder) and Gemma 4 (PLE + dual-rope + hybrid attention) through the same introspection-based engine — proves the boundary isn't tied to one architecture |
| Demoable | **M5**; resilience polish **M6** |

---

## Dependency order

Strictly linear: **M0 → M1 → M2 → M3 → M4 → M5 (→ M6)**.

Two **hard stops** (do not advance past until met):
- **M2** — split token-id sequence must be **bitwise-equal** to the M1 reference.
- **M3** — same exact equality, now over TCP.

Rationale (spec §1.3): every networked step must reproduce the single-machine reference
before any optimization. M2 isolates sharding/RoPE/KV correctness from the network; M3
isolates serialization. Don't conflate them.

---

## Now vs later split (the strategic point)

The Mac alone covers ~80% of the engineering and **100% of the correctness risk**.

| Phase | Needs | Milestones |
|---|---|---|
| **NOW — Mac only** | this machine | M0a (Mac install + localhost ping), M1, M2, M3, M5 **dry-run** on localhost |
| **LATER — needs Windows** | the second node + LAN | M0b (Windows install + cross ping), M4, real-hardware M5, M6 |

**Rule:** do not book Windows/LAN time until **M3 passes**. The Mac-only track de-risks the
whole project first; M4 then only adds the heterogeneous-hardware variable.

---

## Decision register

| # | Decision | Resolution | Decide at |
|---|---|---|---|
| 1 | Model | **Qwen2.5-1.5B-Instruct first** (plain, ungated — proves the core), then **`google/gemma-4-E2B-it`** as the second-family bring-up (ungated, but PLE + dual-rope + hybrid attention). `gemma-3-1b-it` (gated, simpler) is the fallback only if Gemma 4's PLE is too fiddly | M1 / second pass |
| 2 | Split point | Match layer count (Qwen 14/14, Gemma 4 E2B 18/17); verify via `AutoConfig` + check Gemma's KV-sharing groups (don't split inside one); memory-weighting deferred | M1 |
| 3 | Loading path | v1 = full-model-load, keep slice (spec §7.1). Memory path (`init_empty_weights` + per-shard safetensors) deferred | M2 |
| 4 | Mac engine | v1 = PyTorch-MPS. MLX (`engine_mlx.py`) is post-DoD (§12) | v2 |
| 5 | Display | `rich` terminal first; local webpage only if a big screen demands it (this also decides whether `webapp-testing`/`playwright` are needed — see `04`) | M5 |
| 6 | Doc language | English (this suite); spec stays the Korean source of truth | done |

---

## Definition of Done (spec §14) — checklist

DRIFT v1 demo is done when **M0–M5 pass**, i.e.:

- [ ] Audience enters a prompt → response **streams** token-by-token.
- [ ] Generation crosses **both** heterogeneous machines — Mac (Apple GPU, MPS) and Windows (NVIDIA, CUDA) — on one model.
- [ ] Each node **displays** its layer range + device (e.g. `layer 0–7 · MacBook(MPS)`).
- [ ] The data plane is **DRIFT's neutral protocol** (§6), not `torch.distributed`.
- [ ] The node engine is **swappable** behind `ShardEngine` (§7).

M6 (graceful kill-node recovery) is optional polish, not required for Done.
