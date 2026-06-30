# 03 — Goal Execution Plan

Maps the spec's goals (§1) to milestones (§9), fixes the canonical values every other doc
cites, sets the dependency order and the now/later split, records the open decisions, and
states the Definition of Done (§14).

---

## Canonical values (defined here — cite, never redefine)

| Key | Value | Source / note |
|---|---|---|
| Primary model | `meta-llama/Llama-3.2-1B-Instruct` | spec §5; **gated** on HF |
| Primary layers / split | 16 layers → `0–8` (Mac) / `8–16` (Windows) | verify via `AutoConfig` at M1 |
| Fallback model | `Qwen/Qwen2.5-1.5B-Instruct` | non-gated |
| Fallback layers / split | 28 layers → `0–14` / `14–28` | used if HF gating not cleared |
| Boundary dtype | `float16` | spec §5/§6; CPU round-trip is lossless |
| TCP port | `52600` | spec §5; localhost uses two ports (see `06`) |
| Python | `3.12` (via `uv venv --python 3.12`) | **not 3.14** — no torch wheel |
| `transformers` | pinned in `requirements.lock`, **identical on both nodes** | target ≥ 4.44; parity-critical |
| `torch` | per-machine (Mac MPS wheel / Windows CUDA wheel) | versions differ by device — that's expected |
| Code dir | `drift/` | spec §4 |
| Repo root | `/Users/taewoopark/personal/DRIFT` | — |

---

## Goal ↔ milestone matrix

| Goal (spec §1) | Proven by |
|---|---|
| Neutral data plane (no `torch.distributed`/NCCL/gloo) | **M0** (ping over msgpack/TCP), **M3** (lossless tensor transport) |
| Correctness-first / parity | **M1** (oracle), **M2** (sharding/RoPE/KV), **M3** (serialization) |
| Heterogeneous cooperation — *the differentiator* | **M4** (MPS + CUDA on one model) |
| Swappable engine behind an interface | **M2** (`ShardEngine` ABC in use); fully realized in v2 (§12) |
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
| 1 | Model | **Llama-3.2-1B-Instruct first**; auto-fallback to Qwen2.5-1.5B-Instruct if HF gating isn't cleared in ~15 min | M0/M1 |
| 2 | Split point | Match layer count (Llama 8/8, Qwen 14/14); verify via `AutoConfig`; memory-weighting deferred | M1 |
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
