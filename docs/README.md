# DRIFT — Documentation Suite

This folder operationalizes [`../DRIFT-implementation-spec.md`](../DRIFT-implementation-spec.md)
(the authoritative technical blueprint, written in Korean) into an **executable plan**:
phased tasks, a build-orchestration workflow, a project tooling plan, and the quality
gates used to harden these docs.

> **Source of truth.** The spec is authoritative. These docs *cite* it (e.g. "see §9 M2")
> and must never restate or contradict its numbers. Any value shared across docs (split
> point, port, model id, Python/transformers version) is **defined once** in
> [`03-goal-execution-plan.md`](03-goal-execution-plan.md) and cited elsewhere.

## What DRIFT is (one paragraph)

A Mac (PyTorch **MPS**, decoder layers `[0,k)`) and a Windows PC (PyTorch **CUDA**, layers
`[k,N)`) jointly run **one** LLM, split by layer (pipeline parallelism), exchanging hidden
states over a **framework-neutral TCP + msgpack protocol** — explicitly **not**
`torch.distributed`/NCCL. The neutral data plane is the whole point: heterogeneous
runtimes and GPU vendors cooperate on one model (unlike Exo, which is MLX/Apple-only).
This repo implements the working demo of the **D·R·I** slice; the "For Tokens" economy is
out of scope (spec §1.5).

## Files & reading order

| # | File | Purpose |
|---|---|---|
| — | [`README.md`](README.md) | This index + glossary |
| 03 | [`03-goal-execution-plan.md`](03-goal-execution-plan.md) | **Start here.** Goals↔milestones, dependency order, now/later split, canonical values, Definition of Done |
| 01 | [`01-implementation-plan.md`](01-implementation-plan.md) | Phase-by-phase M0–M6: tasks, files, acceptance, risks, effort |
| 06 | [`06-m0-setup-runbook.md`](06-m0-setup-runbook.md) | Copy-paste environment setup (run this on the Mac now) |
| 05 | [`05-parity-debugging-playbook.md`](05-parity-debugging-playbook.md) | Keep open during M2–M4: bisection + fp32 diff |
| 02 | [`02-workflow-plan.md`](02-workflow-plan.md) | Build process + the runnable review Workflow |
| 04 | [`04-skills-mcp-plan.md`](04-skills-mcp-plan.md) | Skills/MCP per milestone; the 3 project skills |
| 07 | [`07-review-log.md`](07-review-log.md) | Doc-review rubric + the 2×OK ledger |
| 08 | [`08-benchmark-plan.md`](08-benchmark-plan.md) | How to benchmark vs similar tools + measured results; harness is [`../drift/bench.py`](../drift/bench.py) |

**Recommended path:** `03` (the map) → `01` (the steps) → `06` (do setup now) → `05`
(have it ready for M2) → `02` → `04` → `07`.

## Status

| Track | State |
|---|---|
| Spec (§0–§14) | ✅ complete (authoritative) |
| Docs suite | ✅ authored, ⏳ hardened to 2×OK (see `07`) |
| Project skills | ✅ scaffolded in `.claude/skills/` (see `04`) |
| Mac env (M0a) | ⏳ deps not installed — run `06` |
| Code (`drift/`) | ⬜ not started — follows from `01` |

## Glossary (10 terms)

- **Shard** — a node that holds a contiguous slice of decoder layers `[start,end)`.
- **Orchestrator** — drives the decode loop; owns tokenizer, `embed_tokens`, final `norm`+`lm_head`, sampler; routes hidden states through shards in order.
- **Wire contract** — the immutable §6 message schema (4-byte length prefix + msgpack dict). Frozen once set.
- **Data plane** — what crosses the boundary: `hidden_states` (floats) + `position_ids` (ints). Framework-neutral.
- **Control plane** — orchestrator calling shards in configured order (no separate discovery).
- **Parity** — bitwise-equal token-id output between the split path and the single-machine reference (the M2/M3 gate).
- **Reference oracle** — `reference.py` greedy output saved once; the parity ground truth (M1).
- **Bisection** — moving the split point one layer at a time to localize where a hidden state first diverges (§13).
- **Relaxed gate** — M4's acceptance: coherent output + early-token match, tolerating late MPS↔CUDA float divergence.
- **2×OK** — the doc-review termination: a fresh reviewer must return `OK` twice consecutively.
