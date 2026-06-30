# 07 — Review Log (rubric + 2×OK ledger)

The static rubric used to harden this docs suite, plus the per-round ledger the review loop
appends to. Termination: a **fresh reviewer returns `OK` twice consecutively**. Mechanism
in [`02`](02-workflow-plan.md) §B Phase 2.

---

## Rubric — 7 dimensions (each PASS / FAIL)

1. **Completeness vs spec** — all 4 requested docs present (impl plan, workflow, goal execution, skills/MCP); every M0–M6 has tasks / files / acceptance / risks / effort; spec §1–§14 doc-relevant concerns covered.
2. **Technical accuracy** — matches the validated findings: `model.model.layers[start:end]`, local RoPE via `rotary_emb` from `position_ids`, per-session cache (`DynamicCache` Qwen / `HybridCache` Gemma), `cache_position` (not `_seen_tokens`); split points match layer counts (Qwen 28→14/14, Gemma 4 E2B 35→18/17); Gemma 4 quirks correct (PLE needs `input_ids` on the wire, dual-rope, hybrid attention, no final-logit softcapping); models are ungated.
3. **No hardcoded wrong API** — docs *instruct introspection* of the installed `transformers` (spec §7.2); never present a fixed HF forward-arg list as gospel.
4. **Actionability** — commands are copy-pasteable (`uv venv --python 3.12`, exact `pip`, `huggingface-cli login`); acceptance tests runnable; paths absolute/consistent.
5. **Internal consistency** — split point, port `52600`, model id, Python `3.12`, transformers pin, filenames identical across all docs; cross-references valid.
6. **Alignment with §1 hard constraints** — no `torch.distributed`/NCCL/gloo anywhere; wire contract immutable; correctness-first ordering; engine isolated behind `ShardEngine`; token economy out of scope.
7. **Scope discipline** — no creep (no replication/seamless failover beyond M6 graceful; no token economy; no premature optimization).

**Verdict rule:** `OK` requires **all 7 dimensions PASS** *and* **zero blocker/major
findings** (minor findings allowed but logged). Otherwise `NEEDS-WORK`.

**Reviewer verdict format:**
```
VERDICT: OK | NEEDS-WORK
DIMENSIONS: {1..7: PASS|FAIL}
FINDINGS: [{ dim, file, severity: blocker|major|minor, issue, suggested_fix }]
```

**Independence:** a **fresh** reviewer subagent each round; the next reviewer adversarially
re-probes the dimensions the previous one passed. `consecutiveOK` increments on `OK`,
resets to 0 on any `NEEDS-WORK`. Terminate at `2`. Safety cap 8 rounds → escalate.

---

## Ledger

### Run 1 — initial suite (Llama/Qwen)

Run `drift-docs-review` (Workflow `wf_9928d3ed-f88`), fresh reviewer each round. **Terminated at round 3 with `consecutiveOK == 2`.**

| Round | Reviewer | Verdict | Blocker/Major | Minor | consecutiveOK | Patches applied |
|---|---|---|---|---|---|---|
| 1 | fresh #1 | NEEDS-WORK | 1 | 1 | 0 | Added `Risks` lines to M5 & M6 in `01`; acknowledged spec §11 llama.cpp fallback as out-of-main-path |
| 2 | fresh #2 | OK | 0 | 1 | 1 | — (minor logged) |
| 3 | fresh #3 | OK | 0 | 1 | 2 ✅ | — (minor logged) |

**Result:** 3 rounds, `terminated: 2xOK`. All 7 dimensions PASS on rounds 2 and 3 with zero blocker/major findings.

### Run 2 — model revision (Qwen primary + Gemma 4 E2B secondary)

Models changed from Llama/Qwen to **Qwen2.5-1.5B (primary) + gemma-4-E2B-it (secondary)** — both ungated; Gemma 4 adds PLE / dual-rope / hybrid-attention handling and `input_ids` on the wire. **Validated by two independent fresh reviewers.** (The `drift-docs-review` workflow stalled on an infra timeout mid-run, so the review was completed with direct, bounded fresh-reviewer agent calls applying the same 7-dimension rubric.)

| Round | Reviewer | Verdict | Blocker/Major | Minor | consecutiveOK | Patches applied |
|---|---|---|---|---|---|---|
| 1 | fresh A | OK | 0 | 0 | 1 | — |
| 2 | fresh B (adversarial) | OK | 0 | 0 | 2 ✅ | — |

**Result:** `terminated: 2xOK`. Both independent reviewers passed all 7 dimensions with zero findings on the revised (Qwen + Gemma 4) suite — no patches needed.

### Run 1 findings detail

- **R1 · major · d1 · `01`** — M5/M6 lacked the rubric-required `Risks` field that M0–M4 carry. *Fixed in round-1 patch.*
- **R1 · minor · d1 · `01`** — spec §11 llama.cpp booth fallback not surfaced while §12 (v2 MLX) was. *Fixed in round-1 patch.*
- **R2 · minor · d5 · `06`/`01`** — `shard_server` CLI overrides (`--name/--start/--end/--device`) + `DRIFT_PORT` env not documented vs the `config.yaml` schema. *Applied as post-loop polish (`01` M0 tasks 6–7).*
- **R3 · minor · d4 · `06`** — localhost two-port smoke test didn't reconcile the second port (`52601`) with the single-port config schema. *Applied as post-loop polish (`06`: `--ports` + M4 single-port note).*

> The two minor findings from the OK rounds were non-blocking (minors don't gate `OK`) but
> were applied afterward for completeness — see the round-1-patched `01` and the post-loop
> edits to `01`/`06`.

---

## Implementation review — M0–M3 code (Exo + references)

After implementing M0–M3, the `drift/` code was reviewed against **Exo** and other
split-inference references (Petals, llama.cpp RPC, vLLM pipeline parallelism), patching
until **2 consecutive all-OK**. Fresh independent reviewers each round (direct agent calls
— the Workflow path had stalled on the docs run).

| Round | Reviewers | Verdict | Blocker/Major | Notes |
|---|---|---|---|---|
| 1 | correctness + Exo/design | both OK | 0 | clean |
| 2 | adversarial-break + spec/reference | both OK | 0 | 6 minors logged (introspected kwargs, EOS breadth, cache sizing, reindex doc, edge-case coverage); Exo's MLX/`mx.distributed` Apple-only coupling re-verified from source |
| 3 | patch-correctness + minors/doc-consistency | both OK | 0 | all round-2 minors **resolved**; ZERO findings |

**Result:** 2 consecutive all-OK (rounds 2 & 3); round 3 fully clean on the hardened code.
Empirically verified: `parity_test --selftest` = 6/6 bitwise-equal to a clean reference
(English/code/Korean; n = 1, 40, 50, 60, 80, 180). Releases: **v0.1.0** (pre-patch reviewed
state) and **v0.1.1** (hardened, round-3-clean).

**Scope:** M0–M3 done on the Mac (100% of the correctness risk). M4 (Windows MPS+CUDA),
M5 (booth display), M6 (kill-node resilience) need the second node.
