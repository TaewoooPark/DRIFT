# 02 — Workflow Construction Plan

Two layers: (A) the **written build process** that gates milestones, and (B) the
**runnable Claude Code Workflow script** that automates doc review (and, later, code
scaffolding + the parity gate).

---

## A. Written build process (the discipline)

1. **Milestone gating.** Do not start `M(n+1)` until `M(n)`'s §9 acceptance passes. M2 and M3 are **hard stops** (bitwise parity) — see [`03`](03-goal-execution-plan.md).
2. **Per-milestone close-out ritual:**
   - Run the milestone's acceptance test (`reference.py` → `parity_test.py` for M2/M3; `ping` for M0).
   - `/code-review` the diff (correctness + simplification).
   - `/verify` the behavior by actually running it (not just tests).
   - Update the status table in [`README.md`](README.md).
3. **Parity-gate ritual (M2/M3/M4).** On any mismatch, stop feature work and follow [`05`](05-parity-debugging-playbook.md): fp32 diff → bisect → suspect list. Never advance a milestone with a known parity failure.
4. **Wire-contract freeze.** Once `protocol.py`'s §6 schema is written at M0, it is immutable (spec §1.2). Node internals may change; the boundary may not.
5. **Mac-first.** Exhaust the Mac-only track (M0a→M3 + M5 dry-run) before booking Windows/LAN time (`03` now/later split).

---

## B. Runnable Workflow script — `drift-docs-review`

A real Workflow-tool script (multi-agent, deterministic control flow). Its **doc-review
loop** is what runs in the docs phase; the scaffold/parity phases are designed here and run
during implementation. Persisted under the session's workflow directory and summarized
below.

### Phases

| Phase | Mode | What it does | Runs when |
|---|---|---|---|
| 0 — Author | parallel (documented option) | Fan out 3–4 agents to draft the 8 docs from the spec | *Skipped in practice* — the main agent authors directly for cross-doc coherence |
| 1 — Reconcile | sequential | One agent makes shared values identical everywhere (split point, `52600`, model id, `3.12`, transformers pin, filenames) | Docs phase |
| 2 — Review loop | loop (the 2×OK gate) | Fresh reviewer scores the rubric → patch agent fixes findings → repeat until 2 consecutive OK | **Docs phase (primary)** |
| 3 — Scaffold + parity gate | parallel then sequential | Scaffold the loose skeleton (`protocol.py`, `engine_base.py`, `config.yaml`, `reference.py` stub), then a sequential parity-gate runner reused at M2/M3 | Implementation phase (M0–M3) |

**Explicitly NOT automated:** the M2 correctness core (RoPE/KV/mask). That needs
human-gated, bisection-driven iteration ([`05`](05-parity-debugging-playbook.md)) — a
fan-out would burn tokens guessing. The Workflow prepares and verifies; the human debugs.

### Phase 2 control flow (the 2×OK loop)

```js
let consecutiveOK = 0, round = 0
const ledger = []
while (consecutiveOK < 2 && round < 8) {
  round++
  // fresh reviewer each round → independence; re-probes previously-passed dims
  const v = await agent(reviewPrompt(round), { schema: VERDICT_SCHEMA, label: `review:r${round}` })
  ledger.push({ round, verdict: v.status, findings: v.findings })
  if (v.status === 'OK') {
    consecutiveOK++
  } else {
    consecutiveOK = 0
    await agent(patchPrompt(v.findings), { label: `patch:r${round}` })   // edits docs/ in place
  }
}
// append ledger to docs/07-review-log.md; escalate to user if round hit the cap without 2×OK
```

- **Reviewer = fresh subagent every round** (no self-anchoring; the next reviewer adversarially re-checks dimensions the prior one passed).
- **`OK`** requires all 7 rubric dimensions PASS **and** zero blocker/major findings (minors logged).
- **Termination:** `consecutiveOK == 2`. Safety cap 8 rounds, then escalate with residual findings.
- Every round is appended to [`07-review-log.md`](07-review-log.md).

### Phase 3 parity-gate runner (reused M2/M3)

Sequential: `reference.py` (build/refresh oracle) → `parity_test.py` (run split path) →
emit `PASS/FAIL` + the fp32 max-abs-diff and first-divergence token index. This is the
machine form of the M2/M3 acceptance test and feeds [`05`](05-parity-debugging-playbook.md).

> The Workflow tool requires explicit user opt-in (multi-agent, token cost). Opt-in was
> given for this project. Even if a phase is never executed, its design lives here, so the
> process is fully documented regardless.
