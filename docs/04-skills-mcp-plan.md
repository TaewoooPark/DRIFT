# 04 — Skills & MCP Plan

Which existing skills/MCPs to use where, the **3 new project skills** that fill DRIFT's
real gaps, and why a custom MCP is not warranted. Scaffolded skills live in
`.claude/skills/` (project-local, auto-discovered).

---

## Existing skills/MCPs — by milestone

| Skill / MCP | Where | Why |
|---|---|---|
| `skill-creator` | Planning / M0 | Author the 3 project skills below |
| `code-review` (+ `simplify`) | Every milestone close-out | Correctness + cleanup on the diff |
| `verify` | Every milestone close-out | Run the app and confirm behavior (esp. M2/M3) |
| `token-efficiency` | Ambient | Cost discipline throughout |
| HF auth MCP (`mcp__claude_ai_Hugging_Face__authenticate`) | M0 / M1 | Convenience for gated-model auth |
| `webapp-testing` / `playwright` | M5 **only if** display is a webpage | Drive/screenshot the booth UI ([`03`](03-goal-execution-plan.md) decision #5) |

Everything else available in the environment (Korean-humanization, physics-lab, paideia,
vercel, arxiv/scholar, Figma/Google) is **irrelevant** to DRIFT and intentionally unused.

---

## New project skills (the gaps no existing skill covers)

The Explore pass found six capability gaps with zero existing coverage. They fold into
three skills:

### 1. `drift-parity-debugger` — **scaffold eagerly (now)**
- **Gaps:** layer-bisection debugging (a) + float max-abs-diff comparison (b).
- **Scope:** fp32 `max_abs_diff`; the `k → diff` bisection sweep; the early-vs-late divergence decision rule; the §13 suspect checklist. Mirrors [`05`](05-parity-debugging-playbook.md).
- **Why eager:** it shapes how `reference.py` / `engine_torch.py` are written to be *bisectable* (e.g. exposing per-boundary hidden states), so it must exist before that code.

### 2. `drift-env-introspect` — **scaffold eagerly (M0-scoped)**
- **Gaps:** HF layer-signature introspection (d) + requirements-lock version-parity validation (f).
- **Scope:** introspect the installed `LlamaDecoderLayer.forward` signature so engine calls match the installed `transformers` (never hardcode — spec §7.2); diff `requirements.lock` vs `requirements.win.lock` and flag any `transformers`/`msgpack` mismatch between the two nodes.
- **Why eager:** prevents the two most insidious silent-parity-breakers (API drift, version skew) from M0 onward.

### 3. `drift-protocol-tester` — **stub now, flesh out before M3**
- **Gaps:** protocol stress-testing (c) + cross-node message tracing (e).
- **Scope:** feed `protocol.py` truncated frames, malformed msgpack, oversized length prefixes, partial `recv`, injected latency; trace `seq_id`/`session_id` and dump frames for cross-node debugging.
- **Why lazy:** only exercised once the socket path exists (M3). A stub now records intent; it's completed when M3 starts.

---

## Custom MCP — **skipped (overkill)**

`mcp-builder` is **not** used. DRIFT's needs (bisection, stress, introspection) are
project-local *procedures* → skills, not a long-running cross-session *service* → MCP. The
only borderline case — a live "fleet control" tool for the booth — is already served by
`orchestrator.py` + `display.py`. Revisit an MCP only post-demo if you want Claude to
interactively drive a multi-node fleet.

---

## Install / activation steps

1. Invoke `skill-creator` for each of the three skills → it scaffolds `.claude/skills/<name>/SKILL.md` (project-local, auto-discovered next session).
2. Reference each skill by name in the relevant milestone of [`01`](01-implementation-plan.md).
3. **Eager now:** `drift-parity-debugger`, `drift-env-introspect`. **Lazy:** `drift-protocol-tester` (stub → complete at M3).

> Status: all three are scaffolded in `.claude/skills/` as part of this docs task (per the
> approved plan). `drift-protocol-tester` is a stub to be completed before M3.
