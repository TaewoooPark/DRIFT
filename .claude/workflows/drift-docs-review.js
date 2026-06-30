export const meta = {
  name: 'drift-docs-review',
  description: 'Harden the DRIFT docs/ suite via a fresh-reviewer 2x-consecutive-OK loop',
  phases: [
    { title: 'Review', detail: 'fresh reviewer scores the 7-dimension rubric each round' },
    { title: 'Patch', detail: 'fix blocker/major findings in docs/ on NEEDS-WORK' },
  ],
}

const ROOT = '/Users/taewoopark/personal/DRIFT'
const SPEC = `${ROOT}/DRIFT-implementation-spec.md`
const DOCS = [
  `${ROOT}/docs/README.md`,
  `${ROOT}/docs/01-implementation-plan.md`,
  `${ROOT}/docs/02-workflow-plan.md`,
  `${ROOT}/docs/03-goal-execution-plan.md`,
  `${ROOT}/docs/04-skills-mcp-plan.md`,
  `${ROOT}/docs/05-parity-debugging-playbook.md`,
  `${ROOT}/docs/06-m0-setup-runbook.md`,
  `${ROOT}/docs/07-review-log.md`,
]
const OTHER = [
  `${ROOT}/README.md`,
  `${ROOT}/.claude/skills/drift-parity-debugger/SKILL.md`,
  `${ROOT}/.claude/skills/drift-env-introspect/SKILL.md`,
  `${ROOT}/.claude/skills/drift-protocol-tester/SKILL.md`,
]

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['status', 'dimensions', 'findings', 'summary'],
  properties: {
    status: { type: 'string', enum: ['OK', 'NEEDS-WORK'] },
    dimensions: {
      type: 'object',
      additionalProperties: false,
      required: ['d1_completeness','d2_technical','d3_no_hardcoded_api','d4_actionability','d5_consistency','d6_constraints','d7_scope'],
      properties: {
        d1_completeness: { type: 'string', enum: ['PASS','FAIL'] },
        d2_technical: { type: 'string', enum: ['PASS','FAIL'] },
        d3_no_hardcoded_api: { type: 'string', enum: ['PASS','FAIL'] },
        d4_actionability: { type: 'string', enum: ['PASS','FAIL'] },
        d5_consistency: { type: 'string', enum: ['PASS','FAIL'] },
        d6_constraints: { type: 'string', enum: ['PASS','FAIL'] },
        d7_scope: { type: 'string', enum: ['PASS','FAIL'] },
      },
    },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['dim','file','severity','issue','suggested_fix'],
        properties: {
          dim: { type: 'string' },
          file: { type: 'string' },
          severity: { type: 'string', enum: ['blocker','major','minor'] },
          issue: { type: 'string' },
          suggested_fix: { type: 'string' },
        },
      },
    },
    summary: { type: 'string' },
  },
}

const RUBRIC = `7 dimensions (each PASS/FAIL). A dimension FAILs ONLY for a real blocker/major issue — do NOT fail a dimension for wording nitpicks (log those as minor findings).
1. d1_completeness: all 4 requested docs present (impl plan=01, workflow=02, goal execution=03, skills/MCP=04); every milestone M0-M6 in 01 has tasks/files/acceptance/risks/effort; spec sections are covered.
2. d2_technical: matches validated facts — model.model.layers[start:end]; embed_tokens/model.norm/lm_head/model.model.rotary_emb; LlamaDecoderLayer.forward accepts position_ids AND position_embeddings=(cos,sin); shards compute RoPE locally from position_ids via rotary_emb (layer-agnostic); per-session DynamicCache using cache_position (NOT deprecated _seen_tokens); split points match layer counts (Llama 16->8/8, Qwen 28->14/14).
3. d3_no_hardcoded_api: docs INSTRUCT introspection of the installed transformers (spec section 7.2); they must never present a fixed HF forward-arg list as authoritative truth.
4. d4_actionability: commands copy-pasteable (uv venv --python 3.12, exact pip, huggingface-cli login); acceptance tests runnable; paths consistent.
5. d5_consistency: split point, port 52600, model id, Python 3.12, transformers pin, filenames identical across ALL docs; cross-references valid. VERIFY by grepping shared values across files.
6. d6_constraints: aligns with spec section 1 hard constraints — NO torch.distributed/NCCL/gloo anywhere (except correctly described as forbidden); wire contract immutable; correctness-first; engine isolated behind ShardEngine; token economy out of scope.
7. d7_scope: no scope creep (no replication/seamless failover beyond M6 graceful; no token-economy implementation).`

function reviewerPrompt(round) {
  const adversarial = round > 1
    ? `\n\nThis is review round ${round}. A PRIOR reviewer may have passed dimensions that still have issues — be ADVERSARIAL and re-probe every dimension independently, especially ones easy to wave through. Do not anchor on prior verdicts.`
    : ''
  return `You are an independent, rigorous documentation reviewer for the DRIFT project. Review the docs/ suite for quality and correctness, then emit a structured verdict.

SOURCE OF TRUTH (read, do NOT edit, do NOT flag for editing): ${SPEC}
DOCS TO REVIEW (read all): ${DOCS.join(', ')}
ALSO IN SCOPE: ${OTHER.join(', ')}

Read every file. Use Grep to VERIFY dimension 5 (consistency) — grep for "52600", "Llama-3.2-1B", "3.12", "14/14", "8/8" across the docs and confirm they agree. Use Grep for dimension 6 — confirm "torch.distributed"/"NCCL"/"gloo" appear ONLY as forbidden, and that "_seen_tokens" appears only as deprecated.

RUBRIC:
${RUBRIC}

VERDICT RULE: status = OK iff ALL 7 dimensions PASS and there are ZERO blocker/major findings (minor findings are allowed and should still be logged). Otherwise status = NEEDS-WORK.

IMPORTANT NOTES:
- docs/07-review-log.md contains a ledger table whose rows are populated by THIS review process. Its current "_pending_" ledger is EXPECTED — do NOT flag the empty/pending ledger as an incompleteness issue. Review only its rubric content.
- The spec is Korean and authoritative; the docs are intentionally English. That is correct, not a finding.
- Be precise in findings: give the exact file, the dimension, a concrete issue, and a concrete suggested_fix. Prefer fewer high-signal findings over many nitpicks.${adversarial}

Return the structured verdict.`
}

function patchPrompt(v) {
  return `You are a documentation patch engineer for the DRIFT project. A reviewer returned NEEDS-WORK. Apply fixes that resolve EVERY blocker/major finding and clear every FAILed dimension. Address minor findings too when low-risk.

CONSTRAINTS:
- Edit ONLY files under ${ROOT}/docs/, plus ${ROOT}/README.md and the SKILL.md files under ${ROOT}/.claude/skills/ if a finding targets them.
- NEVER edit ${SPEC} (the immutable source of truth).
- Do NOT fill in the ledger table rows in docs/07-review-log.md — that is done after the loop. You may fix the rubric text in 07 if a finding targets it.
- Preserve the docs' existing structure, English language, and the canonical values defined in docs/03 (port 52600, Python 3.12, model ids, splits). If a finding is about inconsistency, fix the WRONG copies to match docs/03, do not change docs/03 unless docs/03 itself is the error.
- Use Read then Edit for surgical changes.

REVIEWER SUMMARY: ${v.summary}

FINDINGS TO RESOLVE (JSON):
${JSON.stringify(v.findings, null, 2)}

Apply all fixes now. When done, return a one-line summary of what you changed per file.`
}

function effectiveStatus(v) {
  const allPass = Object.values(v.dimensions).every((x) => x === 'PASS')
  const noBlockerMajor = !v.findings.some((f) => f.severity === 'blocker' || f.severity === 'major')
  return allPass && noBlockerMajor ? 'OK' : 'NEEDS-WORK'
}

let consecutiveOK = 0
let round = 0
const MAX_ROUNDS = 8
const ledger = []

phase('Review')
while (consecutiveOK < 2 && round < MAX_ROUNDS) {
  round++
  const v = await agent(reviewerPrompt(round), { schema: VERDICT_SCHEMA, label: `review:r${round}`, phase: 'Review' })
  if (!v) {
    log(`round ${round}: reviewer returned null — aborting`)
    ledger.push({ round, status: 'ERROR', dimensions: {}, blockerMajor: 0, minor: 0, findings: [], summary: 'reviewer agent returned null' })
    break
  }
  const status = effectiveStatus(v)
  const blockerMajor = v.findings.filter((f) => f.severity !== 'minor')
  const minor = v.findings.filter((f) => f.severity === 'minor')
  ledger.push({ round, status, dimensions: v.dimensions, blockerMajor: blockerMajor.length, minor: minor.length, findings: v.findings, summary: v.summary })
  if (status === 'OK') {
    consecutiveOK++
    log(`round ${round}: OK (consecutive ${consecutiveOK}/2) — ${minor.length} minor finding(s) logged`)
  } else {
    consecutiveOK = 0
    log(`round ${round}: NEEDS-WORK — ${blockerMajor.length} blocker/major, ${minor.length} minor — patching`)
    await agent(patchPrompt(v), { label: `patch:r${round}`, phase: 'Patch' })
  }
}

return {
  rounds: round,
  consecutiveOK,
  terminated: consecutiveOK >= 2 ? '2xOK' : 'cap-or-error',
  ledger,
}
