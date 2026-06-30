# 05 — Parity Debugging Playbook

The central technical risk of DRIFT is **parity**: the split path must reproduce the
single-machine reference (spec §1.3, §9 M2/M3). This is the procedure for when it doesn't.
Used continuously at M2, again at M3, and in relaxed form at M4. The
[`drift-parity-debugger`](../.claude/skills/drift-parity-debugger/SKILL.md) skill automates
the mechanical parts.

---

## The one rule: where divergence happens tells you what it is

| When the token sequence first diverges | Meaning | Action |
|---|---|---|
| **Token 1–2** (very early) | **Logic bug** | Bisect layers; check RoPE/KV/mask. Do **not** blame hardware. |
| **Late** (after ~10 tokens), **M4 only** (MPS↔CUDA) | Expected float-kernel difference | Accept (relaxed M4 gate). |
| **Any divergence in M2 or M3** (same device) | **Always a bug** | fp16 CPU round-trip is lossless; same-device math is deterministic. Bisect. |

> Never excuse an M2/M3 mismatch as "float noise." On one device it is deterministic and
> must be **exact**. Only M4 (two different GPU vendors) earns the relaxed gate.

---

## Step 1 — fp32 max-abs-diff at the boundary

Compare the hidden state leaving the split path against the reference at the **same
position**, in fp32:

```python
import numpy as np
def max_abs_diff(a, b):
    a = a.detach().float().cpu().numpy(); b = b.detach().float().cpu().numpy()
    d = np.abs(a - b)
    return float(d.max()), float(d.mean())
```

- `~0` (≤ 1e-3 fp16-scale) at the boundary but token ids differ → the bug is **after** the shards (final `norm`/`lm_head`/sampler), or in argmax tie-handling.
- Large diff at the boundary → the bug is **inside** a shard → go to Step 2.

## Step 2 — layer bisection

Move the split point one layer at a time and re-diff. With layers `[0,N)` and a known-good
full reference, sweep `k = 1..N-1`:

1. For each `k`, run the two-shard path `[0,k)` + `[k,N)` for a **single prefill** and diff the final hidden state vs the reference at the last position.
2. The smallest `k` where the diff jumps from `~0` to large localizes the first broken layer boundary.
3. Inspect that boundary's inputs: `position_ids`, the cos/sin from `rotary_emb`, the `attention_mask`, and the per-session `DynamicCache` contents.

The skill runs this sweep and prints a `k → max_abs_diff` table.

## Step 3 — the §13 suspect list (in priority order)

1. **RoPE / `position_ids`** — is each shard computing cos/sin from the *absolute* positions, via `model.model.rotary_emb`? A shard for `[k,N)` must still get correct positions (they are layer-agnostic). Wrong positions → early divergence.
2. **KV cache accumulation** — does `decode` append at the correct absolute position? Use `cache_position`, not the deprecated `_seen_tokens`. Off-by-one here corrupts every token after the first.
3. **Attention mask length** — prefill = causal-full; decode = KV-length-aware. A wrong length silently attends to garbage.
4. **Double-applied embed/norm/head** — a shard must run **only** decoder layers. If a shard also applies `embed_tokens`, `norm`, or `lm_head`, the boundary diff explodes immediately.

## Step 3b — model-specific suspects

**Qwen2.5** — plain decoder; nothing beyond Step 3. If Qwen parity is clean but Gemma isn't, the bug is one of the Gemma quirks below (good signal — the split logic itself is fine).

**Gemma 4 E2B** (introspect every value; never hardcode):
1. **PLE (per-layer embeddings)** — each shard must look up `embed_tokens_per_layer` for *its* layer range from `input_ids` (carried on the wire), scale, and add per layer. Missing/misaligned PLE → diff grows steadily through the shard, not at one boundary. **ORCHESTRATOR/SHARD.**
2. **Embedding sqrt(hidden) scaling** — orchestrator must multiply `embed_tokens(ids)` by `sqrt(hidden_size)`. If missing, the diff is large from token 0. **ORCHESTRATOR.**
3. **Dual RoPE theta** (≈10k sliding / 1M global) — a shard that uses one theta for all its layers breaks the global layers. Bisect to the first global layer in the shard. **SHARD.**
4. **Hybrid attention mask** — per layer, the correct full vs 512-sliding mask must be passed (by layer type). Wrong mask → wrong long-range attention, usually late-ish divergence. **MASK.**
5. **HybridCache + KV-sharing groups** — wrong cache type (using `DynamicCache` for Gemma) or splitting inside a KV-sharing group corrupts decode. **SHARD.**
6. **final logit softcapping** — Gemma 4 dropped it (uses QK-norm in-layer); do **not** add softcapping at `lm_head` for Gemma 4 (that was Gemma 2). **ORCHESTRATOR.**

## Step 4 — serialization (M3-specific)

If M2 is exact but M3 diverges, the bug is in the wire path (the *only* new variable):

- `_recvn` must loop until `n` bytes arrive (partial `recv` is the classic failure).
- Length prefix is 4-byte **big-endian** unsigned, matching both ends.
- `np.frombuffer(...)` returns a **read-only** array — `.copy()` before `torch.from_numpy`.
- `shape`/`dtype` in the message must match what's reconstructed.

---

## Quick reference

```
M2/M3 mismatch  → ALWAYS a bug → Step 1 (fp32 diff) → Step 2 (bisect) → Step 3 (suspects)
M3-only mismatch (M2 clean) → Step 4 (wire)
M4 early mismatch → treat as M2 bug → Step 2
M4 late mismatch  → expected, accept (relaxed gate)
```
