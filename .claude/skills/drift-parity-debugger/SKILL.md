---
name: drift-parity-debugger
description: >-
  Debug parity failures in DRIFT split inference — when the layer-split path does not
  reproduce the single-machine reference output (spec §9 M2/M3, §13). Provides the fp32
  max-abs-diff comparison, the layer-bisection sweep (move the split point one layer at a
  time to localize the first diverging boundary), and the decision rule: divergence at
  token 1–2 = logic bug (bisect), late divergence on M4 (MPS↔CUDA) = expected float
  difference. Use when parity_test.py fails, token-id sequences differ from
  reference_out.npz, hidden states diverge between shards, or when writing reference.py /
  engine_torch.py so they expose per-boundary hidden states for bisection. Triggers:
  "parity fail", "tokens diverge", "M2/M3 mismatch", "bisect layers", "max-abs-diff",
  "hidden state differs", "split output wrong".
---

# DRIFT Parity Debugger

Companion automation for `docs/05-parity-debugging-playbook.md`. Use it whenever the split
path's output disagrees with the reference oracle.

## The one rule

| First divergence | Meaning | Action |
|---|---|---|
| Token 1–2 (early) | Logic bug | Bisect; check RoPE / KV / mask |
| Late, **M4 only** (MPS↔CUDA) | Expected float-kernel diff | Accept (relaxed M4 gate) |
| Any divergence in **M2/M3** (same device) | Always a bug | fp16 CPU round-trip is lossless; bisect |

Never excuse an M2/M3 mismatch as float noise — same-device math is deterministic and must
be exact. Only M4 (two GPU vendors) earns the relaxed gate.

## Step 1 — fp32 max-abs-diff

```python
import numpy as np
def max_abs_diff(a, b):
    a = a.detach().float().cpu().numpy(); b = b.detach().float().cpu().numpy()
    d = np.abs(a - b)
    return float(d.max()), float(d.mean())
```

- Boundary diff `~0` but token ids differ → bug is **after** the shards (`norm`/`lm_head`/sampler/argmax ties).
- Boundary diff large → bug is **inside** a shard → Step 2.

## Step 2 — layer bisection sweep

For split points `k = 1..N-1`, run `[0,k)` + `[k,N)` for a single prefill and diff the final
hidden state vs the reference at the last position. The smallest `k` where the diff jumps
from `~0` to large localizes the first broken boundary. Emit a `k → (max, mean)` table.

Requires `reference.py` / `engine_torch.py` to expose the post-boundary hidden state — write
them that way from the start (this is why the skill is eager, per `docs/04`).

## Step 3 — suspect list (priority order)

1. **RoPE / position_ids** — each shard computes cos/sin from absolute positions via `model.model.rotary_emb` (layer-agnostic). Wrong positions → early divergence.
2. **KV accumulation** — `decode` appends at the correct absolute position; use `cache_position`, not deprecated `_seen_tokens`. Off-by-one corrupts every token after the first.
3. **Attention mask length** — prefill causal-full; decode KV-length-aware.
4. **Double-applied embed/norm/head** — a shard must run only decoder layers; if it also applies `embed_tokens`/`norm`/`lm_head`, the boundary diff explodes immediately.

## Step 3b — model-specific suspects

**Qwen2.5** — plain decoder; nothing beyond Step 3. (If Qwen is clean but Gemma isn't, the split logic is fine — it's a Gemma quirk below.)

**Gemma 4 E2B** (introspect, never hardcode — use `drift-env-introspect`):
1. **PLE per-layer embeddings** — shard looks up `embed_tokens_per_layer` for its layers from `input_ids` (on the wire); missing/misaligned → diff grows steadily through the shard.
2. **Embedding sqrt(hidden) scaling** — orchestrator must scale `embed_tokens(ids)`; missing → large diff from token 0.
3. **Dual RoPE theta** (sliding ≈10k / global 1M) — shard must use the correct base per layer type; bisect to the first global layer.
4. **Hybrid attention mask** — pass full vs 512-sliding mask per layer by type.
5. **HybridCache + KV-sharing groups** — don't use `DynamicCache` for Gemma; don't split inside a KV-sharing group.
6. **No final-logit softcapping** for Gemma 4 (that was Gemma 2) — don't add it.

## Step 4 — serialization (M2 clean, M3 broken)

The wire path is the only new variable: `_recvn` must loop (partial recv); 4-byte
big-endian length prefix on both ends; `.copy()` after `np.frombuffer` (read-only);
matching `shape`/`dtype`.
