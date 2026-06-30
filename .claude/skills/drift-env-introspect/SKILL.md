---
name: drift-env-introspect
description: >-
  Guard DRIFT against the two silent parity-breakers — Hugging Face API drift and
  cross-node version skew. Introspects the installed transformers' LlamaDecoderLayer.forward
  signature so engine_torch.py calls match the actual installed version (never hardcode HF
  args — spec §7.2), and diffs requirements.lock (Mac) vs requirements.win.lock (Windows) to
  flag any transformers/msgpack mismatch between the two nodes. Use at M0, before writing
  engine_torch.forward, and before the M4 cross-machine run. Triggers: "introspect forward
  signature", "which transformers args", "version skew", "lock mismatch", "requirements
  parity", "HF API changed", "decoder layer signature".
---

# DRIFT Env Introspect

Companion to `docs/06-m0-setup-runbook.md` and `docs/01` (M0/M2/M4). Two jobs.

## 1. Introspect the decoder-layer forward signature

The installed `transformers` is the source of truth — not any doc. Before wiring
`engine_torch.forward`, print the real signature:

```python
import inspect
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
print(inspect.signature(LlamaDecoderLayer.forward))
```

Confirm which of these the installed version accepts and wire calls accordingly (do **not**
hardcode a remembered list): `hidden_states`, `attention_mask`, `position_ids`,
`past_key_values`/`past_key_value`, `use_cache`, `position_embeddings=(cos, sin)`,
`cache_position`. Modern versions compute RoPE once at model level and pass
`position_embeddings`; shards may instead pass only `position_ids` and compute cos/sin
locally via `model.model.rotary_emb`.

Also confirm the module paths exist on this version: `model.model.layers`,
`model.model.embed_tokens`, `model.model.norm`, `model.lm_head`, `model.model.rotary_emb`.

## 2. Validate cross-node version parity

`torch` differs by device (MPS vs CUDA) — expected. But `transformers` and `msgpack` must be
**identical** on both nodes or HF internals introspect differently and parity breaks.

```python
# diff the two locks; flag any non-torch package whose pinned version differs
def diff_locks(mac_lock: str, win_lock: str):
    def parse(p):
        out = {}
        for line in open(p):
            line = line.strip()
            if "==" in line and not line.startswith("#"):
                name, ver = line.split("==", 1); out[name.lower()] = ver
        return out
    a, b = parse(mac_lock), parse(win_lock)
    critical = ("transformers", "msgpack", "safetensors", "numpy")
    return {k: (a.get(k), b.get(k)) for k in critical if a.get(k) != b.get(k)}
```

Any non-empty result for `transformers`/`msgpack` is a **blocker** for M4 — align the
versions before running cross-machine.
