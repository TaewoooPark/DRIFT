---
name: drift-env-introspect
description: >-
  Guard DRIFT against the silent parity-breakers — Hugging Face API drift, cross-node
  version skew, and unhandled model-architecture quirks. Introspects the LOADED model's
  decoder-layer forward signature (Qwen2DecoderLayer / Gemma4DecoderLayer — never hardcode
  HF args, spec §7.2) plus the model's architecture profile (cache type, RoPE theta(s),
  per-layer attention types, PLE per-layer embeddings, embedding scaling, KV-sharing
  groups), and diffs requirements.lock (Mac) vs requirements.win.lock (Windows) to flag any
  transformers/msgpack mismatch (transformers >= 5.5 for Gemma 4). Use at M0, before writing
  engine_torch.forward, and before the M4 cross-machine run. Triggers: "introspect forward
  signature", "which transformers args", "version skew", "lock mismatch", "requirements
  parity", "HF API changed", "decoder layer signature", "gemma quirks", "cache type", "PLE".
---

# DRIFT Env Introspect

Companion to `docs/06-m0-setup-runbook.md` and `docs/01` (M0/M2/M4). Two jobs.

## 1. Introspect the decoder-layer forward signature

The **loaded model** is the source of truth — not any doc, and not a fixed model class.
Introspect whatever was actually loaded (works for Qwen, Gemma 4, anything):

```python
import inspect
layer_cls = type(model.model.layers[0])     # e.g. Qwen2DecoderLayer / Gemma4DecoderLayer
print(layer_cls.__name__, inspect.signature(layer_cls.forward))
```

Confirm which of these the installed version accepts and wire calls accordingly (do **not**
hardcode a remembered list): `hidden_states`, `attention_mask`, `position_ids`,
`past_key_values`/`past_key_value`, `use_cache`, `position_embeddings=(cos, sin)`,
`cache_position`. Modern versions compute RoPE once at model level and pass
`position_embeddings`; shards may instead pass only `position_ids` and compute cos/sin
locally via `model.model.rotary_emb`.

Also confirm the module paths exist on this version: `model.model.layers`,
`model.model.embed_tokens`, `model.model.norm`, `model.lm_head`, `model.model.rotary_emb`.

## 1b. Profile the model's architecture quirks

Different families need different handling. Introspect the config so the engine adapts
instead of assuming plain-decoder simplicity:

```python
c = model.config
print("layers      :", c.num_hidden_layers)
print("rope_theta  :", getattr(c, "rope_theta", None))            # scalar (Qwen) vs dual local/global (Gemma)
print("layer_types :", getattr(c, "layer_types", None))           # per-layer sliding vs full attention (Gemma)
print("sliding_win :", getattr(c, "sliding_window", None))
print("tie_embeds  :", getattr(c, "tie_word_embeddings", None))
print("has_PLE     :", hasattr(model.model, "embed_tokens_per_layer"))  # Gemma 4 per-layer embeddings
print("softcap     :", getattr(c, "final_logit_softcapping", None))     # Gemma 2 only; None for Qwen/Gemma 4
```

Then decide per model: cache type (`DynamicCache` if no sliding window, else `HybridCache`);
whether the orchestrator must apply **embedding sqrt(hidden) scaling** and feed **`input_ids`**
to shards (PLE); whether shards need **per-layer RoPE theta** and **per-layer attention masks**;
and any **KV-sharing layer groups** to avoid splitting inside. See `docs/05` model-specific suspects.

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
