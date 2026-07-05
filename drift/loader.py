"""Memory-efficient sliced weight loading (spec follow-up to the v1 full-load).

A node materializes **only** the parameters it will actually run — its decoder-
layer slice (a shard) or `embed_tokens`/`norm`/`lm_head` (the head) — instead of
loading the whole model and using a fraction of it. This is what makes DRIFT's
"no single node holds the whole model" claim true *in memory*, not merely in
compute responsibility.

Mechanism: build the model skeleton on the meta device (`init_empty_weights`, no
allocation), then read from the safetensors shards ONLY the tensors whose name
matches a kept prefix — `safe_open` mmaps the file, so unread tensors never touch
RAM — and assign them straight onto the target device. Unmaterialized modules
stay on meta and must never be called.

The parity gate is the safety net: read the wrong bytes (wrong slice, wrong
dtype) and the very first token diverges from the single-machine reference.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from huggingface_hub.errors import EntryNotFoundError
except Exception:  # older huggingface_hub layout
    from huggingface_hub.utils import EntryNotFoundError

_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _needed_files(model_id: str, keep_prefixes: list[str]) -> list[str]:
    """Local paths to ONLY the safetensors shards that hold a kept tensor.

    For a sharded checkpoint (`model.safetensors.index.json`), download just the
    shards whose `weight_map` names a tensor this node keeps — not the whole
    model. So a node's *disk* footprint is its slice too, and a model too big for
    any single machine can still run once it is split across enough nodes.
    A single-file checkpoint has one shard (the whole model); nothing to prune.
    """
    def want(name: str) -> bool:
        return any(name.startswith(p) for p in keep_prefixes)

    try:
        idx_path = hf_hub_download(model_id, "model.safetensors.index.json")
    except EntryNotFoundError:
        idx_path = None

    if idx_path is not None:
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]
        need = sorted({shard for name, shard in weight_map.items() if want(name)})
        if not need:  # nothing matched (unexpected) — don't silently load nothing
            need = sorted(set(weight_map.values()))
            print(f"[loader] WARNING: no weight matched {keep_prefixes} for "
                  f"{model_id}; loading ALL {len(need)} shard(s) — this node will "
                  f"hold the whole model, voiding the per-shard memory guarantee",
                  file=sys.stderr, flush=True)
        return [hf_hub_download(model_id, shard) for shard in need]

    # single-file checkpoint (no index): one shard = the whole model
    try:
        return [hf_hub_download(model_id, "model.safetensors")]
    except EntryNotFoundError:
        pass
    # last resort: pull whatever .safetensors exist and glob them
    path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(
            f"no safetensors weights for {model_id}; sliced loading needs a "
            f".safetensors checkpoint"
        )
    return files


def _read_subset(model_id: str, keep_prefixes: list[str], device: str,
                 torch_dtype: torch.dtype) -> dict:
    """State dict of only the tensors whose name starts with a kept prefix.

    `safe_open` memory-maps the file; `get_tensor` reads a single tensor, so the
    tensors we skip cost no RAM. Cast matches `from_pretrained(dtype=...)`, so a
    sliced load is byte-identical to the full load of the same slice.
    """
    def want(name: str) -> bool:
        return any(name.startswith(p) for p in keep_prefixes)

    sd: dict = {}
    for fp in _needed_files(model_id, keep_prefixes):
        with safe_open(fp, framework="pt") as sf:
            for name in sf.keys():
                if want(name):
                    t = sf.get_tensor(name)
                    # Match from_pretrained(dtype=…): cast floats only, leave
                    # integer buffers intact (else they'd be silently corrupted).
                    if t.is_floating_point():
                        t = t.to(dtype=torch_dtype)
                    sd[name] = t.to(device=device)
    return sd


def _materialize_rotary(inner, cfg, device: str) -> None:
    """Place the rotary embedding on `device`, bit-exactly.

    `inv_freq` is a non-persistent buffer computed in __init__ (not stored in
    safetensors); `init_empty_weights` does NOT intercept it, so the skeleton
    already holds the correct CPU-computed values. `from_pretrained` likewise
    computes it on CPU and then `.to(device)`s the model — so *moving* the
    existing buffer reproduces that path bit-for-bit. Recomputing on an
    accelerator instead would round `inv_freq` differently (~3e-8) and silently
    break RoPE parity after a few dozen tokens. Rebuild only as a fallback when
    the buffer is genuinely on meta.
    """
    rot = getattr(inner, "rotary_emb", None)
    if rot is None:
        return
    inv = getattr(rot, "inv_freq", None)
    if inv is not None and inv.device.type != "meta":
        inner.rotary_emb = rot.to(device)  # CPU-computed → move (matches from_pretrained)
    else:
        inner.rotary_emb = type(rot)(config=cfg, device=device)  # fallback


def build_sliced(model_id: str, dtype: str, device: str, keep_prefixes: list[str],
                 need_rotary: bool, tie: bool | None = None):
    """Build a model whose only materialized weights are `keep_prefixes`.

    Returns (lm, cfg). Modules outside `keep_prefixes` remain on the meta device
    and must not be invoked.

    - `need_rotary`: shards need RoPE; the head does not.
    - `tie`: when the head keeps only `embed_tokens` (weight-tied models store no
      separate `lm_head.weight`), re-tie after load so `lm_head` points at the
      freshly materialized embedding. Defaults to the model's config.
    """
    torch_dtype = _TORCH_DTYPE[dtype]
    cfg = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        lm = AutoModelForCausalLM.from_config(cfg)
    lm.eval()

    sd = _read_subset(model_id, keep_prefixes, device, torch_dtype)
    # assign=True swaps in the real device tensors; strict=False leaves the
    # unkept modules on meta (never called on this node).
    lm.load_state_dict(sd, strict=False, assign=True)

    if tie is None:
        tie = bool(getattr(cfg, "tie_word_embeddings", False))
    if tie:
        lm.tie_weights()

    if need_rotary:
        _materialize_rotary(lm.model, cfg, device)
    return lm, cfg
