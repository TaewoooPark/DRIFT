"""PyTorch ShardEngine (MPS/CUDA/CPU) — spec §7.

Runs a contiguous slice of decoder layers. The HF internals are *introspected*
(loaded model's own modules and the masking/rotary utilities), never hardcoded,
so the same engine serves Qwen, Gemma 4, etc. (drift-env-introspect principle).

Correctness note (KV indexing, docs/05): a shard holding global layers [k, N)
re-indexes those layers' attention `layer_idx` to local 0..n-1, so a fresh
per-session `DynamicCache` (indexed from 0) reports the correct past length.
Without this, the decode-time causal mask would be built from an empty layer-0
slot and parity would break after the first token.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from .engine_base import ShardEngine

_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class TorchShardEngine(ShardEngine):
    def __init__(
        self,
        model_id: str,
        start_layer: int,
        end_layer: int,
        device: str,
        dtype: str = "float16",
        name: str | None = None,
        model=None,
    ):
        self.model_id = model_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.dtype = dtype
        self.torch_dtype = _TORCH_DTYPE[dtype]
        self.name = name or f"layers[{start_layer}:{end_layer})"
        self._shared_model = model  # optional pre-loaded model to share (in-process)

        self.lm = None
        self.inner = None
        self.config = None
        self.rotary = None
        self.layers = None
        self.layer_types = None
        self.has_sliding = False
        self.caches: dict[str, DynamicCache] = {}

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self.lm is not None:
            return
        if self._shared_model is not None:
            self.lm = self._shared_model
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=self.torch_dtype
            )
            self.lm.to(self.device)
            self.lm.eval()

        self.inner = self.lm.model  # the text transformer (Qwen2Model / Gemma4TextModel)
        self.config = self.inner.config
        self.rotary = self.inner.rotary_emb
        self.has_sliding = getattr(self.inner, "has_sliding_layers", False)

        all_layers = self.inner.layers
        self.layers = [all_layers[i] for i in range(self.start_layer, self.end_layer)]

        gtypes = getattr(self.config, "layer_types", None) or (
            ["full_attention"] * self.config.num_hidden_layers
        )
        self.layer_types = [gtypes[i] for i in range(self.start_layer, self.end_layer)]

        # Re-index kept layers to local cache slots (see module docstring).
        for local_i, layer in enumerate(self.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "layer_idx"):
                attn.layer_idx = local_i
            if hasattr(layer, "layer_idx"):
                layer.layer_idx = local_i

    # --------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, session_id, hidden, position_ids, input_ids, mode):
        self.load()

        cache = self.caches.get(session_id)
        if cache is None:
            cache = DynamicCache(config=self.config)
            self.caches[session_id] = cache

        hidden = hidden.to(device=self.device, dtype=self.torch_dtype)

        if not torch.is_tensor(position_ids):
            position_ids = torch.tensor(position_ids, dtype=torch.long)
        position_ids = position_ids.to(self.device).long()
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        mask_kwargs = dict(
            config=self.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=cache,
            position_ids=position_ids,
        )
        mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.has_sliding:
            mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                **mask_kwargs
            )

        pos_emb = self.rotary(hidden, position_ids)

        for local_i, layer in enumerate(self.layers):
            mask = mask_mapping.get(self.layer_types[local_i], mask_mapping["full_attention"])
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_embeddings=pos_emb,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
        return hidden

    # ----------------------------------------------------------------- reset
    def reset(self, session_id: str) -> None:
        self.caches.pop(session_id, None)

    def ping_info(self) -> dict:
        return {
            "name": self.name,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "device": self.device,
            "loaded": self.lm is not None,
        }
