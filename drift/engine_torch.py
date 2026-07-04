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

import copy
import inspect

import torch
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
        embed_duty: bool = False,
        head_duty: bool = False,
    ):
        self.model_id = model_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.dtype = dtype
        self.torch_dtype = _TORCH_DTYPE[dtype]
        self.name = name or f"layers[{start_layer}:{end_layer})"
        self._shared_model = model  # optional pre-loaded model to share (in-process)
        # M10 thin head: the first node also embeds; the last also norms+heads.
        self.embed_duty = embed_duty
        self.head_duty = head_duty

        self.lm = None
        self.inner = None
        self.config = None
        self.rotary = None
        self.layers = None
        self.layer_types = None
        self.has_sliding = False
        self.embed_tokens = None
        self.norm_mod = None
        self.lm_head = None
        self.caches: dict[str, DynamicCache] = {}

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self.lm is not None:
            return
        if self._shared_model is not None:
            # In-process (M2 parity baseline): reference a full, shared model and
            # use only its disjoint layer slice.
            self.lm = self._shared_model
        else:
            # Real node (socket): materialize ONLY this shard's layer slice, so
            # the node never holds the whole model in memory (drift/loader.py).
            # A thin-head edge node also keeps embed_tokens (first node) and/or
            # norm + lm_head (last node) — still only its slice + a tiny edge.
            from transformers import AutoConfig

            from .loader import build_sliced

            tie = bool(getattr(AutoConfig.from_pretrained(self.model_id),
                               "tie_word_embeddings", False))
            keep = [f"model.layers.{i}." for i in range(self.start_layer, self.end_layer)]
            if self.embed_duty:
                keep.append("model.embed_tokens.")
            if self.head_duty:
                keep.append("model.norm.")
                # lm_head weight == embed_tokens when tied; otherwise its own tensor.
                keep.append("model.embed_tokens." if tie else "lm_head.")
            self.lm, _ = build_sliced(
                self.model_id, self.dtype, self.device,
                keep_prefixes=keep, need_rotary=True,
                tie=(None if self.head_duty else False),
            )

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
        # NOTE: this mutates the (possibly shared) model's layer indices, so the
        # parent model object must NOT be used for a full forward() afterward.
        # In-process sharing is safe because shards own DISJOINT layer slices and
        # the orchestrator only uses embed_tokens/norm/lm_head (not the layers).
        for local_i, layer in enumerate(self.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "layer_idx"):
                attn.layer_idx = local_i
            if hasattr(layer, "layer_idx"):
                layer.layer_idx = local_i

        # Introspect the loaded layer's forward params (spec §7.2 — never hardcode
        # the arg list); we pass only kwargs this version's layer actually accepts.
        self._layer_params = set(
            inspect.signature(type(self.layers[0]).forward).parameters
        )

        # A cache config sized to THIS shard's layer count (not the full model),
        # so DynamicCache slots match the re-indexed local layers exactly.
        self._cache_config = copy.copy(self.config)
        try:
            self._cache_config.num_hidden_layers = len(self.layers)
        except Exception:
            self._cache_config = self.config

        # Thin-head edge modules (uses the model's OWN modules, so Gemma's scaled
        # embedding etc. apply automatically — nothing hardcoded).
        if self.embed_duty:
            self.embed_tokens = self.inner.embed_tokens
        if self.head_duty:
            self.norm_mod = self.inner.norm
            self.lm_head = self.lm.lm_head

    # ------------------------------------------------------- thin-head duties
    @torch.no_grad()
    def embed(self, input_ids):
        """Token ids → hidden (first node's duty in thin-head mode)."""
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor([input_ids])
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return self.embed_tokens(input_ids.to(self.device))

    @torch.no_grad()
    def head_argmax(self, hidden) -> int:
        """Hidden → norm → lm_head → greedy next-token id (last node's duty)."""
        logits = self.lm_head(self.norm_mod(hidden[:, -1:, :]))[:, -1, :]
        return int(torch.argmax(logits, dim=-1))

    # --------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, session_id, hidden, position_ids, input_ids, mode):
        self.load()

        cache = self.caches.get(session_id)
        if cache is None:
            cache = DynamicCache(config=self._cache_config)
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
            call_kwargs = {
                "attention_mask": mask,
                "position_embeddings": pos_emb,
                "position_ids": position_ids,
                "past_key_values": cache,
                "use_cache": True,
            }
            # pass only what this transformers version's layer accepts (§7.2)
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in self._layer_params}
            hidden = layer(hidden, **call_kwargs)
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
