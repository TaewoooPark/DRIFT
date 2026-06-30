"""DRIFT — Decentralized Routed Inference For Tokens.

Heterogeneous split inference: one LLM split by decoder layer across nodes that
exchange hidden states over a framework-neutral TCP + msgpack protocol (NOT
torch.distributed). See DRIFT-implementation-spec.md and docs/.
"""

__all__ = ["protocol", "engine_base", "engine_torch", "shard_server", "orchestrator", "config"]
