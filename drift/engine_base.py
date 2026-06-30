"""ShardEngine interface — the core of swappability (spec §7).

The node-internal inference engine lives behind this interface. Today it is
PyTorch (MPS/CUDA); tomorrow it could be MLX. The wire boundary (protocol.py)
never changes, so heterogeneous runtimes can cooperate — DRIFT's key property.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ShardEngine(ABC):
    """Runs a contiguous slice of decoder layers [start_layer, end_layer)."""

    @abstractmethod
    def load(self) -> None:
        """Load weights for this shard's layer range and capture the rotary
        module. Idempotent."""

    @abstractmethod
    def forward(self, session_id: str, hidden, position_ids, input_ids, mode: str):
        """Run layers [start, end) over `hidden` and return the new hidden.

        mode: "prefill" (whole prompt) or "decode" (S=1). Updates the per-session
        KV cache internally. `position_ids` are absolute; `input_ids` is provided
        for PLE-style models (plain models ignore it).
        """

    @abstractmethod
    def reset(self, session_id: str) -> None:
        """Discard the KV cache for a session (generation finished)."""

    @abstractmethod
    def ping_info(self) -> dict:
        """Return identity for health checks: name/start_layer/end_layer/device."""
