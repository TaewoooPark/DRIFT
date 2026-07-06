"""Shared helpers: config loading and identical tokenization.

The reference oracle (reference.py) and the split path (orchestrator.py) MUST
tokenize a prompt identically, or parity is meaningless. Both call
`build_input_ids` here.
"""

from __future__ import annotations

import yaml


def _fallback_chat_prompt(messages: list[dict]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role") or "user"
        text = msg.get("content") or ""
        if text:
            lines.append(f"{role.capitalize()}: {text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_input_ids(tokenizer, prompt: str | list[dict]):
    """Deterministic input_ids [1, S] via the model's chat template.

    Falls back to plain encoding if the tokenizer has no chat template.
    """
    if isinstance(prompt, list):
        if getattr(tokenizer, "chat_template", None):
            try:
                out = tokenizer.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                return out["input_ids"]
            except Exception:
                pass
        return tokenizer(_fallback_chat_prompt(prompt), return_tensors="pt").input_ids

    if getattr(tokenizer, "chat_template", None):
        try:
            out = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            return out["input_ids"]
        except Exception:
            pass
    return tokenizer(prompt, return_tensors="pt").input_ids


# --------------------------------------------------------------- UX helpers
# Small, dependency-light helpers that let the launchers auto-configure a run
# instead of making the user hand-write layer ranges, devices, and IPs.

def pick_device(prefer: str | None = None) -> str:
    """Best available torch device: mps (Apple) → cuda (NVIDIA) → cpu.

    Pass an explicit device to force it; ``None``/``"auto"`` auto-detects.
    """
    if prefer and prefer != "auto":
        return prefer
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def split_layers(n_layers: int, n_nodes: int) -> list[tuple[int, int]]:
    """Tile ``[0, n_layers)`` into ``n_nodes`` contiguous half-open ranges.

    Even split; the first ``n_layers % n_nodes`` shards get one extra layer.
    Guarantees no gaps/overlaps — the split-point invariant the manual states.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    if n_nodes > n_layers:
        raise ValueError(f"cannot split {n_layers} layers across {n_nodes} nodes")
    base, extra = divmod(n_layers, n_nodes)
    ranges, start = [], 0
    for i in range(n_nodes):
        size = base + (1 if i < extra else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def model_num_layers(model_id: str) -> int:
    """Decoder-layer count from the model config, without loading weights."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None and hasattr(cfg, "text_config"):
        n = cfg.text_config.num_hidden_layers
    if n is None:
        raise ValueError(f"could not read num_hidden_layers for {model_id}")
    return int(n)


def free_port() -> int:
    """Ask the OS for an unused TCP port."""
    import socket

    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def lan_ip() -> str:
    """This machine's primary LAN IP (best effort; falls back to 127.0.0.1)."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def env_info() -> dict:
    """Runtime identity a node advertises so the head can catch parity hazards.

    - `endian`: the fp16 tensor bytes on the wire are native byte order
      (`numpy.tobytes()`), so two nodes MUST agree — a big-endian ↔ little-endian
      pair would misread every hidden state. (ARM Macs and x86 PCs are both
      little-endian, so this is a guard, not a common failure.)
    - `torch` / `transformers`: version skew silently changes mask/RoPE internals
      and can break bitwise parity across machines — worth a warning.
    """
    import sys

    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "endian": sys.byteorder,
    }
