"""Shared helpers: config loading and identical tokenization.

The reference oracle (reference.py) and the split path (orchestrator.py) MUST
tokenize a prompt identically, or parity is meaningless. Both call
`build_input_ids` here.
"""

from __future__ import annotations

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_input_ids(tokenizer, prompt: str):
    """Deterministic input_ids [1, S] via the model's chat template.

    Falls back to plain encoding if the tokenizer has no chat template.
    """
    if getattr(tokenizer, "chat_template", None):
        out = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        ids = out["input_ids"]
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    return ids
