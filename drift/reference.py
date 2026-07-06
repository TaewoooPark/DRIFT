"""M1 — single-machine reference oracle (spec §9 M1).

Loads the full model normally and greedily generates a fixed number of tokens
from a fixed prompt, saving the token-id sequence + first-step logits. This is
the parity ground truth M2/M3 must reproduce bitwise.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .common import build_input_ids, load_config


def run_reference(cfg: dict, device: str | None = None, out_path: str = "reference_out.npz") -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    device = device or cfg.get("device", "cpu")
    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[cfg.get("dtype", "float16")]
    model_id = cfg["model_id"]
    n = cfg["generation"]["max_new_tokens"]
    prompt = cfg["generation"]["prompt"]

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_id)

    n_layers = model.config.num_hidden_layers
    print(f"[M1] model={model_id} layers={n_layers} device={device} dtype={cfg.get('dtype')}", flush=True)

    input_ids = build_input_ids(tok, prompt).to(device)
    cache = DynamicCache(config=model.config)

    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1, :]
        next_id = int(torch.argmax(logits, dim=-1))
        first_logits = logits[0].detach().float().cpu().numpy()
        generated = [next_id]
        for _ in range(n - 1):
            cur = torch.tensor([[next_id]], device=device)
            out = model(input_ids=cur, past_key_values=cache, use_cache=True)
            logits = out.logits[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1))
            generated.append(next_id)

    np.savez(
        out_path,
        token_ids=np.array(generated, dtype=np.int64),
        first_logits=first_logits,
        n_layers=np.int64(n_layers),
        prompt_len=np.int64(input_ids.shape[1]),
    )
    print(f"[M1] saved {out_path}: {len(generated)} tokens; first 10 ids = {generated[:10]}", flush=True)
    print(f"[M1] text: {tok.decode(generated)!r}", flush=True)
    return {"token_ids": generated, "first_logits": first_logits, "n_layers": n_layers}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DRIFT M1 reference oracle")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reference_out.npz")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_reference(cfg, device=args.device, out_path=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
