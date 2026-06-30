"""DRIFT orchestrator — embed / route / norm+head / sample + decode loop (spec §8).

Routing goes through an *injectable transport* (docs/01 cross-cutting decision):
the same decode loop runs over an in-process callable (M2) or a socket client
(M3+), so the network is the only variable between milestones.
"""

from __future__ import annotations

import argparse
import socket
import sys

import torch

from . import protocol
from .common import build_input_ids, load_config
from .engine_torch import TorchShardEngine


# ----------------------------------------------------------------- head model
class HeadModel:
    """Holds embed_tokens + final norm + lm_head (spec §8 v1 simplification).

    Uses the model's OWN modules, so model-specific embedding behavior (e.g.
    Gemma's scaled embedding) is applied automatically — nothing hardcoded.
    """

    def __init__(self, model_id: str, device: str, dtype: str):
        import transformers

        torch_dtype = {"float16": torch.float16, "float32": torch.float32,
                       "bfloat16": torch.bfloat16}[dtype]
        self.device = device
        self.lm = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
        self.lm.to(device).eval()
        self.inner = self.lm.model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    @torch.no_grad()
    def embed(self, input_ids):
        return self.inner.embed_tokens(input_ids.to(self.device))

    @torch.no_grad()
    def norm(self, hidden):
        return self.inner.norm(hidden)

    @torch.no_grad()
    def head(self, hidden):
        return self.lm.lm_head(hidden)


# ------------------------------------------------------------------ transports
class InProcessTransport:
    """M2: call engines directly, no socket."""

    def __init__(self, engines: dict):
        self.engines = engines

    def forward(self, name, session_id, hidden, position_ids, input_ids, mode):
        return self.engines[name].forward(session_id, hidden, position_ids, input_ids, mode)

    def reset(self, name, session_id):
        self.engines[name].reset(session_id)

    def ping(self, name):
        return {"ok": True, **self.engines[name].ping_info()}


class SocketTransport:
    """M3+: speak the §6 protocol over TCP."""

    def __init__(self, shards: list, dtype: str, device: str):
        self.shards = {s["name"]: s for s in shards}
        self.dtype = dtype
        self.device = device
        self.socks: dict = {}
        self.seq = 0

    def _sock(self, name):
        if name not in self.socks:
            s = self.shards[name]
            sk = socket.create_connection((s["host"], s["port"]))
            sk.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socks[name] = sk
        return self.socks[name]

    def forward(self, name, session_id, hidden, position_ids, input_ids, mode):
        self.seq += 1
        msg = {
            "type": mode,
            "session_id": session_id,
            "seq_id": self.seq,
            "shape": list(hidden.shape),
            "dtype": self.dtype,
            "position_ids": list(position_ids),
            "input_ids": list(input_ids),
            "tensor": protocol.tensor_to_bytes(hidden, self.dtype),
        }
        sk = self._sock(name)
        protocol.send_msg(sk, msg)
        reply = protocol.recv_msg(sk)
        if not reply.get("ok"):
            raise RuntimeError(f"shard {name} error: {reply.get('error')}")
        return protocol.bytes_to_tensor(reply["tensor"], reply["shape"], reply["dtype"], self.device)

    def reset(self, name, session_id):
        sk = self._sock(name)
        protocol.send_msg(sk, {"type": "reset", "session_id": session_id})
        protocol.recv_msg(sk)

    def ping(self, name):
        sk = self._sock(name)
        protocol.send_msg(sk, {"type": "ping"})
        return protocol.recv_msg(sk)


# ---------------------------------------------------------------- orchestrator
class Orchestrator:
    def __init__(self, head: HeadModel, transport, order: list, device: str):
        self.head = head
        self.transport = transport
        self.order = order  # shard names, in routing order
        self.device = device

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int, stop_on_eos: bool = False,
                 session_id: str = "s0") -> dict:
        tok = self.head.tokenizer
        input_ids = build_input_ids(tok, prompt).to(self.device)
        S = input_ids.shape[1]
        # Narrow EOS set (not all special ids, which would stop on benign tokens).
        eos: set[int] = set()
        if stop_on_eos:
            if tok.eos_token_id is not None:
                eos.add(int(tok.eos_token_id))
            gen_eos = getattr(getattr(self.head.lm, "generation_config", None), "eos_token_id", None)
            if isinstance(gen_eos, int):
                eos.add(gen_eos)
            elif isinstance(gen_eos, (list, tuple)):
                eos.update(int(x) for x in gen_eos)

        hidden = self.head.embed(input_ids)
        pos = list(range(S))
        ids_list = input_ids[0].tolist()
        for name in self.order:
            hidden = self.transport.forward(name, session_id, hidden, pos, ids_list, "prefill")
            hidden = hidden.to(self.device)

        logits = self.head.head(self.head.norm(hidden[:, -1:, :]))[:, -1, :]
        first_logits = logits[0].detach().float().cpu().numpy()
        next_id = int(torch.argmax(logits, dim=-1))
        generated = [next_id]
        p = S
        for _ in range(max_new_tokens - 1):
            if stop_on_eos and next_id in eos:
                break
            cur = torch.tensor([[next_id]], device=self.device)
            hidden = self.head.embed(cur)
            for name in self.order:
                hidden = self.transport.forward(name, session_id, hidden, [p], [next_id], "decode")
                hidden = hidden.to(self.device)
            logits = self.head.head(self.head.norm(hidden[:, -1:, :]))[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1))
            generated.append(next_id)
            p += 1

        for name in self.order:
            self.transport.reset(name, session_id)
        return {"token_ids": generated, "first_logits": first_logits, "text": tok.decode(generated)}


# ----------------------------------------------------------- builders / CLI
def build_inprocess(cfg: dict) -> Orchestrator:
    """M2: one shared model; engines reference its disjoint layer slices."""
    device = cfg.get("device", "cpu")
    head = HeadModel(cfg["model_id"], device, cfg.get("dtype", "float16"))
    engines = {}
    for s in cfg["shards"]:
        engines[s["name"]] = TorchShardEngine(
            model_id=cfg["model_id"], start_layer=s["start_layer"], end_layer=s["end_layer"],
            device=device, dtype=cfg.get("dtype", "float16"), name=s["name"], model=head.lm,
        )
        engines[s["name"]].load()
    order = [s["name"] for s in cfg["shards"]]
    return Orchestrator(head, InProcessTransport(engines), order, device)


def build_socket(cfg: dict, ports: list | None = None) -> Orchestrator:
    device = cfg.get("device", "cpu")
    head = HeadModel(cfg["model_id"], device, cfg.get("dtype", "float16"))
    shards = [dict(s) for s in cfg["shards"]]
    if ports:
        for s, p in zip(shards, ports):
            s["port"] = p
    transport = SocketTransport(shards, cfg.get("dtype", "float16"), device)
    order = [s["name"] for s in shards]
    return Orchestrator(head, transport, order, device)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DRIFT orchestrator")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ping", action="store_true", help="ping shards over TCP and exit")
    ap.add_argument("--ports", help="comma-separated localhost ports overriding config")
    ap.add_argument("--prompt")
    ap.add_argument("--max-new-tokens", type=int)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ports = [int(x) for x in args.ports.split(",")] if args.ports else None

    if args.ping:
        shards = [dict(s) for s in cfg["shards"]]
        if ports:
            for s, p in zip(shards, ports):
                s["port"] = p
        t = SocketTransport(shards, cfg.get("dtype", "float16"), cfg.get("device", "cpu"))
        ok = True
        for s in shards:
            try:
                reply = t.ping(s["name"])
                print(f"[ping] {s['name']} @ {s['host']}:{s['port']} -> {reply}", flush=True)
                ok = ok and reply.get("ok", False)
            except Exception as e:
                print(f"[ping] {s['name']} @ {s['host']}:{s['port']} -> FAILED: {e}", flush=True)
                ok = False
        print("M0 ping:", "PASS" if ok else "FAIL", flush=True)
        return 0 if ok else 1

    # generate over TCP (M3+)
    orch = build_socket(cfg, ports)
    prompt = args.prompt or cfg["generation"]["prompt"]
    n = args.max_new_tokens or cfg["generation"]["max_new_tokens"]
    out = orch.generate(prompt, n, stop_on_eos=True)
    print(out["text"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
