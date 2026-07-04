"""DRIFT orchestrator — embed / route / norm+head / sample + decode loop (spec §8).

Routing goes through an *injectable transport* (docs/01 cross-cutting decision):
the same decode loop runs over an in-process callable (M2) or a socket client
(M3+), so the network is the only variable between milestones.
"""

from __future__ import annotations

import argparse
import queue
import socket
import sys
import threading

import torch

from . import protocol
from .common import build_input_ids, lan_ip, load_config
from .engine_torch import TorchShardEngine


class NodeUnavailable(RuntimeError):
    """A shard node dropped and could not be reached again (M6 kill-node)."""


# ----------------------------------------------------------------- head model
class HeadModel:
    """Holds embed_tokens + final norm + lm_head (spec §8 v1 simplification).

    Uses the model's OWN modules, so model-specific embedding behavior (e.g.
    Gemma's scaled embedding) is applied automatically — nothing hardcoded.
    """

    def __init__(self, model_id: str, device: str, dtype: str, sliced: bool = False):
        import transformers

        self.device = device
        if sliced:
            # Real head (socket): materialize ONLY embed_tokens + norm (+ lm_head
            # if untied), never the decoder layers — the head holds ~15%, not 100%.
            from .loader import build_sliced

            cfg = transformers.AutoConfig.from_pretrained(model_id)
            tie = bool(getattr(cfg, "tie_word_embeddings", False))
            keep = ["model.embed_tokens.", "model.norm."]
            if not tie:
                keep.append("lm_head.")
            self.lm, _ = build_sliced(model_id, dtype, device, keep_prefixes=keep,
                                      need_rotary=False, tie=tie)
        else:
            # In-process (M2 baseline): full model, shared with the shard engines.
            torch_dtype = {"float16": torch.float16, "float32": torch.float32,
                           "bfloat16": torch.bfloat16}[dtype]
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

    def route(self, names, session_id, hidden, position_ids, input_ids, mode):
        """Run the whole ordered pipeline; the transport-agnostic entry point the
        decode loop calls (star loops here, chain streams peer-to-peer)."""
        for name in names:
            hidden = self.forward(name, session_id, hidden, position_ids, input_ids, mode)
        return hidden

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

    def _drop(self, name):
        """Forget a node's socket so the next call reconnects."""
        sk = self.socks.pop(name, None)
        if sk is not None:
            try:
                sk.close()
            except Exception:
                pass

    def _roundtrip(self, name, msg):
        """Send + receive one message, with a single reconnect for a transient
        drop. Raises NodeUnavailable if the node is truly gone (M6)."""
        for attempt in (1, 2):
            try:
                sk = self._sock(name)
                protocol.send_msg(sk, msg)
                return protocol.recv_msg(sk)
            except (ConnectionError, OSError) as e:
                self._drop(name)
                if attempt == 2:
                    raise NodeUnavailable(
                        f"node {name} dropped mid-run and did not come back: {e}") from e

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
        reply = self._roundtrip(name, msg)
        if not reply.get("ok"):
            raise RuntimeError(f"shard {name} error: {reply.get('error')}")
        return protocol.bytes_to_tensor(reply["tensor"], reply["shape"], reply["dtype"], self.device)

    def route(self, names, session_id, hidden, position_ids, input_ids, mode):
        """Star routing: every hop round-trips through the head (2N crossings/token)."""
        for name in names:
            hidden = self.forward(name, session_id, hidden, position_ids, input_ids, mode)
            hidden = hidden.to(self.device)
        return hidden

    def reset(self, name, session_id):
        # Cleanup path — a dropped node needs no reset, so never raise here.
        try:
            sk = self._sock(name)
            protocol.send_msg(sk, {"type": "reset", "session_id": session_id})
            protocol.recv_msg(sk)
        except (ConnectionError, OSError):
            self._drop(name)

    def configure(self, name, start_layer, end_layer, model_id, dtype, device=None):
        """Push a layer range to an unassigned (fungible) node."""
        sk = self._sock(name)
        protocol.send_msg(sk, {
            "type": "configure", "model_id": model_id, "dtype": dtype,
            "start_layer": start_layer, "end_layer": end_layer, "device": device,
        })
        reply = protocol.recv_msg(sk)
        if not reply.get("ok"):
            raise RuntimeError(f"configure {name} failed: {reply.get('error')}")
        return reply

    def ping(self, name):
        sk = self._sock(name)
        protocol.send_msg(sk, {"type": "ping"})
        return protocol.recv_msg(sk)


class ChainTransport(SocketTransport):
    """M7: peer-to-peer chain. The hidden state flows node→node→…→tail and the
    tail streams it to the head's `collect` sink — instead of star-routing every
    hop back through the head.

    Wins: **N+1** tensor crossings/token (vs 2N for the star) and — the point —
    the head's bandwidth is O(1) in the node count, not O(N). The head sends one
    tensor to the first node and receives one from the tail; the inner hops are
    node-to-node, so the head stops being the data-plane hub. `configure` / `ping`
    / `reset` (control + cleanup, off the hot path) are inherited from the star.

    The extra wire fields are additive and optional: `route` (the downstream
    [host,port] list) and `collect` (the head's sink [host,port]). A node without
    them behaves exactly like the star.
    """

    def __init__(self, shards: list, dtype: str, device: str, collect_host: str | None = None):
        super().__init__(shards, dtype, device)
        self._queues: dict[str, queue.Queue] = {}
        self._qlock = threading.Lock()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("", 0))
        srv.listen(16)
        self._collect_srv = srv
        # Address the tail dials to deliver the final hidden state. Defaults to
        # this host's LAN ip (reachable from a localhost node and a LAN node alike).
        self.collect_host = collect_host or lan_ip()
        self.collect_port = srv.getsockname()[1]
        threading.Thread(target=self._collect_accept, daemon=True).start()

    def _q(self, session_id: str) -> queue.Queue:
        with self._qlock:
            q = self._queues.get(session_id)
            if q is None:
                q = self._queues[session_id] = queue.Queue()
            return q

    def _collect_accept(self) -> None:
        while True:
            try:
                conn, _ = self._collect_srv.accept()
            except OSError:
                return
            threading.Thread(target=self._collect_reader, args=(conn,), daemon=True).start()

    def _collect_reader(self, conn: socket.socket) -> None:
        """Read final hidden states the tail streams here; hand each to its
        session's queue. One persistent connection carries a whole generation."""
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                try:
                    msg = protocol.recv_msg(conn)
                except (ConnectionError, OSError):
                    break
                self._q(msg.get("session_id", "s0")).put(msg)
                try:
                    protocol.send_msg(conn, {"ok": True})  # ack so the tail's send returns
                except (ConnectionError, OSError):
                    break
        finally:
            conn.close()

    def route(self, names, session_id, hidden, position_ids, input_ids, mode):
        self.seq += 1
        first = names[0]
        downstream = [[self.shards[n]["host"], self.shards[n]["port"]] for n in names[1:]]
        msg = {
            "type": mode, "session_id": session_id, "seq_id": self.seq,
            "shape": list(hidden.shape), "dtype": self.dtype,
            "position_ids": list(position_ids), "input_ids": list(input_ids),
            "tensor": protocol.tensor_to_bytes(hidden, self.dtype),
            "route": downstream,
            "collect": [self.collect_host, self.collect_port],
        }
        q = self._q(session_id)
        ack = self._roundtrip(first, msg)  # first node relays down the chain
        if not ack.get("ok"):
            raise NodeUnavailable(f"chain entry {first} error: {ack.get('error')}")
        try:
            reply = q.get(timeout=120)
        except queue.Empty as e:
            raise NodeUnavailable(
                "chain tail never reached the collect sink — a node dropped mid-chain") from e
        return protocol.bytes_to_tensor(reply["tensor"], reply["shape"], reply["dtype"], self.device)


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
        hidden = self.transport.route(self.order, session_id, hidden, pos, ids_list, "prefill").to(self.device)

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
            hidden = self.transport.route(self.order, session_id, hidden, [p], [next_id], "decode").to(self.device)
            logits = self.head.head(self.head.norm(hidden[:, -1:, :]))[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1))
            generated.append(next_id)
            p += 1

        for name in self.order:
            self.transport.reset(name, session_id)
        return {"token_ids": generated, "first_logits": first_logits, "text": tok.decode(generated)}

    @torch.no_grad()
    def generate_stream(self, prompt: str, max_new_tokens: int, stop_on_eos: bool = True,
                        session_id: str = "s0"):
        """Same decode loop as generate(), but *yields* decoded text as it lands.

        Yields incremental UTF-8-safe text deltas (decode the full id list each
        step and emit the new suffix, so multibyte tokens never break). EOS is
        not emitted. The result is identical to generate() — this only changes
        when the tokens reach the caller.
        """
        tok = self.head.tokenizer
        input_ids = build_input_ids(tok, prompt).to(self.device)
        S = input_ids.shape[1]
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
        hidden = self.transport.route(self.order, session_id, hidden, pos, ids_list, "prefill").to(self.device)
        logits = self.head.head(self.head.norm(hidden[:, -1:, :]))[:, -1, :]
        next_id = int(torch.argmax(logits, dim=-1))

        generated: list[int] = []
        prev = ""
        p = S
        for _ in range(max_new_tokens):
            if stop_on_eos and next_id in eos:
                break
            generated.append(next_id)
            text = tok.decode(generated)
            if len(text) > len(prev):
                yield text[len(prev):]
                prev = text
            cur = torch.tensor([[next_id]], device=self.device)
            hidden = self.head.embed(cur)
            hidden = self.transport.route(self.order, session_id, hidden, [p], [next_id], "decode").to(self.device)
            logits = self.head.head(self.head.norm(hidden[:, -1:, :]))[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1))
            p += 1

        for name in self.order:
            self.transport.reset(name, session_id)


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
    head = HeadModel(cfg["model_id"], device, cfg.get("dtype", "float16"), sliced=True)
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
