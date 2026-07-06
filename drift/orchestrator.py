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

from . import crypto, protocol, receipts
from .common import build_input_ids, lan_ip, load_config
from .engine_torch import TorchShardEngine
from .receipts import ReceiptVerifier


class NodeUnavailable(RuntimeError):
    """A shard node dropped and could not be reached again (M6 kill-node)."""


# ----------------------------------------------------------------- head model
class HeadModel:
    """Holds embed_tokens + final norm + lm_head (spec §8 v1 simplification).

    Uses the model's OWN modules, so model-specific embedding behavior (e.g.
    Gemma's scaled embedding) is applied automatically — nothing hardcoded.
    """

    def __init__(self, model_id: str, device: str, dtype: str, sliced: bool = False,
                 thin: bool = False):
        import transformers

        self.device = device
        self.thin = thin
        if thin:
            # M10 thin head: hold ONLY the tokenizer — zero model weights. embed
            # moves to the first node, norm+lm_head+argmax to the last. The head
            # sends token ids in and gets a token id out.
            self.lm = None
            self.inner = None
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
            try:
                self.gen_cfg = transformers.GenerationConfig.from_pretrained(model_id)
            except Exception:
                self.gen_cfg = None
            return
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

    def close(self):
        pass


class SocketTransport:
    """M3+: speak the §6 protocol over TCP."""

    def __init__(self, shards: list, dtype: str, device: str):
        self.shards = {s["name"]: s for s in shards}
        self.dtype = dtype
        self.device = device
        self.wire_dtype = dtype   # M14: "int8" halves wire bytes (lossy); else = compute dtype
        self.socks: dict = {}
        self.seq = 0
        # M11: per-token receipts + head anchors, read by the orchestrator's verifier.
        self.last_receipts: list = []
        self.last_anchor_in = None
        self.last_anchor_out = None
        self._last_receipt = None
        self._last_recv_hash = None

    def _sock(self, name):
        """A cached channel to a node (encrypted when a network key is set)."""
        if name not in self.socks:
            s = self.shards[name]
            self.socks[name] = crypto.dial(s["host"], s["port"])
        return self.socks[name]

    def _drop(self, name):
        """Forget a node's channel so the next call reconnects (+ re-handshakes)."""
        ch = self.socks.pop(name, None)
        if ch is not None:
            ch.close()

    def _roundtrip(self, name, msg):
        """Send + receive one message, with a single reconnect for a transient
        drop. Raises NodeUnavailable if the node is truly gone (M6)."""
        for attempt in (1, 2):
            try:
                ch = self._sock(name)
                ch.send(msg)
                return ch.recv()
            except (ConnectionError, OSError) as e:
                self._drop(name)
                if attempt == 2:
                    raise NodeUnavailable(
                        f"node {name} dropped mid-run and did not come back: {e}") from e

    def forward(self, name, session_id, hidden, position_ids, input_ids, mode):
        self.seq += 1
        tb, scale = protocol.tensor_to_wire(hidden, self.wire_dtype)
        msg = {
            "type": mode,
            "session_id": session_id,
            "seq_id": self.seq,
            "shape": list(hidden.shape),
            "dtype": self.wire_dtype,
            "scale": scale,
            "position_ids": list(position_ids),
            "input_ids": list(input_ids),
            "tensor": tb,
        }
        reply = self._roundtrip(name, msg)
        if not reply.get("ok"):
            raise RuntimeError(f"shard {name} error: {reply.get('error')}")
        self._last_receipt = reply.get("receipt")
        self._last_recv_hash = receipts.hash_bytes(reply["tensor"]) if "tensor" in reply else None
        return protocol.wire_to_tensor(reply["tensor"], reply["shape"], reply["dtype"],
                                       self.device, reply.get("scale"))

    def route(self, names, session_id, hidden, position_ids, input_ids, mode):
        """Star routing: every hop round-trips through the head (2N crossings/token)."""
        self.last_receipts = []
        self.last_anchor_in = receipts.hash_bytes(protocol.tensor_to_bytes(hidden, self.dtype))
        for name in names:
            hidden = self.forward(name, session_id, hidden, position_ids, input_ids, mode)
            hidden = hidden.to(self.device)
            if self._last_receipt is not None:
                self.last_receipts.append(self._last_receipt)
        self.last_anchor_out = self._last_recv_hash
        return hidden

    def reset(self, name, session_id):
        # Cleanup path — a dropped node needs no reset, so never raise here.
        try:
            ch = self._sock(name)
            ch.send({"type": "reset", "session_id": session_id})
            ch.recv()
        except (ConnectionError, OSError):
            self._drop(name)

    def configure(self, name, start_layer, end_layer, model_id, dtype, device=None,
                  embed_duty=False, head_duty=False):
        """Push a layer range (and thin-head edge duties) to an unassigned node."""
        ch = self._sock(name)
        ch.send({
            "type": "configure", "model_id": model_id, "dtype": dtype,
            "start_layer": start_layer, "end_layer": end_layer, "device": device,
            "embed_duty": embed_duty, "head_duty": head_duty,
        })
        reply = ch.recv()
        if not reply.get("ok"):
            raise RuntimeError(f"configure {name} failed: {reply.get('error')}")
        return reply

    def ping(self, name):
        ch = self._sock(name)
        ch.send({"type": "ping"})
        return ch.recv()

    def close(self):
        for ch in list(self.socks.values()):
            try:
                ch.close()
            except Exception:
                pass
        self.socks.clear()


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
        session's queue. One persistent connection carries a whole generation.
        The tail dials in as a client, so we complete the server handshake first
        (encrypted when a network key is set)."""
        try:
            ch = crypto.accept_wrap(conn)  # server side of the secure handshake
        except (ConnectionError, OSError, ValueError):
            conn.close()
            return
        try:
            while True:
                try:
                    msg = ch.recv()
                except (ConnectionError, OSError, ValueError):
                    break
                self._q(msg.get("session_id", "s0")).put(msg)
                try:
                    ch.send({"ok": True})  # ack so the tail's send returns
                except (ConnectionError, OSError):
                    break
        finally:
            ch.close()

    def route(self, names, session_id, hidden, position_ids, input_ids, mode):
        self.seq += 1
        first = names[0]
        downstream = [[self.shards[n]["host"], self.shards[n]["port"]] for n in names[1:]]
        tb, scale = protocol.tensor_to_wire(hidden, self.wire_dtype)
        msg = {
            "type": mode, "session_id": session_id, "seq_id": self.seq,
            "shape": list(hidden.shape), "dtype": self.wire_dtype, "scale": scale,
            "position_ids": list(position_ids), "input_ids": list(input_ids),
            "tensor": tb,
            "route": downstream,
            "collect": [self.collect_host, self.collect_port],
        }
        self.last_anchor_in = receipts.hash_bytes(msg["tensor"])
        q = self._q(session_id)
        ack = self._roundtrip(first, msg)  # first node relays down the chain
        if not ack.get("ok"):
            raise NodeUnavailable(f"chain entry {first} error: {ack.get('error')}")
        try:
            reply = q.get(timeout=120)
        except queue.Empty as e:
            raise NodeUnavailable(
                "chain tail never reached the collect sink — a node dropped mid-chain") from e
        self.last_receipts = reply.get("receipts", [])
        self.last_anchor_out = receipts.hash_bytes(reply["tensor"])
        return protocol.wire_to_tensor(reply["tensor"], reply["shape"], reply["dtype"],
                                       self.device, reply.get("scale"))

    def route_token(self, names, session_id, input_ids, position_ids, mode) -> int:
        """Thin-head chain (M10): the head sends only token ids in and gets a token
        id back. The first node embeds, the tail norms+heads+argmaxes — the head
        does no tensor math and holds no model weights. No tensor crosses its
        boundary, just ints."""
        self.seq += 1
        first = names[0]
        downstream = [[self.shards[n]["host"], self.shards[n]["port"]] for n in names[1:]]
        msg = {
            "type": mode, "session_id": session_id, "seq_id": self.seq,
            "position_ids": list(position_ids), "input_ids": list(input_ids),
            "embed": True,  # the entry node embeds these ids (no tensor sent)
            "dtype": self.wire_dtype,  # inter-node tensors use this wire dtype
            "route": downstream,
            "collect": [self.collect_host, self.collect_port],
        }
        self.last_anchor_in = receipts.hash_ints(input_ids)
        q = self._q(session_id)
        ack = self._roundtrip(first, msg)
        if not ack.get("ok"):
            raise NodeUnavailable(f"chain entry {first} error: {ack.get('error')}")
        try:
            reply = q.get(timeout=120)
        except queue.Empty as e:
            raise NodeUnavailable(
                "chain tail never reached the collect sink — a node dropped mid-chain") from e
        self.last_receipts = reply.get("receipts", [])
        self.last_anchor_out = receipts.hash_ints([int(reply["token"])])
        return int(reply["token"])

    def close(self):
        try:
            self._collect_srv.close()  # unblocks the accept loop (OSError → returns)
        except Exception:
            pass
        super().close()


# ---------------------------------------------------------------- orchestrator
class Orchestrator:
    def __init__(self, head: HeadModel, transport, order: list, device: str):
        self.head = head
        self.transport = transport
        self.order = order  # shard names, in routing order
        self.device = device
        self.thin = getattr(head, "thin", False)  # M10: head holds no model weights
        # M9 failover: a cluster that can re-split over survivors, or None (no recovery).
        self.cluster = None
        self.recoveries = 0   # how many times a mid-run drop was recovered
        self.progress = 0     # tokens produced so far (lets a watcher time a kill)
        # M11: verify each token's signed receipts against the head's anchors.
        self.verify = False
        self.verifier: ReceiptVerifier | None = None
        self.n_layers = None
        self.journal = None   # M13: append verified receipts here for the ledger

    def _check_receipts(self) -> None:
        if not self.verify or self.verifier is None:
            return
        t = self.transport
        r = getattr(t, "last_receipts", None)
        if r:
            self.verifier.check(r, getattr(t, "last_anchor_in", None),
                                getattr(t, "last_anchor_out", None), self.n_layers)
            if self.journal:
                try:
                    receipts.append_journal(self.journal, r)
                except OSError:
                    pass

    def _eos_set(self, stop_on_eos: bool) -> set[int]:
        """The narrow EOS id set (not all special ids, which stop on benign tokens)."""
        eos: set[int] = set()
        if not stop_on_eos:
            return eos
        tok = self.head.tokenizer
        if tok.eos_token_id is not None:
            eos.add(int(tok.eos_token_id))
        # In thin mode the head has no lm; fall back to its GenerationConfig.
        gcfg = getattr(self.head.lm, "generation_config", None) or getattr(self.head, "gen_cfg", None)
        gen_eos = getattr(gcfg, "eos_token_id", None)
        if isinstance(gen_eos, int):
            eos.add(gen_eos)
        elif isinstance(gen_eos, (list, tuple)):
            eos.update(int(x) for x in gen_eos)
        return eos

    def _prefill(self, session_id, seq_ids):
        """Feed the whole sequence; return (next_id, first_logits_or_None). In thin
        mode the pipeline embeds + heads, so the head just exchanges token ids."""
        if self.thin:
            nid = self.transport.route_token(self.order, session_id, seq_ids,
                                             list(range(len(seq_ids))), "prefill")
            self._check_receipts()
            return nid, None
        hidden = self.head.embed(torch.tensor([seq_ids], device=self.device))
        hidden = self.transport.route(self.order, session_id, hidden,
                                      list(range(len(seq_ids))), seq_ids, "prefill").to(self.device)
        self._check_receipts()
        logits = self.head.head(self.head.norm(hidden[:, -1:, :]))[:, -1, :]
        return int(torch.argmax(logits, dim=-1)), logits[0].detach().float().cpu().numpy()

    def _decode(self, session_id, tok_id, pos, return_logits: bool = False):
        """Feed one token at absolute position `pos`; return the next token id."""
        if self.thin:
            nid = self.transport.route_token(self.order, session_id, [tok_id], [pos], "decode")
            self._check_receipts()
            if return_logits:
                raise ValueError("sampling is not available in thin-head mode")
            return nid
        hidden = self.head.embed(torch.tensor([[tok_id]], device=self.device))
        hidden = self.transport.route(self.order, session_id, hidden, [pos], [tok_id],
                                      "decode").to(self.device)
        self._check_receipts()
        logits = self.head.head(self.head.norm(hidden[:, -1:, :]))[:, -1, :]
        if return_logits:
            return int(torch.argmax(logits, dim=-1)), logits[0].detach().float().cpu()
        return int(torch.argmax(logits, dim=-1))

    def _sampling_enabled(self, opts: dict | None) -> bool:
        if not opts:
            return False
        temperature = 0.0 if opts.get("temperature") is None else float(opts["temperature"])
        top_p = 1.0 if opts.get("top_p") is None else float(opts["top_p"])
        top_k = 0 if opts.get("top_k") is None else int(opts["top_k"])
        min_p = 0.0 if opts.get("min_p") is None else float(opts["min_p"])
        presence = 0.0 if opts.get("presence_penalty") is None \
            else float(opts["presence_penalty"])
        frequency = 0.0 if opts.get("frequency_penalty") is None \
            else float(opts["frequency_penalty"])
        repetition = 1.0 if opts.get("repetition_penalty") is None \
            else float(opts["repetition_penalty"])
        return (
            temperature > 0.0
            or top_p < 1.0
            or top_k > 0
            or min_p > 0.0
            or presence != 0.0
            or frequency != 0.0
            or repetition != 1.0
        )

    def _pick_token(self, logits, history_ids: list[int], opts: dict | None,
                    step: int) -> int:
        """Greedy by default; optional OpenAI/llama.cpp-style sampling controls.

        The default branch is exactly the old argmax path. Sampling runs on CPU so
        a seeded torch.Generator gives deterministic draws independent of MPS/CUDA.
        """
        if not self._sampling_enabled(opts):
            return int(torch.argmax(logits, dim=-1))

        opts = opts or {}
        temperature = 1.0 if opts.get("temperature") is None else float(opts["temperature"])

        scores = logits.detach().float().cpu().clone()
        if history_ids:
            counts: dict[int, int] = {}
            for tid in history_ids:
                counts[int(tid)] = counts.get(int(tid), 0) + 1
            presence = 0.0 if opts.get("presence_penalty") is None \
                else float(opts["presence_penalty"])
            frequency = 0.0 if opts.get("frequency_penalty") is None \
                else float(opts["frequency_penalty"])
            repetition = 1.0 if opts.get("repetition_penalty") is None \
                else float(opts["repetition_penalty"])
            for tid, count in counts.items():
                if 0 <= tid < scores.numel():
                    scores[tid] -= presence + frequency * count
                    if repetition != 1.0:
                        scores[tid] = scores[tid] / repetition if scores[tid] > 0 \
                            else scores[tid] * repetition

        if temperature <= 0:
            return int(torch.argmax(scores, dim=-1))
        scores = scores / temperature
        top_k = 0 if opts.get("top_k") is None else int(opts["top_k"])
        if top_k > 0 and top_k < scores.numel():
            keep = torch.topk(scores, top_k).indices
            mask = torch.full_like(scores, float("-inf"))
            mask[keep] = scores[keep]
            scores = mask

        probs = torch.softmax(scores, dim=-1)
        min_p = 0.0 if opts.get("min_p") is None else float(opts["min_p"])
        if min_p > 0:
            cutoff = torch.max(probs) * min_p
            probs = torch.where(probs >= cutoff, probs, torch.zeros_like(probs))

        top_p = 1.0 if opts.get("top_p") is None else float(opts["top_p"])
        if 0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            sorted_probs = torch.where(remove, torch.zeros_like(sorted_probs), sorted_probs)
            filtered = torch.zeros_like(probs)
            filtered[sorted_idx] = sorted_probs
            probs = filtered

        total = torch.sum(probs)
        if not torch.isfinite(total) or float(total) <= 0:
            return int(torch.argmax(logits, dim=-1))
        probs = probs / total
        generator = None
        if opts.get("seed") is not None:
            generator = torch.Generator()
            generator.manual_seed(int(opts["seed"]) + int(step))
        return int(torch.multinomial(probs, 1, generator=generator))

    def _recover(self, session_id: str) -> None:
        """A node dropped mid-run: re-split over the survivors (+ spares), so the
        caller can re-prefill the sequence-so-far and continue. Raises if no
        cluster is attached or nothing survives."""
        if self.cluster is None:
            raise NodeUnavailable("a node dropped and no cluster is attached for recovery")
        old = self.transport
        self.transport, self.order = self.cluster.rebuild()
        self.recoveries += 1
        try:
            old.close()
        except Exception:
            pass
        for name in self.order:  # clear any stale KV; a fresh re-prefill follows
            try:
                self.transport.reset(name, session_id)
            except Exception:
                pass

    @torch.no_grad()
    def generate(self, prompt: str | list[dict], max_new_tokens: int, stop_on_eos: bool = False,
                 session_id: str = "s0", generation_options: dict | None = None) -> dict:
        tok = self.head.tokenizer
        input_ids = build_input_ids(tok, prompt).to(self.device)
        prompt_ids = input_ids[0].tolist()
        eos = self._eos_set(stop_on_eos)
        sampling = self._sampling_enabled(generation_options)
        if sampling and self.thin:
            raise ValueError("sampling is not available in thin-head mode")

        generated: list[int] = []
        first_logits = None
        while True:  # (re)start on a mid-run node drop (M9)
            try:
                # (Re)prefill the whole sequence so far. Fresh start → seq = prompt;
                # after a drop → prompt + tokens-generated-so-far, which rebuilds every
                # survivor's KV. Greedy decoding is deterministic over a fixed prefix, so
                # the prefill's argmax is exactly the next token an uninterrupted run would
                # emit — the recovered continuation is bitwise-identical.
                seq = prompt_ids + generated
                next_id, logits0 = self._prefill(session_id, seq)
                if first_logits is None:
                    first_logits = logits0
                if sampling and logits0 is not None:
                    next_id = self._pick_token(torch.tensor(logits0), seq,
                                               generation_options, len(generated))
                p = len(seq)
                while len(generated) < max_new_tokens:
                    generated.append(next_id)
                    self.progress = len(generated)
                    if (stop_on_eos and next_id in eos) or len(generated) >= max_new_tokens:
                        break
                    if sampling:
                        _, logits = self._decode(session_id, next_id, p, return_logits=True)
                        next_id = self._pick_token(
                            logits, prompt_ids + generated, generation_options, len(generated)
                        )
                    else:
                        next_id = self._decode(session_id, next_id, p)
                    p += 1
                break  # finished with no unrecovered drop
            except NodeUnavailable:
                self._recover(session_id)  # re-split over survivors, then the loop re-prefills

        for name in self.order:
            self.transport.reset(name, session_id)
        return {"token_ids": generated, "first_logits": first_logits, "text": tok.decode(generated)}

    @torch.no_grad()
    def generate_stream(self, prompt: str | list[dict], max_new_tokens: int,
                        stop_on_eos: bool = True,
                        session_id: str = "s0", generation_options: dict | None = None):
        """Streaming twin of generate(): the SAME _prefill/_decode machinery —
        thin-aware, receipt-verifying (M11), journaling (M13), and failover-
        recovering (M9) — but *yields* decoded text deltas as tokens land.

        Yields incremental UTF-8-safe text deltas (decode the full id list each
        step and emit the new suffix, so multibyte tokens never break). EOS is
        not emitted. The emitted visible text is identical to generate(); this
        only changes *when* tokens reach the caller.

        Previously this was a second, drifted copy of the decode loop that did no
        receipt verification, no journaling, no failover, and crashed in thin
        mode — even though `drift run`/`drift up` use ONLY this path. Routing it
        through _prefill/_decode (which the gates already prove bitwise) fixes all
        four at once.
        """
        tok = self.head.tokenizer
        input_ids = build_input_ids(tok, prompt).to(self.device)
        prompt_ids = input_ids[0].tolist()
        eos = self._eos_set(stop_on_eos)
        sampling = self._sampling_enabled(generation_options)
        if sampling and self.thin:
            raise ValueError("sampling is not available in thin-head mode")

        generated: list[int] = []
        prev = ""
        try:
            while True:  # (re)start on a mid-run node drop (M9)
                try:
                    # (Re)prefill the whole sequence so far. Fresh start → the
                    # prompt; after a drop → prompt + tokens-so-far, which rebuilds
                    # every survivor's KV. Greedy is deterministic over a fixed
                    # prefix, so the resumed continuation is bitwise-identical.
                    seq = prompt_ids + generated
                    next_id, logits0 = self._prefill(session_id, seq)
                    if sampling and logits0 is not None:
                        next_id = self._pick_token(torch.tensor(logits0), seq,
                                                   generation_options, len(generated))
                    p = len(seq)
                    while len(generated) < max_new_tokens:
                        if stop_on_eos and next_id in eos:
                            break
                        generated.append(next_id)
                        self.progress = len(generated)
                        text = tok.decode(generated)
                        if len(text) > len(prev):
                            yield text[len(prev):]
                            prev = text
                        if len(generated) >= max_new_tokens:
                            break
                        if sampling:
                            _, logits = self._decode(session_id, next_id, p, return_logits=True)
                            next_id = self._pick_token(
                                logits, prompt_ids + generated,
                                generation_options, len(generated)
                            )
                        else:
                            next_id = self._decode(session_id, next_id, p)
                        p += 1
                    break  # finished with no unrecovered drop
                except NodeUnavailable:
                    self._recover(session_id)  # re-split over survivors, then re-prefill
        finally:
            for name in self.order:
                try:
                    self.transport.reset(name, session_id)
                except Exception:
                    pass


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
