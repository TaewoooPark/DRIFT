"""Signed per-hop receipts + live verification (M11).

The M6/M4 spot-check (drift/verify.py) challenges a node with a *fixed* input —
a node honest only on the challenge escapes it. M11 binds verification to the
**real traffic**: every hop signs a receipt over what it actually consumed and
produced, and the head checks the chain of receipts on each generated token.

A receipt is
    { node: <ed25519 pubkey hex>, session, seq, mode, start, end,
      in_hash: sha256(input), out_hash: sha256(output), sig }
signed with the node's identity key. The head checks, per token:

  1. **signatures** — each receipt is signed by the node that claims it;
  2. **adjacency** — node i's out_hash == node i+1's in_hash (they handled the
     same bytes), so a hop that corrupts or swaps the stream is caught;
  3. **anchors** — the first hop's in_hash matches what the head sent, and the
     last hop's out_hash matches what the head received (the chain is pinned at
     both ends, so it can't be truncated or spliced);
  4. **coverage** — the layer ranges are contiguous and start at 0.

Violations mark the implicated node(s) SUSPECT in a local reputation table.

What this catches on live traffic: wire corruption, dropped/reordered/duplicated
hops, forged or unsigned receipts, a node lying about what it computed vs. sent.
What it does NOT catch (a node that *consistently* miscomputes and signs the
result) is the job of the recompute audit — see drift/verify.py, which now checks
against the same committed receipts. Redundant N-of-M execution is future work.
"""

from __future__ import annotations

import hashlib

import msgpack

from . import crypto

# Fields covered by the signature, in a fixed order → deterministic to sign/verify.
_SIGNED = ("node", "session", "seq", "mode", "start", "end", "in_hash", "out_hash")


def hash_bytes(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def hash_ints(ints) -> bytes:
    """Hash a list of ints (token ids / a single token) canonically."""
    return hashlib.sha256(msgpack.packb([int(x) for x in ints], use_bin_type=True)).digest()


def _canon(r: dict) -> bytes:
    return msgpack.packb([r[k] for k in _SIGNED], use_bin_type=True)


def make_receipt(sk, pub_hex: str, session: str, seq: int, mode: str,
                 start: int, end: int, in_hash: bytes, out_hash: bytes) -> dict:
    body = {"node": pub_hex, "session": session, "seq": int(seq), "mode": mode,
            "start": int(start), "end": int(end), "in_hash": in_hash, "out_hash": out_hash}
    body["sig"] = crypto.sign(sk, _canon(body))
    return body


def verify_receipt(r: dict) -> bool:
    try:
        return crypto.verify_sig(r["node"], r["sig"], _canon(r))
    except (KeyError, TypeError):
        return False


class ReceiptVerifier:
    """Checks a chain of receipts per token; accrues a local reputation table."""

    def __init__(self):
        self.reputation: dict[str, str] = {}   # pubkey hex -> "ok" | "suspect"
        self.checked = 0
        self.violations: list[tuple[str, str]] = []  # (node, reason) across the run

    def _flag(self, node: str, reason: str) -> None:
        self.reputation[node] = "suspect"
        self.violations.append((node, reason))

    def suspects(self) -> list[str]:
        return [n for n, v in self.reputation.items() if v == "suspect"]

    def check(self, receipts: list, anchor_in: bytes | None, anchor_out: bytes | None,
              n_layers: int | None = None) -> bool:
        """Verify one token's worth of receipts (in route order). Returns True if
        clean; records violations + reputation otherwise."""
        self.checked += 1
        if not receipts:
            return True
        ok = True
        for r in receipts:                                   # 1) signatures
            self.reputation.setdefault(r.get("node", "?"), "ok")
            if not verify_receipt(r):
                self._flag(r.get("node", "?"), "invalid signature"); ok = False
        for i in range(len(receipts) - 1):                   # 2) adjacency
            if receipts[i]["out_hash"] != receipts[i + 1]["in_hash"]:
                self._flag(receipts[i]["node"], "output != next hop's input")
                self._flag(receipts[i + 1]["node"], "input != prev hop's output")
                ok = False
        if anchor_in is not None and receipts[0]["in_hash"] != anchor_in:   # 3) anchors
            self._flag(receipts[0]["node"], "entry input != head anchor"); ok = False
        if anchor_out is not None and receipts[-1]["out_hash"] != anchor_out:
            self._flag(receipts[-1]["node"], "exit output != head anchor"); ok = False
        if n_layers is not None:                             # 4) coverage
            if receipts[0]["start"] != 0 or receipts[-1]["end"] != n_layers:
                self._flag(receipts[-1]["node"], "layer coverage not [0, n)"); ok = False
            for i in range(len(receipts) - 1):
                if receipts[i]["end"] != receipts[i + 1]["start"]:
                    self._flag(receipts[i]["node"], "layer gap/overlap"); ok = False
        return ok
