"""Gossip membership — the network learns its own members (M12).

Zero-config LAN discovery (drift/discovery.py, mDNS) only sees a local subnet.
M12 lets a node **join from anywhere** by pointing at one seed and gossiping: it
announces itself and pulls the seed's peer list, then periodically re-exchanges
with known peers (anti-entropy), so membership converges across the whole network
— including nodes reached only over a tunnel.

Each peer entry is signed by that peer's own Ed25519 identity over
(pubkey, host, port, device, ts), so a node vouches for its own address and no
one can forge an entry for another's key. Merges keep the highest `ts` per
pubkey. (A v1: `ts` is peer-chosen and addresses are self-asserted — good enough
to bootstrap a cooperative network, not a defense against a determined Sybil. A
membership authority / proof-of-work admission is future work.)
"""

from __future__ import annotations

import time

import msgpack

from . import crypto

_SIGNED = ("pubkey", "host", "port", "device", "ts")


def _canon(e: dict) -> bytes:
    return msgpack.packb([e[k] for k in _SIGNED], use_bin_type=True)


def self_entry(sk, pubkey: str, host: str, port: int, device: str, ts: float | None = None) -> dict:
    e = {"pubkey": pubkey, "host": host, "port": int(port), "device": device,
         "ts": float(ts if ts is not None else time.time())}
    e["sig"] = crypto.sign(sk, _canon(e))
    return e


def verify_entry(e: dict) -> bool:
    try:
        return crypto.verify_sig(e["pubkey"], e["sig"], _canon(e))
    except (KeyError, TypeError):
        return False


class PeerTable:
    """Signed peer entries keyed by pubkey; last-writer-wins by `ts`."""

    def __init__(self):
        self.peers: dict[str, dict] = {}

    def add(self, e: dict) -> bool:
        if not verify_entry(e):
            return False
        cur = self.peers.get(e["pubkey"])
        if cur is None or e["ts"] >= cur["ts"]:
            self.peers[e["pubkey"]] = e
            return cur is None
        return False

    def merge(self, entries) -> int:
        """Fold in a peer list; return how many *new* pubkeys were learned."""
        return sum(1 for e in (entries or []) if self.add(e))

    def list(self) -> list[dict]:
        return list(self.peers.values())

    def endpoints(self, exclude: str | None = None) -> list[dict]:
        """Members as orchestrator endpoints, in a deterministic (pubkey) order."""
        out = []
        for i, e in enumerate(sorted(self.peers.values(), key=lambda x: x["pubkey"])):
            if exclude and e["pubkey"] == exclude:
                continue
            out.append({"name": f"m{i}", "host": e["host"], "port": e["port"],
                        "device": e.get("device"), "pubkey": e["pubkey"]})
        return out


def fetch_peers(host, port) -> list[dict]:
    """Pull a node's peer list (used by `drift run --expand`)."""
    ch = crypto.dial(host, int(port))
    try:
        ch.send({"type": "peers_get"})
        return ch.recv().get("peers", [])
    finally:
        ch.close()


def gossip_once(host, port, my_peers: list[dict]) -> list[dict]:
    """Announce our peer list to one node and merge its reply (anti-entropy)."""
    ch = crypto.dial(host, int(port))
    try:
        ch.send({"type": "peer_announce", "peers": my_peers})
        return ch.recv().get("peers", [])
    finally:
        ch.close()
