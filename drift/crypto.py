"""Wire security — network membership (PSK) + confidentiality (AEAD) + identity.

Two independent secrets, both under ``~/.config/drift`` (override with env):

  * **network key** — a 32-byte pre-shared key (PSK) shared by every node + head
    on one DRIFT network. It authenticates *membership*: a dialer that doesn't
    hold it cannot derive the channel key, so its first frame fails to decrypt
    and the connection is dropped. This is what closes the open-compute hole —
    ``drift node --tunnel`` refuses to run without one.
  * **node identity** — an Ed25519 keypair, one per machine. Not used to set up
    the channel; it signs the per-hop receipts in M11 (trustless verification).

Keyed connections run a lightweight authenticated handshake before any DRIFT
message:

    client ──{eph_pub_c, nonce_c}──▶ server          (plaintext; public values)
    client ◀──{eph_pub_s, nonce_s}── server
    both:  shared = X25519(eph)                       (ephemeral → forward secrecy)
           k_c2s, k_s2c = HKDF-SHA256(shared ‖ PSK, salt=nonce_c‖nonce_s)

then every frame is ChaCha20-Poly1305 with a per-direction counter nonce. Mixing
the PSK into the HKDF input is what authenticates membership (an attacker without
it derives a different key). This is a pragmatic v1 (Noise-NNpsk0-shaped), not a
formally verified protocol — it buys confidentiality + membership auth over the
public tunnel, and is a clean seam to harden later.

Unkeyed = plaintext, for local dev. Keying is network-wide: all peers keyed with
the same PSK, or all unkeyed. A mismatch fails fast and loudly.
"""

from __future__ import annotations

import binascii
import os
import struct

import msgpack

from . import protocol

# 256 MB: generous for a long-prompt prefill tensor, but bounds a hostile length
# prefix so a peer can't make us allocate gigabytes (the 4B prefix allows 4 GB).
MAX_FRAME_BYTES = 256 * 1024 * 1024

_HS_INFO = b"drift-secure-v1"


# --------------------------------------------------------------- key material
def config_dir() -> str:
    d = os.environ.get("DRIFT_CONFIG_DIR") or os.path.join(
        os.path.expanduser("~"), ".config", "drift")
    os.makedirs(d, exist_ok=True)
    return d


def _default_network_key_path() -> str:
    return os.path.join(config_dir(), "network.key")


def _default_identity_path() -> str:
    return os.path.join(config_dir(), "identity.key")


def _decode_key(s: str) -> bytes:
    s = s.strip()
    try:
        raw = binascii.unhexlify(s)
    except (binascii.Error, ValueError):
        import base64
        raw = base64.b64decode(s)
    if len(raw) != 32:
        raise ValueError(f"network key must be 32 bytes, got {len(raw)}")
    return raw


def network_key() -> bytes | None:
    """The active network PSK, or None (plaintext). Env > file > absent.

    ``DRIFT_NETWORK_KEY`` (hex/base64) wins; else ``DRIFT_NETWORK_KEY_FILE`` or
    the default path if the file exists.
    """
    env = os.environ.get("DRIFT_NETWORK_KEY")
    if env:
        return _decode_key(env)
    path = os.environ.get("DRIFT_NETWORK_KEY_FILE") or _default_network_key_path()
    if os.path.exists(path):
        with open(path) as f:
            return _decode_key(f.read())
    return None


def load_identity():
    """This node's Ed25519 signing key, generating + persisting one on first use."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    path = os.environ.get("DRIFT_IDENTITY_FILE") or _default_identity_path()
    if os.path.exists(path):
        with open(path, "rb") as f:
            return Ed25519PrivateKey.from_private_bytes(_decode_key(f.read().decode()))
    sk = Ed25519PrivateKey.generate()
    raw = sk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                           serialization.NoEncryption())
    with open(path, "w") as f:
        f.write(binascii.hexlify(raw).decode())
    os.chmod(path, 0o600)
    return sk


def identity_pubkey_hex(sk=None) -> str:
    from cryptography.hazmat.primitives import serialization

    sk = sk or load_identity()
    pub = sk.public_key().public_bytes(serialization.Encoding.Raw,
                                       serialization.PublicFormat.Raw)
    return binascii.hexlify(pub).decode()


# ------------------------------------------------------------------- channels
class PlainChannel:
    """Unkeyed transport: msgpack frames, no encryption (local dev)."""

    def __init__(self, sock):
        self.sock = sock

    def send(self, obj) -> None:
        protocol.send_msg(self.sock, obj)

    def recv(self) -> dict:
        return protocol.recv_msg(self.sock)

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


class SecureChannel:
    """AEAD transport: ChaCha20-Poly1305 with a per-direction counter nonce."""

    def __init__(self, sock, send_key: bytes, recv_key: bytes):
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        self.sock = sock
        self._enc = ChaCha20Poly1305(send_key)
        self._dec = ChaCha20Poly1305(recv_key)
        self._sctr = 0
        self._rctr = 0

    @staticmethod
    def _nonce(ctr: int) -> bytes:
        return ctr.to_bytes(12, "big")

    def send(self, obj) -> None:
        body = msgpack.packb(obj, use_bin_type=True)
        ct = self._enc.encrypt(self._nonce(self._sctr), body, None)
        self._sctr += 1
        self.sock.sendall(struct.pack(">I", len(ct)) + ct)

    def recv(self) -> dict:
        (n,) = struct.unpack(">I", protocol._recvn(self.sock, 4))
        if n > MAX_FRAME_BYTES:
            raise ValueError(f"secure frame length {n} exceeds cap {MAX_FRAME_BYTES}")
        ct = protocol._recvn(self.sock, n)
        body = self._dec.decrypt(self._nonce(self._rctr), ct, None)
        self._rctr += 1
        return msgpack.unpackb(body, raw=False)

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def _derive(shared: bytes, psk: bytes, nonce_c: bytes, nonce_s: bytes):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    material = HKDF(algorithm=hashes.SHA256(), length=64,
                    salt=nonce_c + nonce_s, info=_HS_INFO).derive(shared + psk)
    return material[:32], material[32:]  # (client→server, server→client)


def _eph():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    priv = X25519PrivateKey.generate()
    pub = priv.public_key().public_bytes(serialization.Encoding.Raw,
                                         serialization.PublicFormat.Raw)
    return priv, pub


def _peer_pub(raw: bytes):
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey

    return X25519PublicKey.from_public_bytes(raw)


def client_handshake(sock, psk: bytes) -> SecureChannel:
    priv, pub = _eph()
    nonce_c = os.urandom(16)
    protocol.send_msg(sock, {"drift_hs": 1, "eph": pub, "nonce": nonce_c})
    hello = protocol.recv_msg(sock)
    if hello.get("drift_hs") != 1:
        raise ConnectionError("peer is not speaking the DRIFT secure handshake (key mismatch?)")
    shared = priv.exchange(_peer_pub(hello["eph"]))
    k_c2s, k_s2c = _derive(shared, psk, nonce_c, hello["nonce"])
    return SecureChannel(sock, send_key=k_c2s, recv_key=k_s2c)


def server_handshake(sock, psk: bytes) -> SecureChannel:
    hello = protocol.recv_msg(sock)
    if hello.get("drift_hs") != 1:
        raise ConnectionError("peer is not speaking the DRIFT secure handshake (key mismatch?)")
    priv, pub = _eph()
    nonce_s = os.urandom(16)
    protocol.send_msg(sock, {"drift_hs": 1, "eph": pub, "nonce": nonce_s})
    shared = priv.exchange(_peer_pub(hello["eph"]))
    k_c2s, k_s2c = _derive(shared, psk, hello["nonce"], nonce_s)
    return SecureChannel(sock, send_key=k_s2c, recv_key=k_c2s)


def dial(host, port):
    """Connect to a DRIFT peer; returns a Channel (secure if a network key is set)."""
    import socket

    sock = socket.create_connection((host, int(port)))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    key = network_key()
    if key is None:
        return PlainChannel(sock)
    try:
        return client_handshake(sock, key)
    except Exception:
        sock.close()
        raise


def accept_wrap(conn):
    """Wrap an accepted connection; returns a Channel (secure if a key is set)."""
    import socket

    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    key = network_key()
    if key is None:
        return PlainChannel(conn)
    return server_handshake(conn, key)


# ------------------------------------------------------------------ keygen CLI
def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="drift keygen",
                                 description="generate the DRIFT network key + node identity")
    ap.add_argument("--print", action="store_true", dest="show",
                    help="print the existing network key (to share with other machines) and exit")
    ap.add_argument("--force", action="store_true", help="overwrite an existing network key")
    args = ap.parse_args(argv)

    net_path = _default_network_key_path()
    if args.show:
        key = network_key()
        if key is None:
            print("no network key set — run `drift keygen` to create one", flush=True)
            return 1
        print(binascii.hexlify(key).decode(), flush=True)
        return 0

    if os.path.exists(net_path) and not args.force:
        print(f"network key already exists at {net_path} (use --force to replace)", flush=True)
    else:
        with open(net_path, "w") as f:
            f.write(binascii.hexlify(os.urandom(32)).decode())
        os.chmod(net_path, 0o600)
        print(f"wrote network key → {net_path}", flush=True)

    sk = load_identity()  # generates + persists on first call
    print(f"node identity     → {_default_identity_path()}", flush=True)
    print(f"identity pubkey   : {identity_pubkey_hex(sk)}", flush=True)
    print("\nShare the network key with every machine (same key = same network):", flush=True)
    print(f"  export DRIFT_NETWORK_KEY={binascii.hexlify(network_key()).decode()}", flush=True)
    print("Then `drift node` / `drift run` on any of them run encrypted automatically.", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
