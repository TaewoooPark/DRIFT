"""DRIFT wire protocol — THE CONTRACT (spec §6, immutable once set).

Transport: TCP. Each message = 4-byte big-endian unsigned length prefix + a
msgpack-encoded dict. Language/framework neutral — any runtime that implements
this framing can join.

Request schema:
    {
      "type":        "prefill" | "decode" | "reset" | "ping" | "configure",
      "session_id":  str,
      "seq_id":      int,                 # monotonic, for ordering/debug
      "shape":       [B, S, D],           # hidden_states shape (decode: S=1)
      "dtype":       "float16",
      "position_ids":[int, ...],          # length S, absolute positions (RoPE)
      "input_ids":   [int, ...],          # length S, token ids — for PLE models
                                          # (Gemma 4); plain models ignore it.
      "tensor":      <bytes>,             # row-major hidden_states raw bytes
    }
Response schema:
    { "ok": bool, "shape": [B,S,D], "dtype": "float16", "tensor": <bytes>, "error": str|null }
ping response: { "ok": true, "assigned": bool, "name", "start_layer", "end_layer", "device",
                 "torch", "transformers", "endian" }
    The `endian`/`torch`/`transformers` fields let the head reject a byte-order
    mismatch (the fp16 tensor bytes are native-endian) and warn on version skew
    before assigning layers — both silently break cross-machine parity otherwise.

`configure` lets the orchestrator assign a layer range to an *unassigned* node
(so users never hand-write ranges): a new message TYPE — the 4B+msgpack framing
is unchanged — carrying { "type":"configure", "model_id", "dtype",
"start_layer", "end_layer", "device"? }. The node builds/loads its engine and
replies with its ping info. Pre-assigned nodes (fixed range at launch) ignore it.

The boundary carries only hidden_states (floats) + position_ids + input_ids (ints).
No KV, no framework objects. fp16 CPU round-trip is bit-lossless, so serialization
does not perturb parity (spec §9 premise).
"""

from __future__ import annotations

import struct

import msgpack
import numpy as np

# numpy dtype names used on the wire, keyed by the string in the message.
_NP_DTYPE = {"float16": np.float16, "float32": np.float32, "bfloat16": None}

# Bound the 4-byte length prefix (which by itself would admit a 4 GB body): a
# hostile or corrupt peer must not be able to make us pre-allocate gigabytes.
# 256 MB is generous headroom for a long-prompt prefill hidden state.
MAX_MSG_BYTES = 256 * 1024 * 1024


def send_msg(sock, obj: dict) -> None:
    """Frame and send one message: 4-byte BE length prefix + msgpack body."""
    body = msgpack.packb(obj, use_bin_type=True)
    sock.sendall(struct.pack(">I", len(body)) + body)


def recv_msg(sock) -> dict:
    """Receive exactly one framed message (length-capped against alloc-DoS)."""
    (n,) = struct.unpack(">I", _recvn(sock, 4))
    if n > MAX_MSG_BYTES:
        raise ValueError(f"message length {n} exceeds cap {MAX_MSG_BYTES} — refusing to allocate")
    return msgpack.unpackb(_recvn(sock, n), raw=False)


def _recvn(sock, n: int) -> bytes:
    """Read exactly n bytes (loops over partial recv — the classic failure)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def tensor_to_bytes(t, dtype: str = "float16") -> bytes:
    """Serialize a torch tensor to row-major raw bytes via a CPU fp16 cast.

    The CPU fp16 round-trip is bit-lossless (spec §6 note), so this does not
    affect parity.
    """
    import torch

    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    return t.detach().to("cpu", torch_dtype).contiguous().numpy().tobytes()


def bytes_to_tensor(b: bytes, shape, dtype: str, device: str):
    """Reconstruct a torch tensor from raw bytes.

    `np.frombuffer` returns a read-only array, so `.copy()` before `from_numpy`.
    """
    import torch

    np_dtype = _NP_DTYPE.get(dtype)
    if np_dtype is None:
        raise ValueError(f"unsupported wire dtype: {dtype}")
    arr = np.frombuffer(b, dtype=np_dtype).reshape(tuple(shape)).copy()
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    return torch.from_numpy(arr).to(device=device, dtype=torch_dtype)
