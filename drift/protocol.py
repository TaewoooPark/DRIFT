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

# numpy dtype names used on the wire, keyed by the string in the message. numpy
# has no native bfloat16, so bf16 is handled specially (byte-copied via a uint16
# reinterpretation) in tensor_to_bytes / bytes_to_tensor, not through this table.
_NP_DTYPE = {"float16": np.float16, "float32": np.float32}
_WIRE_DTYPES = ("float16", "float32", "bfloat16", "int8")

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
    """Serialize a torch tensor to row-major raw bytes via a CPU cast.

    A same-dtype round-trip is a pure byte copy → bit-lossless (spec §6 note), so
    this does not affect parity. numpy has no bfloat16, so bf16 is byte-copied
    through a uint16 reinterpretation — still an exact 16-bit copy, so bf16 is a
    first-class bitwise-safe wire dtype (compute dtype must match).
    """
    import torch

    if dtype == "bfloat16":
        u = t.detach().to("cpu", torch.bfloat16).contiguous().view(torch.uint16)
        return u.numpy().tobytes()
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    return t.detach().to("cpu", torch_dtype).contiguous().numpy().tobytes()


def bytes_to_tensor(b: bytes, shape, dtype: str, device: str):
    """Reconstruct a torch tensor from raw bytes.

    `np.frombuffer` returns a read-only array, so `.copy()` before `from_numpy`.
    """
    import torch

    if dtype == "bfloat16":
        arr = np.frombuffer(b, dtype=np.uint16).reshape(tuple(shape)).copy()
        return torch.from_numpy(arr).view(torch.bfloat16).to(device=device)
    np_dtype = _NP_DTYPE.get(dtype)
    if np_dtype is None:
        raise ValueError(f"unsupported wire dtype: {dtype}")
    arr = np.frombuffer(b, dtype=np_dtype).reshape(tuple(shape)).copy()
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    return torch.from_numpy(arr).to(device=device, dtype=torch_dtype)


# ------------------------------------------------ int8 wire quantization (M14)
# Halve the wire bytes on a network-bound link: send the hidden state as int8.
# A SINGLE per-tensor scale is fatal here — the residual stream has a few outlier
# channels whose magnitude dominates the scale and crushes every other dim to
# ~0 (this is exactly why LLM activation quantization is hard). So we quantize
# GROUP-WISE: an independent int8 scale per block of `_INT8_GROUP` hidden dims,
# so an outlier only rescales its own block. Bytes: H int8 + (H/group) fp16
# scales ≈ 0.51× fp16. Still lossy → the relaxed gate, never the bitwise one.
_INT8_GROUP = 128


def tensor_to_wire(t, wire_dtype: str):
    """Serialize a tensor for the wire. Returns (bytes, scale). scale is None for
    fp16/fp32; for int8 it is the fp16 bytes of the per-group scales."""
    if wire_dtype != "int8":
        return tensor_to_bytes(t, wire_dtype), None
    import torch

    x = t.detach().to("cpu", torch.float32).numpy()
    H = x.shape[-1]
    G = _INT8_GROUP
    pad = (-H) % G
    flat = x.reshape(-1, H)
    if pad:
        flat = np.pad(flat, ((0, 0), (0, pad)))
    ng = flat.shape[1] // G
    blocks = flat.reshape(-1, ng, G)
    scale = np.abs(blocks).max(axis=2, keepdims=True)
    scale[scale == 0.0] = 1.0
    q = np.round(blocks / scale * 127.0).clip(-127, 127).astype(np.int8)
    scale_bytes = scale.reshape(-1, ng).astype(np.float16).tobytes()
    return q.tobytes(), scale_bytes


def wire_to_tensor(b: bytes, shape, dtype: str, device: str, scale=None):
    """Reconstruct a tensor from the wire (fp16/fp32 raw, or group-wise int8)."""
    if dtype != "int8":
        return bytes_to_tensor(b, shape, dtype, device)
    import torch

    shape = tuple(shape)
    H = shape[-1]
    G = _INT8_GROUP
    ng = (H + G - 1) // G
    N = 1
    for d in shape[:-1]:
        N *= d
    q = np.frombuffer(b, dtype=np.int8).reshape(N, ng, G).astype(np.float32)
    sc = np.frombuffer(scale, dtype=np.float16).reshape(N, ng, 1).astype(np.float32)
    x = (q * sc / 127.0).reshape(N, ng * G)[:, :H].reshape(shape)
    return torch.from_numpy(x.copy()).to(device=device, dtype=torch.float16)
