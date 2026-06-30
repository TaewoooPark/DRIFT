---
name: drift-protocol-tester
description: >-
  STUB — to be completed before milestone M3 (spec §9). Stress-tests the DRIFT neutral wire
  protocol (protocol.py, spec §6) and traces cross-node messages. Will feed truncated frames,
  malformed msgpack, oversized length prefixes, partial recv, and injected latency to the
  framing layer, and trace seq_id/session_id with frame dumps for cross-node debugging. Use
  once the socket path exists (M3+). Triggers: "protocol test", "wire fuzzing", "malformed
  msgpack", "truncated frame", "partial recv", "trace seq_id", "frame dump", "socket
  robustness".
---

# DRIFT Protocol Tester (STUB)

> **Status: stub.** Per `docs/04-skills-mcp-plan.md`, this skill is scaffolded now to record
> intent and completed when M3 starts (the socket path is the first thing it can exercise).
> Do not rely on it before M3.

Planned scope, against `protocol.py` (spec §6: 4-byte big-endian length prefix + msgpack
dict; `send_msg`/`recv_msg`/`_recvn`):

## Stress / fuzz cases (to implement)
- **Truncated frame** — send `len` prefix then fewer bytes; `_recvn` must block/raise cleanly, never return short.
- **Malformed msgpack** — valid length prefix, garbage body; `recv_msg` must raise, not hang.
- **Oversized length prefix** — huge `len`; must not allocate unbounded / must time out.
- **Partial recv** — deliver the body in 1-byte chunks; `_recvn` must reassemble correctly.
- **Injected latency / reordering** — wrap the socket to delay sends; confirm `seq_id` monotonicity is what catches reordering.
- **Peer close mid-frame** — `_recvn` must raise `ConnectionError("peer closed")`.

## Tracing (to implement)
- Log every frame as `{seq_id, session_id, type, shape, dtype, nbytes}` on both ends.
- Dump raw frames to disk on mismatch for offline diff.
- Correlate a token's `seq_id` across orchestrator → shard A → shard B for latency breakdown.

## Invariant under test
The wire contract (spec §1.2, §6) is **immutable**. These tests assert the framing/round-trip
is lossless and robust **without** changing the schema — fp16 CPU round-trip is bitwise-
lossless, so any content change is a framing bug, not a protocol-design question.

When completing this skill at M3, mirror the failure taxonomy in
`docs/05-parity-debugging-playbook.md` Step 4.
