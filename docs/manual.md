# DRIFT — Operations Manual

**How to run DRIFT for real — start to finish.** Language: **English** ·
[한국어](manual.ko.md) · [中文](manual.zh.md) · [日本語](manual.ja.md)

The first half is the whole job: install, try it, run one model across your machines.
The second half — **Customize & fine-tune** — is only for when the defaults aren't enough.
For benchmark methodology and numbers, see [`benchmarks.md`](benchmarks.md).

---

## Table of contents

**Getting it running**
1. [Install](#1--install)
2. [Run it on one machine](#2--run-it-on-one-machine)
3. [Run it across your machines — a worked example](#3--run-it-across-your-machines--a-worked-example)

**Customize & fine-tune**
4. [`config.yaml` reference](#4--configyaml-reference)
5. [Choosing a split point](#5--choosing-a-split-point)
6. [Models](#6--models)
7. [Devices & dtype](#7--devices--dtype)
8. [How generation works](#8--how-generation-works)
9. [Driving the shards by hand](#9--driving-the-shards-by-hand)
10. [CLI reference](#10--cli-reference)
11. [The wire & sessions](#11--the-wire--sessions)
12. [Memory](#12--memory)
13. [Troubleshooting](#13--troubleshooting)
14. [Decentralization — chain, encryption, failover, gossip, ledger, int8](#14--decentralization-v10)

---

## 1 · Install

Requires **Python 3.12** and [`uv`](https://github.com/astral-sh/uv). Both bundled models are
**ungated** — no Hugging Face login. Run this **on every machine**:

```bash
git clone https://github.com/TaewoooPark/DRIFT && cd DRIFT
bash scripts/install.sh          # macOS / Linux
# Windows (NVIDIA):  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
drift doctor                     # checks Python, torch, device, config, ports
```

The installer makes a 3.12 venv and installs DRIFT (`drift` CLI). The platform-correct torch
wheel picks the GPU backend automatically — MPS on Apple, CUDA on Linux; on Windows the
script pulls the CUDA build. `drift doctor` should show your device (`mps` or `cuda`).

---

## 2 · Run it on one machine

```bash
drift up 2                        # spawn 2 local nodes, auto-split the model, open a chat
drift up 2 --prompt "hello world" # …or a one-shot answer
```

`drift up N` launches N worker nodes on this machine, reads the model's layer count, splits it
evenly, assigns each node its range, and generates. No layer ranges, no ports, no device flags.
That's the fastest way to see it work; the next section puts the nodes on *different* machines.

---

## 3 · Run it across your machines — a worked example

**Goal:** type `hello world` on your **Mac**, and have the answer computed using **both** the
Mac (Apple/MPS) **and** a Windows PC (NVIDIA/CUDA).

**Roles.** The **head** types the prompt and holds `embed` + `lm_head`; the decoder layers live
on the **nodes**. So to use *both* GPUs for the layers, the Mac runs a **node** *and* the head,
and the PC runs a **node**:

```bash
# ── on the Windows PC (NVIDIA) ───────────────────  one terminal
drift node --port 52601           # auto device = cuda, announced on the LAN
#   (allow python through Windows Defender Firewall on Private networks,
#    and turn on Network Discovery so the Mac can find it)

# ── on the Mac (Apple) ───────────────────────────  terminal 1: a worker
export PYTORCH_ENABLE_MPS_FALLBACK=1
drift node --port 52600           # auto device = mps

# ── on the Mac ───────────────────────────────────  terminal 2: the head
drift run --prompt "hello world"  # finds both nodes, splits 28 layers, streams
```

**What you'll see** — the head discovers both nodes, splits the model, and streams:

```
[run] discovering nodes on the LAN …
[run] found 192.168.0.22:52601(cuda), 127.0.0.1:52600(mps)

  model : Qwen/Qwen2.5-1.5B-Instruct
  head  : embed + norm + lm_head  · device=mps
  node  : 127.0.0.1:52600     layers [0:14)   · device=mps      ← the Mac computes these
  node  : 192.168.0.22:52601  layers [14:28)  · device=cuda     ← the PC computes these

Hello! How can I help you today?
```

**If the head can't find the PC** (mDNS is often blocked on guest / corporate Wi-Fi), name the
nodes explicitly — that's why we pinned ports above:

```bash
drift run --nodes 192.168.0.22:52601,127.0.0.1:52600 --prompt "hello world"
```

(The Windows box by its LAN IP; the Mac's own node as `127.0.0.1`.) Check reachability first
with `drift doctor --nodes 192.168.0.22:52601`.

**Same commands, any pair.** Two Macs or two Windows PCs work identically — `drift node`
auto-detects each device, `drift run` finds and splits. Only two things are specific to the
mixed Mac + Windows case:

- **Cross-vendor float drift.** MPS and CUDA round fp16 slightly differently, so a long greedy
  answer may diverge from a single machine in *later* tokens. This is expected, not a bug (early
  tokens match, the text stays coherent). Two same-vendor nodes reproduce a single machine
  **bitwise**.
- **Two OSes.** Install with `install.sh` on the Mac and `install.ps1` on the PC; everything
  after is identical.

---

**Customize & fine-tune** — everything below is optional, for when the one-command flow above
isn't enough (a different model, an uneven split, exact ports, driving the pieces by hand).

---

## 4 · `config.yaml` reference

`config.yaml` is the single source of truth for the model, precision, and (for the by-hand
flow) the shard table. `drift up` / `drift run` read `model_id`, `dtype`, and `generation`
from it; they compute the split themselves.

```yaml
model_id: "Qwen/Qwen2.5-1.5B-Instruct"   # any HF causal-LM id
dtype: "float16"                          # float16 | float32  (see §7)
device: "mps"                             # default device: mps | cuda | cpu
port: 52600                               # default port for a shard that sets none

shards:                                   # only used by the by-hand flow (§9) / `drift run` fallback
  - { name: "mac",     host: "127.0.0.1", port: 52600, start_layer: 0,  end_layer: 14, device: "mps" }
  - { name: "windows", host: "127.0.0.1", port: 52601, start_layer: 14, end_layer: 28, device: "mps" }

generation:
  max_new_tokens: 50
  prompt: "Give me a short introduction to large language models."
```

| Key | Meaning |
|---|---|
| `model_id` | Hugging Face model id. Downloaded once to the local HF cache. |
| `dtype` | Compute **and** wire dtype. `float16` (default, lossless CPU round-trip) or `float32`. `bfloat16` is **not** valid on the wire — §7. |
| `device` | Default device for the head and for a shard that omits its own. `mps` / `cuda` / `cpu`. |
| `port` | Fallback port for a shard with no `port` and no `DRIFT_PORT`. |
| `shards[]` | Ordered shard table for the by-hand flow (§9) and the `drift run` fallback when discovery finds nothing. |
| `shards[].host` / `port` | Where the orchestrator dials this shard. `127.0.0.1` local; a LAN IP for remote. |
| `shards[].start_layer` / `end_layer` | Half-open decoder-layer range `[start, end)`. |
| `shards[].device` | Device for this shard. |
| `generation.max_new_tokens` | Default token budget (override with `--max-new-tokens`). |
| `generation.prompt` | Default prompt when `--prompt` is omitted. |

---

## 5 · Choosing a split point

`drift run` splits evenly by node count; you only think about this for the by-hand flow (§9)
or an uneven split. The ranges must **tile** the decoder layers: contiguous, in order, no gaps,
no overlaps, covering `[0, num_hidden_layers)`. The head owns `embed_tokens`, the final norm,
and `lm_head` — never part of a shard range.

```
model: 28 decoder layers (Qwen2.5-1.5B)
        └── [0, 14)  /  [14, 28)                 ✅ tiles 0..28 (the even 2-way split)
        └── [0, 10) / [10, 20) / [20, 28)        ✅ three shards, also valid
        └── [0, 14) / [16, 28)                   ❌ gap at 14–15
        └── [0, 16) / [14, 28)                   ❌ overlap at 14–15
```

Where you cut costs nothing in correctness (any tiling is bitwise-exact on one device); it only
shifts how much compute and weight memory each node carries. Skew it toward the faster machine
if they differ. Layer counts: Qwen2.5-1.5B = 28, Gemma-4-E2B = 35.

---

## 6 · Models

The engine **introspects** the loaded model (decoder-layer class, `rotary_emb`, cache type,
per-layer attention) instead of hardcoding an architecture — so new families drop in by id.
Set `model_id` in `config.yaml` (or `drift run --model <id>`); nothing else.

| Model | Layers | Even split | Notes |
|---|---:|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` (default) | 28 | `0–14 / 14–28` | plain decoder; the parity baseline |
| `google/gemma-4-E2B-it` | 35 | `0–18 / 18–35` | Per-Layer Embeddings (nodes rebuild them from `input_ids`), dual RoPE θ, hybrid attention, `HybridCache`; needs `transformers ≥ 5.5` |

A larger model just has more layers to split — keep the ranges contiguous over its count.

---

## 7 · Devices & dtype

**Devices** — `mps` (Apple GPU), `cuda` (NVIDIA GPU), `cpu` (portable, slow). Each node's
device is independent; that independence is the whole point. `drift node` auto-detects it;
override with `--device`.

**dtype** — `float16` (default) or `float32`. The wire serializes tensors as raw bytes of this
dtype and the fp16 CPU round-trip is bit-lossless, so serialization never perturbs the result.
`bfloat16` is **not** supported on the wire — use `float16`.

**Same vendor vs mixed** — two shards on the same device family reproduce a single machine
**bitwise**. Mixing `mps` and `cuda` gives bit-level fp16 rounding differences, so greedy
decoding may diverge in later tokens (expected — §3).

---

## 8 · How generation works

- **Greedy.** Both the reference oracle and the orchestrator take `argmax` each step; there is
  no temperature/top-p sampling in the CLI yet. Parity tests force greedy for determinism.
- **EOS.** `drift run` / `drift up` stop at the model's end-of-sequence id(s) (a narrow set, not
  every special token). The parity/reference paths run a fixed `max_new_tokens` with no stop.
- **Prefill then decode.** The whole prompt is processed once (positions `0…S-1`), then one
  token at a time. Each node keeps its own KV cache across steps.

---

## 9 · Driving the shards by hand

The `drift node` / `drift run` flow is the easy path. The lower-level commands give exact
control (fixed ports/ranges, no discovery) and are what the parity gate and benchmark use.

**1) Point `config.yaml` at each machine** — set each shard's `host`/`device` and open ports:

```yaml
shards:
  - { name: "mac",     host: "192.168.0.11", port: 52600, start_layer: 0,  end_layer: 14, device: "mps"  }
  - { name: "windows", host: "192.168.0.22", port: 52601, start_layer: 14, end_layer: 28, device: "cuda" }
```

**2) Start a pre-assigned shard server on each box** (bind `0.0.0.0` to accept remote peers):

```bash
# on the Mac
DRIFT_PORT=52600 python -m drift.shard_server --name mac     --start 0  --end 14 --device mps  --host 0.0.0.0 --preload
# on the Windows PC
set DRIFT_PORT=52601
python -m drift.shard_server --name windows --start 14 --end 28 --device cuda --host 0.0.0.0 --preload
```

**3) Drive from the head** — reads hosts/ports from `config.yaml`:

```bash
python -m drift.orchestrator --ping                                  # both shards reply
python -m drift.orchestrator --prompt "Explain pipeline parallelism." # generate over the wire
```

Open the ports in each firewall. The orchestrator node also loads the model (for
embed/norm/head), so run it on whichever box is convenient.

---

## 10 · CLI reference

Every command takes `--config` (default `config.yaml`).

### `drift` — the high-level commands

| Command | What it does |
|---|---|
| `drift doctor` | preflight: Python/torch/device, deps, `config.yaml` tiling, port reachability (`--nodes`), firewall hints |
| `drift up N` | localhost: spawn N nodes, auto-split, chat (or `--prompt` for one-shot) |
| `drift node` | run THIS machine as a worker: auto device, `--port`, LAN-announced, waits for the head |
| `drift run` | the head: discover nodes (or `--nodes host:port,…`), auto-split, configure, stream/chat |
| `drift serve` | OpenAI-compatible HTTP/SSE API over the DRIFT orchestrator (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`) |
| `drift keygen` | create/print the network key + node identity (§14) |
| `drift ledger` | per-node contribution from a receipt journal — `--verify` · `--csv` (§14) |

`up`, `node`, `run` take `--max-new-tokens`; `run`/`up` also take `--chain`, `--thin`,
`--int8` (§14); `run` also takes `--model`, `--nodes`, `--no-discover`, `--expand`;
`node` takes `--tunnel`, `--join`, `--no-advertise`; `serve` takes `--api-key`,
`--cors-origin`, `--served-model-name`, and the same node/model routing flags as
`run`. Omit `--prompt` for a chat.

### Lower-level modules

| Module | Key flags |
|---|---|
| `python -m drift.shard_server` | `--name --start --end --device --host --port --preload --tamper` (+ `DRIFT_PORT`) |
| `python -m drift.orchestrator` | `--ping` · `--prompt` · `--max-new-tokens` · `--ports` |
| `python -m drift.reference` | `--device --out` — single-machine oracle |
| `python -m drift.parity_test` | `--mode inprocess\|socket` · `--ports` · `--selftest` · `--prefix-match K` |
| `python -m drift.itest` | real-node gate: `--nodes N` · `--chain --secure --thin --int8` · `--kill K --tamper K --expand N --ledger` |
| `python -m drift.verify` | trustless recompute spot-check: `--nodes host:port,… --tol` |
| `python -m drift.ledger` | `<journal.jsonl> --verify --csv` |
| `python -m drift.bench` | `--quick --no-socket --json` (see [`benchmarks.md`](benchmarks.md)) |

---

## 11 · The wire & sessions

- **Contract (`drift/protocol.py`, frozen):** every message is a 4-byte big-endian length
  prefix + a msgpack dict (one ChaCha20-Poly1305 frame when a key is set). Any runtime that
  implements this framing can be a node — no PyTorch on the wire. Message types: `ping` /
  `configure` / `prefill` / `decode` / `reset` / `peers_get` / `peer_announce`.
- **What crosses:** only `hidden_states` (fp16, or int8 with `--int8`) + `position_ids` +
  `input_ids`. In chain mode two optional fields (`route`, `collect`) carry the downstream path;
  each hop attaches a signed `receipt`. The **KV cache never crosses** — each node keeps its own.
  Per-token traffic is `hidden_size × 2` bytes (fp16) or ≈`hidden_size × 1` (int8) plus a few ints.
- **Fungible nodes.** A `drift node` starts unassigned; the head sends a `configure` (model +
  layer range) so you never hand-write ranges. Pre-assigned servers (§9) skip it.
- **Sessions.** A generation is a `session_id`; each node holds a per-session KV cache and the
  head sends `reset` when it ends. A node serves **one connection at a time** — don't point two
  heads at one node.

---

## 12 · Memory

Plan for the **full model in RAM/VRAM on every node** today: each node loads the whole
checkpoint and then uses only its layer slice, and the head loads the model too (for
embed/norm/head). The *active* parameters per node are smaller (the heaviest node's own layers
are ~42% of the model for the default 2-way split — see [`benchmarks.md`](benchmarks.md)), but
trimming the load to just the slice is future work. Until then: use a model that fits each
node, or split across **more** nodes to shrink each node's active share.

---

## 13 · Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `drift run` finds no nodes | mDNS blocked (guest/corporate Wi-Fi) → name them: `drift run --nodes host:port,…`. Confirm each `drift node` printed its address. |
| `ConnectionRefusedError` | Node not up, or wrong host/port. Start the node first; check the port; `drift doctor --nodes host:port`. |
| Works locally, not across machines | A node bound to `127.0.0.1`. `drift node` binds `0.0.0.0` by default; for the by-hand server pass `--host 0.0.0.0`. Open the firewall port. |
| Windows: peers can't reach it | Allow `python.exe` through Defender Firewall (Private), enable Network Discovery. |
| Output drifts only in **late** tokens (MPS↔CUDA) | Expected vendor fp16 rounding (§3, §7). Not a bug. |
| Parity **FAIL at token 1–2** | A real bug (mask/KV/RoPE), not float noise. |
| Out of memory on load | Each process loads the full checkpoint (§12). Use a smaller model, more nodes, or `--no-socket` for the bench. |
| `unsupported wire dtype` | compute `dtype` must be `float16` or `float32` (§7); `int8` is a *wire* option (`--int8`, §14), not a compute dtype. |
| `refusing --tunnel without a network key` | a public endpoint would be open compute — run `drift keygen` and `export DRIFT_NETWORK_KEY` first (§14). |
| A node flagged SUSPECT | the receipt verifier caught a mismatch (§14) — check that node's version/health; a genuine tamper is real. |
| Rare MPS op error on Mac | Ensure `export PYTORCH_ENABLE_MPS_FALLBACK=1` in the shell that launched the process. |

---

## 14 · Decentralization (v1.0)

The split core is unchanged; these are opt-in layers on top. Every one is gated
bitwise (or, for int8, under the relaxed gate) — see `python -m drift.itest`.

### Peer-to-peer chain — `--chain`
By default every hop round-trips through the head (star). `--chain` streams the
hidden state node→node→…→tail→head instead: tensor crossings/token drop from `2N`
to `N+1`, and the head's bandwidth becomes O(1) in the node count.
```bash
drift up 3 --chain
drift run --chain --nodes a:52600,b:52601,c:52602 --prompt "…"
```

### Weightless head — `--thin` (implies `--chain`)
`embed_tokens` moves to the first node, `norm`+`lm_head`+`argmax` to the last. The
head holds only the tokenizer and exchanges token ids, no tensor.
```bash
drift up 2 --thin
```

### Encrypted + authenticated wire — `drift keygen`
A network shares one pre-shared key. Every connection then runs X25519 ECDH →
HKDF(mix PSK) → ChaCha20-Poly1305; a dialer without the key is dropped.
```bash
drift keygen                       # writes ~/.config/drift/network.key + identity; prints the key
export DRIFT_NETWORK_KEY=<hex>     # on EVERY machine (head + nodes) — now encrypted
drift keygen --print               # re-print the key to share
```
Unkeyed = plaintext (fine for a LAN you own). `drift node --tunnel` **refuses** to
run without a key (a public endpoint must not be open compute).

### Join from anywhere — `drift node --join` / `drift run --expand`
A node gossip-joins a network via one seed; the head discovers the whole
membership and splits across it.
```bash
drift node --join seed-host:52600            # learn the members
drift run --expand --nodes seed-host:52600   # split across all discovered members
```

### Failover
If a node dies mid-generation, the head re-splits over the survivors (+ any
spare), replays the sequence-so-far, and continues — bitwise-identical to an
uninterrupted run. Nothing to configure; it just recovers (or surfaces a clean
`NodeUnavailable` if nothing survives).

### Verification & the contribution ledger
Every hop signs an Ed25519 receipt; the head verifies them on live traffic. Set a
journal to record them, then tally:
```bash
export DRIFT_JOURNAL=~/drift.jsonl && drift run --chain --prompt "…"
drift ledger ~/drift.jsonl --verify --csv out.csv
```
`drift verify --nodes host:port,…` is the recompute spot-check (a node you don't own).

### Half-size wire — `--int8`
Send the hidden state as group-wise int8 (≈0.51× the bytes). Lossy — runs under
the relaxed gate, never bitwise. Measure with `drift itest --int8`.
```bash
drift run --chain --int8 --prompt "…"
```

### Env vars
| Var | Effect |
|---|---|
| `DRIFT_NETWORK_KEY` | hex/base64 PSK — encrypts + authenticates the wire |
| `DRIFT_NETWORK_KEY_FILE` | path to a key file (default `~/.config/drift/network.key`) |
| `DRIFT_IDENTITY_FILE` | this node's Ed25519 identity (default `~/.config/drift/identity.key`) |
| `DRIFT_ADVERTISE_HOST` | address peers use to reach this node (default: LAN ip) |
| `DRIFT_JOURNAL` | path to append verified receipts for `drift ledger` |

---

Reproduce the published numbers with `python -m drift.bench`; methodology is in
[`benchmarks.md`](benchmarks.md).
