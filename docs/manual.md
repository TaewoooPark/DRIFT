# DRIFT — Operations Manual

**Everything you can control when running DRIFT for real.** Language: **English** ·
[한국어](manual.ko.md) · [中文](manual.zh.md) · [日本語](manual.ja.md)

For the benchmark methodology and measured numbers, see
[`benchmarks.md`](benchmarks.md). This document is about *operating* the system.

---

## Table of contents

1. [Install](#1--install)
2. [The 60-second run](#2--the-60-second-run)
3. [`config.yaml` reference](#3--configyaml-reference)
4. [Choosing a split point](#4--choosing-a-split-point)
5. [Running across two machines (Mac + Windows)](#5--running-across-two-machines-mac--windows)
6. [CLI reference](#6--cli-reference)
7. [Models](#7--models)
8. [Devices & dtype](#8--devices--dtype)
9. [How generation works](#9--how-generation-works)
10. [The wire & sessions](#10--the-wire--sessions)
11. [Memory](#11--memory)
12. [Troubleshooting](#12--troubleshooting)

---

## 1 · Install

Requires **Python 3.12** (PyTorch has no 3.14 wheel yet) and
[`uv`](https://github.com/astral-sh/uv). Both bundled models are **ungated** — no Hugging
Face login.

```bash
git clone https://github.com/TaewoooPark/DRIFT && cd DRIFT
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install "torch" "transformers>=5.5" safetensors msgpack numpy huggingface_hub accelerate pyyaml
export PYTORCH_ENABLE_MPS_FALLBACK=1        # lets rare unimplemented MPS ops fall back to CPU
```

On Windows/NVIDIA, install the CUDA build of PyTorch for your toolkit instead of the default
wheel; everything else is identical. `PYTORCH_ENABLE_MPS_FALLBACK` is Mac-only and harmless
elsewhere.

---

## 2 · The 60-second run

Two shards on **one** machine (localhost), then a real generation over TCP:

```bash
# terminal A — front half, layers [0,14)
DRIFT_PORT=52600 python -m drift.shard_server --name mac     --start 0  --end 14 --device mps --preload
# terminal B — back half, layers [14,28)
DRIFT_PORT=52601 python -m drift.shard_server --name windows --start 14 --end 28 --device mps --preload
# terminal C — health check, then generate
python -m drift.orchestrator --ping   --ports 52600,52601
python -m drift.orchestrator --prompt "Explain pipeline parallelism in two sentences." --ports 52600,52601
```

The last command **is** the product: the orchestrator embeds the prompt, routes the hidden
state through shard A then shard B, and prints the decoded answer — each shard running only
its own layers. Everything below is how to change what that run does.

---

## 3 · `config.yaml` reference

`config.yaml` is the single source of truth. The orchestrator, shard servers, reference
oracle, and benchmark all read it.

```yaml
model_id: "Qwen/Qwen2.5-1.5B-Instruct"   # any HF causal-LM id
dtype: "float16"                          # float16 | float32  (see §8)
device: "mps"                             # default device: mps | cuda | cpu
port: 52600                               # default single-port (overridden per shard below)

shards:
  - { name: "mac",     host: "127.0.0.1", port: 52600, start_layer: 0,  end_layer: 14, device: "mps" }
  - { name: "windows", host: "127.0.0.1", port: 52601, start_layer: 14, end_layer: 28, device: "mps" }

generation:
  max_new_tokens: 50
  prompt: "Give me a short introduction to large language models."
```

| Key | Meaning |
|---|---|
| `model_id` | Hugging Face model id. Downloaded once to the local HF cache. |
| `dtype` | Compute **and** wire dtype. `float16` (default, lossless CPU round-trip) or `float32`. `bfloat16` is **not** valid over the wire — see §8. |
| `device` | Default device when a shard omits its own. `mps` / `cuda` / `cpu`. |
| `port` | Fallback port for a shard that omits `port` and sets no `DRIFT_PORT`. |
| `shards[]` | Ordered list of shards. The orchestrator routes through them **in this order**. |
| `shards[].name` | Logical name, used by `--ports`/routing and shown in `--ping`. |
| `shards[].host` | Where the orchestrator dials this shard. `127.0.0.1` for local; a LAN IP for remote (§5). |
| `shards[].port` | TCP port the shard listens on / the orchestrator dials. |
| `shards[].start_layer` / `end_layer` | Half-open decoder-layer range `[start, end)` this shard owns. |
| `shards[].device` | Device for this shard (`mps` on the Mac, `cuda` on the PC…). |
| `generation.max_new_tokens` | Default token count for `reference` and the orchestrator demo. |
| `generation.prompt` | Default prompt for `reference` and the orchestrator when `--prompt` is omitted. |

---

## 4 · Choosing a split point

The `shards[]` ranges must **tile** the model's decoder layers: contiguous, in order, no
gaps, no overlaps, covering `[0, num_hidden_layers)`. The orchestrator itself owns
`embed_tokens`, the final norm, and `lm_head` — those are **not** part of any shard range.

```
model: 28 decoder layers (Qwen2.5-1.5B)
        └── shard A: [0, 14)   ── shard B: [14, 28)      ✅ tiles 0..28
        └── [0, 10) / [10, 20) / [20, 28)                ✅ three shards, also valid
        └── [0, 14) / [16, 28)                           ❌ gap at 14–15
        └── [0, 16) / [14, 28)                           ❌ overlap at 14–15
```

- **Number of shards** is just the length of `shards[]` — 2 is the demo, more is fine; the
  orchestrator routes through all of them in list order.
- **Where to cut** trades nothing for correctness (any tiling is bitwise-exact on one device);
  it only shifts *how much compute and weight memory* lands on each node. An even layer split
  is the default; skew it if one machine is faster.
- **Layer counts:** Qwen2.5-1.5B = 28, Gemma-4-E2B = 35. Read it from any run's startup log
  (`reference` prints `layers=…`) or the model config.

---

## 5 · Running across two machines (Mac + Windows)

The localhost run in §2 becomes a real cluster with three changes.

**1) Point the config at each machine.** On the orchestrator node, set each shard's `host` to
that machine's LAN IP and pick open ports:

```yaml
shards:
  - { name: "mac",     host: "192.168.0.11", port: 52600, start_layer: 0,  end_layer: 14, device: "mps"  }
  - { name: "windows", host: "192.168.0.22", port: 52601, start_layer: 14, end_layer: 28, device: "cuda" }
```

**2) Bind each shard server to a reachable address.** The server defaults to `127.0.0.1`
(local only). To accept remote connections, start it with `--host 0.0.0.0`:

```bash
# on the Mac (192.168.0.11)
DRIFT_PORT=52600 python -m drift.shard_server --name mac     --start 0  --end 14 --device mps  --host 0.0.0.0 --preload
# on the Windows PC (192.168.0.22)
set DRIFT_PORT=52601
python -m drift.shard_server --name windows --start 14 --end 28 --device cuda --host 0.0.0.0 --preload
```

**3) Run the orchestrator from the node that holds the head.** It reads `host`/`port` straight
from `config.yaml`, so **omit `--ports`** (which only overrides ports, not hosts):

```bash
python -m drift.orchestrator --ping                                  # both shards should reply
python -m drift.orchestrator --prompt "Write a haiku about winter."  # front half on Apple, back half on NVIDIA
```

Notes:
- Open the chosen ports in each machine's firewall.
- The orchestrator node also loads the model (for embed/norm/head), so run it on whichever
  box is convenient — commonly the same as shard A.
- Across **different GPU vendors** (MPS ↔ CUDA), fp16 rounds slightly differently on each, so
  greedy output can diverge in *later* tokens. Early tokens match and the text stays coherent;
  this is expected. On the **same** device family the split is bitwise-exact (see
  [`benchmarks.md`](benchmarks.md)).

---

## 6 · CLI reference

Every entry point takes `--config` (default `config.yaml`).

### `drift.shard_server` — run one shard

```bash
DRIFT_PORT=<port> python -m drift.shard_server [flags]
```

| Flag | Default | Meaning |
|---|---|---|
| `--name` | `shard` | Logical shard name. |
| `--start` / `--end` | from config shard[0] | Decoder-layer range `[start, end)`. |
| `--device` | config `device` | `mps` / `cuda` / `cpu`. |
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` to accept remote nodes. |
| `--port` | `$DRIFT_PORT` or config `port` | Listen port. |
| `--preload` | off | Load weights **before** listening (recommended; avoids a cold first request). |

### `drift.orchestrator` — health check & generate

```bash
python -m drift.orchestrator [--ping] [--prompt "…"] [--max-new-tokens N] [--ports P1,P2]
```

| Flag | Meaning |
|---|---|
| `--ping` | Ping every shard over TCP and exit (this is the "M0" reachability check). |
| `--prompt` | Prompt to generate from. Falls back to `generation.prompt`. |
| `--max-new-tokens` | Token budget. Falls back to `generation.max_new_tokens`. |
| `--ports` | Comma-separated ports overriding each shard's config port (host unchanged — local use). |

Generation here is greedy and **stops at EOS**.

### `drift.reference` — single-machine oracle

```bash
python -m drift.reference [--device DEV] [--out reference_out.npz]
```

Loads the whole model on one device and greedily generates `generation.max_new_tokens` from
`generation.prompt`, saving the token ids + first-step logits. This is the ground truth the
split path is checked against.

### `drift.parity_test` — correctness gate

```bash
python -m drift.parity_test --mode inprocess               # split in one process, no sockets
python -m drift.parity_test --mode socket --ports 52600,52601   # split over TCP (servers must be up)
python -m drift.parity_test --selftest                     # 6 prompts (EN/code/Korean, n=1…180)
```

| Flag | Meaning |
|---|---|
| `--mode` | `inprocess` (direct calls) or `socket` (over the wire). |
| `--ports` | Ports for socket mode. |
| `--ref` | Reference file to compare against (default `reference_out.npz`). |
| `--selftest` | Re-derive a fresh reference and compare across several prompts/lengths; no npz needed. |

### `drift.bench` — benchmarks

```bash
python -m drift.bench [--quick] [--no-socket] [--json out.json]
```

See [`benchmarks.md`](benchmarks.md). `--no-socket` skips the server-spawning overhead
measurement on low-RAM machines.

---

## 7 · Models

The engine **introspects** the loaded model (decoder-layer class, `rotary_emb`, cache type,
per-layer attention) instead of hardcoding an architecture, so new families drop in by id.

| Model | Layers | Split example | Notes |
|---|---:|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` (default) | 28 | `0–14 / 14–28` | plain decoder; the parity baseline |
| `google/gemma-4-E2B-it` | 35 | `0–18 / 18–35` | Per-Layer Embeddings (shards rebuild them from `input_ids`), dual RoPE θ, hybrid attention, `HybridCache`; needs `transformers ≥ 5.5` |

To switch models, set `model_id` and a valid tiling in `config.yaml` — nothing else. A larger
model simply has more layers to split; keep the ranges contiguous over its layer count.

---

## 8 · Devices & dtype

**Devices** — `mps` (Apple GPU), `cuda` (NVIDIA GPU), `cpu` (portable, slow). A shard's
`device` is independent of the others; that independence is the whole point.

**dtype** — `float16` (default) or `float32`. The wire serializes tensors as raw bytes of
this dtype, and the fp16 CPU round-trip is bit-lossless, so serialization never perturbs the
result. `bfloat16` is **not** supported on the wire — if you need bf16 compute, that is not
yet wired through; use `float16`.

**Same device vs mixed vendors:** two shards on the same device family reproduce a single
machine **bitwise**. Mixing `mps` and `cuda` gives bit-level fp16 rounding differences between
vendors, so greedy decoding may diverge in later tokens (expected, not a bug — §5).

---

## 9 · How generation works

- **Greedy only.** Both the reference oracle and the orchestrator pick `argmax` each step;
  there is no temperature/top-p sampling exposed in the CLI. Parity tests force greedy so the
  output is deterministic and comparable.
- **EOS.** The orchestrator's `--prompt` path stops at the model's end-of-sequence id(s)
  (a narrow set — only true EOS, not every special token). The parity/reference paths run a
  fixed `max_new_tokens` with no early stop, for exact comparison.
- **Prefill then decode.** The whole prompt is processed once (prefill, positions `0…S-1`),
  then one token at a time (decode, `S, S+1, …`). Each shard keeps its own KV cache across
  steps.

---

## 10 · The wire & sessions

- **Contract (`drift/protocol.py`, frozen):** every message is a 4-byte big-endian length
  prefix + a msgpack dict. Any runtime that implements this framing can be a node — there is
  no PyTorch on the wire.
- **What crosses:** only `hidden_states` (fp16) + `position_ids` + `input_ids`. The **KV cache
  never crosses** — each shard keeps its own. Per-token traffic is `hidden_size × 2` bytes plus
  a few ints (a couple of KB), independent of parameter count.
- **Sessions.** A generation is a `session_id` (default `s0`). Each shard holds a per-session
  KV cache; the orchestrator sends a `reset` when a generation ends. A shard server handles
  **one connection at a time** (sequential; concurrency is future work) — so don't point two
  orchestrators at one shard simultaneously.
- **TCP tuning.** Connections set `TCP_NODELAY`; the servers set `SO_REUSEADDR`.

---

## 11 · Memory

Plan for the **full model in RAM/VRAM on every node** today: each shard server currently loads
the whole checkpoint and then uses only its layer slice, and the orchestrator loads the model
too (for embed/norm/head). The *active* parameters per node are smaller — the heaviest node's
own layers are ~42% of the model for the default 2-way split (see
[`benchmarks.md`](benchmarks.md)) — but trimming the on-disk load to just the slice is future
work. Until then:

- Use a model that fits each node's memory, or split across **more** nodes to shrink each
  node's *active* share.
- On a memory-tight Mac, `python -m drift.bench --no-socket` skips spawning extra full-model
  server processes.

---

## 12 · Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `ConnectionRefusedError` from the orchestrator | Shard not up yet, or wrong `host`/`port`. Start the server first; confirm it printed `listening on …`; check the port matches. |
| Works on localhost, not across machines | Server bound to `127.0.0.1`. Restart it with `--host 0.0.0.0` and open the firewall port. |
| `--ping` fails for one shard | That shard's process died or its port/host is wrong. Re-check its `--start/--end/--device` and that the model loaded. |
| Parity **FAIL at token 1–2** | A real bug (mask/KV/RoPE), not float noise — the split logic diverged. |
| Greedy output drifts only in **late** tokens, MPS↔CUDA | Expected vendor fp16 rounding (§5). Not a bug. |
| Out of memory on load | Each process loads the full checkpoint (§11). Use a smaller model, more shards, or `--no-socket` for the bench. |
| `unsupported wire dtype` | `dtype` must be `float16` or `float32` (§8). |
| Rare MPS op error on Mac | Ensure `export PYTORCH_ENABLE_MPS_FALLBACK=1` is set in the shell that launched the process. |
| A shard seems to hang after a health check | A stray second connection to a one-at-a-time server. Use a single orchestrator; reuse its connection (§10). |

---

Reproduce the published numbers with `python -m drift.bench`; methodology is in
[`benchmarks.md`](benchmarks.md).
