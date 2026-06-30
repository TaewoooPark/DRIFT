# 06 — M0 Setup Runbook (copy-paste)

The exact commands to reach **M0** (spec §9). Run the **Mac** track now (M0a); the
**Windows** track (M0b) is deferred until M4. Canonical values (Python 3.12, port 52600,
model id) are defined in [`03`](03-goal-execution-plan.md).

> **Why Python 3.12, not the system 3.14:** this Mac runs Python 3.14, for which PyTorch
> does not publish wheels. Pin 3.12 in an isolated venv via `uv`.

---

## Mac (M0a) — node `mac`, device `mps`, layers 0–8

```bash
cd /Users/taewoopark/personal/DRIFT

# 1. Isolated env on Python 3.12 (uv is already installed)
uv venv --python 3.12 .venv
source .venv/bin/activate
python --version            # expect: Python 3.12.x

# 2. Deps. torch's default macOS arm64 wheel includes MPS — no special index needed.
uv pip install torch transformers safetensors msgpack numpy huggingface_hub "accelerate"

# 3. Sanity: MPS must be available
python -c "import torch; print('mps', torch.backends.mps.is_available())"   # expect: mps True

# 4. Auth for the gated Llama model (skip if using Qwen fallback)
huggingface-cli login        # paste an HF token that has accepted the Llama 3.2 license

# 5. Verify layer count → this fixes the split point (03 decision #2)
python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B-Instruct').num_hidden_layers)"
# expect: 16   (if gating blocks you, use Qwen/Qwen2.5-1.5B-Instruct -> 28)

# 6. Freeze the lock — this exact transformers version must match Windows later
uv pip freeze > requirements.lock

# 7. Run shells need the MPS CPU-fallback flag
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

**Fallback to Qwen** (if Llama gating isn't cleared in ~15 min): set `model_id:
"Qwen/Qwen2.5-1.5B-Instruct"` in `config.yaml`, split `0–14`/`14–28`. No `huggingface-cli
login` needed.

---

## Windows (M0b) — node `windows`, device `cuda`, layers 8–16 (defer to M4)

```powershell
cd <repo>
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# CUDA wheel — pick the index matching the installed driver (example: cu124)
pip install torch --index-url https://download.pytorch.org/whl/cu124
# transformers/msgpack/etc. MUST match the Mac lock exactly (parity-critical)
pip install transformers safetensors msgpack numpy huggingface_hub accelerate

python -c "import torch; print('cuda', torch.cuda.is_available())"   # expect: cuda True
huggingface-cli login        # same gated-model note
pip freeze > requirements.win.lock
```

> **Version parity is non-negotiable.** `torch` differs by device (MPS vs CUDA) — fine.
> But `transformers` (and msgpack) **must be identical** on both nodes, or HF internals
> introspect differently and parity (M2/M3 logic carried to M4) breaks. The
> `drift-env-introspect` skill ([`04`](04-skills-mcp-plan.md)) diffs the two locks.

---

## M0 acceptance — localhost two-port ping

On the Mac, before any Windows node exists, prove the neutral protocol end-to-end by
running two shard servers locally on different ports and pinging both:

```bash
# terminal A
DRIFT_PORT=52600 python -m drift.shard_server --name mac     --start 0 --end 8  --device mps
# terminal B  (a second local shard to exercise the protocol; device can be cpu for the smoke test)
DRIFT_PORT=52601 python -m drift.shard_server --name shard2  --start 8 --end 16 --device cpu
# terminal C — point the orchestrator at both localhost ports explicitly
python -m drift.orchestrator --ping --ports 52600,52601
```

> **Why `--ports` here:** the `config.yaml` schema (spec §5) carries a single top-level
> `port: 52600` and per-shard `host` only — designed for **M4**, where the two shards live
> on **distinct hosts sharing one port**. The localhost smoke test instead puts two shards
> on one host, so it needs two ports: `shard_server` reads a `DRIFT_PORT` env override and
> the orchestrator takes a `--ports` list (both layered over the config). At M4 you drop
> these overrides and use the plain config (`52600` across the two LAN IPs).

**Pass (§9 M0):** the orchestrator prints a valid `ping` reply
`{ok, name, start_layer, end_layer, device}` from **both** ports.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No matching distribution found for torch` | You're on Python 3.14 — recreate the venv with `--python 3.12`. |
| `mps False` | On Intel Mac or non-arm64 Python; use an arm64 Python 3.12. |
| `401/403` pulling Llama | Token hasn't accepted the Llama 3.2 license, or wrong token — accept on the HF model page, or switch to Qwen. |
| MPS op error mid-run | Ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` is exported in the run shell. |
| Port already in use | Pick another port; localhost uses `52600`/`52601`. |
