# OpenAI Compatibility Matrix

DRIFT's HTTP server is a text-generation OpenAI-compatible surface over the
existing DRIFT orchestrator. It does not replace the node-to-node TCP+msgpack
wire protocol.

## Supported HTTP Surface

| Endpoint | Status | Notes |
|---|---:|---|
| `GET /v1/models` | Supported | Includes DRIFT capability metadata. |
| `POST /v1/chat/completions` | Supported | Non-streaming and SSE streaming. |
| `POST /v1/completions` | Supported | Legacy prompt API, non-streaming and SSE streaming. |
| `POST /v1/responses` | Minimal | Text response shape only; no tools/audio/multimodal. |
| `POST /v1/embeddings` | Supported where possible | Non-thin mode pools final hidden state; thin mode returns capability error. |
| `POST /v1/chat/completions/input_tokens` | Supported | Token count helper for chat messages. |
| `POST /tokenize`, `/detokenize` | Supported | llama.cpp-style helpers, with `/v1/` aliases. |
| `GET /health`, `/ready`, `/metrics` | Supported | Health, readiness/capability metadata, Prometheus-style metrics. |

## Supported Parameters

| Area | Status |
|---|---|
| `model`, `messages`, `prompt`, `input` | Supported. |
| `stream`, `stream_options.include_usage` | Supported, including `[DONE]` and usage-only final chunks. |
| `max_tokens`, `max_completion_tokens`, `max_output_tokens` | Supported. |
| `temperature`, `top_p`, `top_k`, `min_p`, `seed` | Supported in non-thin mode. |
| `presence_penalty`, `frequency_penalty`, `repetition_penalty` | Supported in non-thin mode. |
| `stop` strings | Supported for non-streaming and streaming text chunks. |
| `stop_token_ids` | Supported for non-streaming generation. |
| `encoding_format=float/base64` for embeddings | Supported. |

## Explicitly Unsupported

These requests return OpenAI-shaped errors rather than being silently ignored.

| Feature | Current behavior |
|---|---|
| Tools / function calling / `tool_choice` / `parallel_tool_calls` | Unsupported error. |
| Assistant `tool_calls` message content | Unsupported error. |
| JSON mode / JSON schema via `response_format` | Unsupported error except `{"type":"text"}`. |
| Multimodal image/audio content | Unsupported error. |
| Audio transcription/translation APIs | Not exposed. |
| Embedding `dimensions` truncation/projection | Unsupported error. |
| Thin-head sampling or embeddings | Capability error; thin mode does not return logits/hidden states to the head. |

## Verification

Local no-model API contract tests:

```bash
python3 -m pytest \
  tests/test_openai_api.py \
  tests/test_openai_compat_payloads.py \
  tests/test_openai_sdk_smoke.py \
  tests/test_sampling_controls.py -q
```

OpenAI Python SDK smoke with an isolated venv:

```bash
python3 -m venv /tmp/drift-openai-sdk-venv
/tmp/drift-openai-sdk-venv/bin/python -m pip install openai pytest starlette uvicorn httpx
PYTHONPATH=$PWD /tmp/drift-openai-sdk-venv/bin/python -m pytest tests/test_openai_sdk_smoke.py -q
```

Smoke a running DRIFT server:

```bash
python scripts/openai_compat_smoke.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

Existing DRIFT preservation gates should still be run separately on hardware
that has the required PyTorch/MPS/CUDA stack:

```bash
python3 -m drift.cli help
python3 -m drift.cli run --help
python3 -m drift.cli node --help
python3 -m drift.cli serve --help
python3 -m drift.parity_test --mode inprocess
python3 -m drift.parity_test --mode socket
python3 -m drift.itest --nodes 2
python3 -m drift.itest --nodes 2 --chain
python3 -m drift.itest --nodes 2 --thin
python3 -m drift.itest --nodes 2 --int8
```
