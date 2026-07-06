# OpenAI Compatibility Audit

This audit maps the review checklist for a vLLM/llama.cpp-like OpenAI surface to
the current DRIFT PR. DRIFT's claim is deliberately scoped to text-generation
OpenAI compatibility over the existing distributed DRIFT orchestrator.

## Verified In This PR

| Checklist area | Evidence |
|---|---|
| Core endpoints | `tests/test_openai_api.py` and `tests/test_openai_compat_payloads.py` cover `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/chat/completions/input_tokens`, `/tokenize`, `/detokenize`, `/health`, `/ready`, and `/metrics`. |
| OpenAI Python SDK | `tests/test_openai_sdk_smoke.py` starts a real uvicorn server and exercises models, chat, streaming chat, completions, embeddings, Responses, streaming Responses, JSON mode, tools, `n`, and logprobs through the official SDK. |
| OpenAI JS SDK | `tests/test_openai_js_sdk_smoke.py` is an opt-in smoke (`DRIFT_RUN_JS_SDK_SMOKE=1`) that installs the official JS SDK and exercises models, chat, streaming chat, completions, embeddings, Responses, and streaming Responses. |
| Chat parameters | Tests cover `model`, `messages`, `stream`, `stream_options.include_usage`, `max_tokens`, `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `stop`, `stop_token_ids`, `n`, `seed`, presence/frequency/repetition penalties, `logprobs`, `top_logprobs`, `response_format`, `tools`, `tool_choice`, `parallel_tool_calls`, and `user`. |
| Message forms | Tests cover `system`, `developer`, `user`, `assistant`, and `tool` roles, string content, text content arrays, assistant `tool_calls`, and explicit multimodal unsupported errors. |
| SSE streaming | Tests cover `text/event-stream`, `data: ...\n\n`, assistant role first chunks, delta content chunks, finish reasons, `[DONE]`, usage-only final chunks, stop truncation, and official SDK iteration. Responses streaming uses typed events. |
| Response and error shape | Tests cover OpenAI-shaped chat/completion/Responses payloads, usage fields, logprobs fields, request IDs, authentication errors, invalid model errors, invalid JSON, unknown parameters, invalid tool/format parameters, and context overflow. |
| Sampling and stopping | `tests/test_sampling_controls.py` covers greedy determinism, seeded sampling repeatability, and penalties. API tests cover option propagation, stop strings, stop token ids, and max-token finish reasons. |
| Tool calling and JSON | Tests cover `tool_choice=auto`, `tool_choice=required`, specific forced tools, parsed model-emitted tool-call JSON, legacy `functions`/`function_call`, JSON object mode, and JSON schema compatibility coercion. |
| Embeddings | Tests cover string input, string-array input, token-id input, batch order, fixed dimension, float/base64 encoding, usage, empty input errors, context overflow, unsupported dimensions, and thin/unsupported capability errors. |
| Operations | Tests cover CORS, Bearer and `x-api-key` auth, malformed JSON, context overflow, request IDs, Prometheus-style metrics, and concurrent request session isolation. `scripts/openai_compat_smoke.py` exercises the main surface against a running `drift serve`. |
| DRIFT preservation smoke | CLI help gates are dependency-light. The OpenAI adapter calls the existing orchestrator and does not change the TCP+msgpack node protocol. |

## Scoped Unsupported

| Feature | Reason |
|---|---|
| Audio transcription/translation | DRIFT currently exposes a text-generation compatibility server. Audio APIs are not claimed. |
| Multimodal image/audio chat content | The endpoint returns explicit OpenAI-shaped unsupported errors. |
| Tool execution | DRIFT returns OpenAI-shaped tool calls for the client to execute; it does not run tools server-side. |
| Full constrained decoding for JSON schema | The compatibility layer extracts/coerces JSON and fills simple required fields; it does not guarantee schema-constrained decoding. |
| Embedding quality claims for causal LMs | The endpoint exposes a documented pooled-hidden-state embedding mode where the head can access hidden states; it does not claim a dedicated embedding model. |

## Still Requires External Full-Stack Evidence

These are not disproven, but they are not fully proven by the local no-model test
environment:

| Area | Needed evidence |
|---|---|
| LangChain, LiteLLM, Open WebUI, LlamaIndex, AutoGen | Install each client, point it at `drift serve`, and run smoke flows against a real server. |
| Large concurrency and long-running memory behavior | Run 10/50/100-client load, cancellation, timeout, keep-alive, and leak checks against a model-backed server. |
| Full DRIFT parity/itest matrix | Run `python3 -m drift.parity_test --mode inprocess`, `--mode socket`, and `python3 -m drift.itest` variants on a machine with the required PyTorch/model stack. |
| Secure wire, receipt verifier, ledger, failover | Run the existing DRIFT integration tests with `DRIFT_NETWORK_KEY`, receipt verification, ledger verification, and failover scenarios on hardware that can load the configured model. |
