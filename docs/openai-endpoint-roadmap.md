# OpenAI-Compatible Endpoint Roadmap

This branch tracks the work needed to make DRIFT usable as a broad
OpenAI-compatible local/distributed inference server while preserving the
existing `drift node`, `drift run`, `drift up`, wire protocol, receipts, ledger,
failover, parity, and benchmark workflows.

The release cadence for this branch is one pushed commit and one GitHub release
per completed stage. Patch follow-ups after a stage use the same stage number
with a minor suffix.

## Stage 1 - Server Foundation

- Add a `drift serve` command that starts an HTTP server without changing the
  DRIFT node-to-node TCP+msgpack data plane.
- Build a long-lived OpenAI API app around a reusable DRIFT orchestrator.
- Support `GET /health`, `GET /v1/models`, and a minimal
  `POST /v1/chat/completions`.
- Support both non-streaming responses and OpenAI-style SSE streaming chunks.
- Serialize generation through a per-backend lock so session state, sockets,
  sequence numbers, receipts, and KV resets do not interleave.
- Add tests with a fake backend so the API contract is checked without loading
  model weights.
- Release tag: `openai-endpoint-stage-01`.

## Stage 2 - Chat Semantics And Error Fidelity

- Normalize OpenAI `messages` content variants: strings, text parts, image parts
  with explicit unsupported errors, tool messages, and assistant prefixes.
- Preserve system/developer/user/assistant/tool role ordering in prompt
  rendering while still falling back cleanly for tokenizers without chat
  templates.
- Return OpenAI-shaped error objects with stable HTTP status codes.
- Validate `model`, `messages`, `max_tokens`, `stream`, `n`, `user`, and unknown
  fields consistently.
- Add request IDs and response metadata matching common OpenAI SDK expectations.
- Release tag: `openai-endpoint-stage-02`.

## Stage 3 - Completions API

- Add `POST /v1/completions` for legacy OpenAI-compatible clients.
- Accept single prompt, prompt arrays, and token-id prompt arrays where practical.
- Return `choices[].text`, `finish_reason`, `usage`, and deterministic IDs.
- Stream completion chunks with the OpenAI SSE shape.
- Share generation limits, locking, usage accounting, and error handling with
  chat completions.
- Release tag: `openai-endpoint-stage-03`.

## Stage 4 - Streaming Robustness

- Match OpenAI streaming details closely enough for the official OpenAI Python
  SDK, LangChain, LiteLLM, and curl clients.
- Emit initial role chunks, delta chunks, final finish chunks, optional usage
  chunks, and `[DONE]`.
- Handle client disconnects and cancellation without leaking DRIFT sessions.
- Add timeout handling and clear node-unavailable errors.
- Add SDK smoke tests for streaming and non-streaming calls.
- Release tag: `openai-endpoint-stage-04`.

## Stage 5 - Generation Controls

- Extend the orchestrator beyond greedy-only decoding where the architecture can
  support it.
- Implement or explicitly gate `temperature`, `top_p`, `top_k`, `min_p`,
  `frequency_penalty`, `presence_penalty`, `repetition_penalty`, `seed`, and
  `stop`.
- Preserve exact greedy behavior when sampling parameters are omitted.
- Define thin-head limitations clearly, since thin mode currently returns token
  IDs from the tail rather than logits to the head.
- Add parity tests proving existing greedy DRIFT behavior is unchanged.
- Release tag: `openai-endpoint-stage-05`.

## Stage 6 - Tokenization, Context, And Usage Accounting

- Add llama.cpp-style tokenizer helper endpoints where useful:
  `/tokenize`, `/detokenize`, and `/v1/tokenize` aliases if compatible clients
  expect them.
- Calculate prompt, completion, and total token usage consistently for chat and
  completion endpoints.
- Enforce model context limits and return OpenAI-shaped context-length errors.
- Support stop strings and stop token IDs without corrupting UTF-8 streaming.
- Release tag: `openai-endpoint-stage-06`.

## Stage 7 - Embeddings And Model Capabilities

- Add `POST /v1/embeddings` with capability detection.
- For causal LMs, either provide a documented pooled-hidden-state embedding mode
  or return a precise unsupported-model error when that would be misleading.
- Expose model capability metadata so clients can discover chat/completion versus
  embedding support.
- Add optional normalized embedding outputs and `encoding_format` handling.
- Release tag: `openai-endpoint-stage-07`.

## Stage 8 - Production Server Controls

- Add API-key bearer auth, CORS controls, host/port configuration, access logs,
  health/readiness probes, and graceful shutdown.
- Add queueing/concurrency controls, request cancellation, and per-request
  session IDs.
- Keep node wire security (`DRIFT_NETWORK_KEY`) separate from HTTP client auth.
- Add configuration through CLI flags, environment variables, and `config.yaml`.
- Release tag: `openai-endpoint-stage-08`.

## Stage 9 - Compatibility Harness

- Add an automated compatibility suite covering:
  OpenAI Python SDK, curl, LangChain, LiteLLM, llama.cpp-like clients, and
  vLLM-like request payloads.
- Snapshot representative response bodies and streaming chunk sequences.
- Add negative tests for unsupported tools, multimodal input, embeddings on
  unsupported models, bad auth, invalid model IDs, and context overflow.
- Run existing DRIFT gates beside the new HTTP tests.
- Release tag: `openai-endpoint-stage-09`.

## Stage 10 - Preservation Audit And PR

- Re-run and document the existing DRIFT verification matrix:
  `doctor`, `parity`, `itest`, receipts/ledger, chain, thin, int8, secure wire,
  and benchmark smoke checks where hardware permits.
- Confirm the OpenAI server does not alter the frozen TCP+msgpack protocol or
  existing CLI behavior.
- Update README and operations manuals in supported languages.
- Cut the final branch release and open a PR without merging.
- Release tag: `openai-endpoint-stage-10`.
