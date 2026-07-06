"""OpenAI-compatible HTTP surface for DRIFT.

This module is intentionally an adapter: it does not replace DRIFT's node
transport. The HTTP server accepts OpenAI-shaped requests at the edge, then
calls the existing orchestrator, which still talks to nodes over the frozen
TCP+msgpack protocol.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Iterable, Protocol


Prompt = str | list[dict]


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int] | None = None
    finish_reason: str = "stop"


class OpenAIBackend(Protocol):
    model_id: str
    default_max_tokens: int
    context_length: int | None
    supports_embeddings: bool

    def generate(self, prompt: Prompt, max_tokens: int, session_id: str,
                 options: dict | None = None) -> GenerationResult:
        ...

    def stream(self, prompt: Prompt, max_tokens: int, session_id: str,
               options: dict | None = None) -> Iterable[str]:
        ...

    def count_tokens(self, prompt: Prompt) -> int:
        ...

    def decode_tokens(self, token_ids: list[int]) -> str:
        ...

    def encode_tokens(self, prompt: Prompt) -> list[int]:
        ...

    def embed(self, prompt: Prompt, session_id: str) -> list[float]:
        ...


class DriftBackend:
    """Long-lived bridge from HTTP requests to one DRIFT orchestrator.

    The lock is deliberately coarse for stage 1. The underlying transport keeps
    session, socket, receipt, and sequence state, so the first compatibility
    layer serializes generation rather than letting concurrent requests
    interleave shared state.
    """

    def __init__(self, orchestrator, model_id: str, default_max_tokens: int = 128):
        self.orchestrator = orchestrator
        self.model_id = model_id
        self.default_max_tokens = default_max_tokens
        self.supports_sampling = not getattr(orchestrator, "thin", False)
        self.supports_embeddings = not getattr(orchestrator, "thin", False)
        self.context_length = self._context_length()
        self._lock = threading.Lock()

    def _context_length(self) -> int | None:
        cfg = getattr(getattr(self.orchestrator.head, "lm", None), "config", None)
        for name in ("max_position_embeddings", "seq_length", "n_positions"):
            value = getattr(cfg, name, None)
            if isinstance(value, int) and value > 0:
                return value
        value = getattr(self.orchestrator.head.tokenizer, "model_max_length", None)
        if isinstance(value, int) and 0 < value < 1_000_000_000:
            return value
        return None

    def generate(self, prompt: Prompt, max_tokens: int, session_id: str,
                 options: dict | None = None) -> GenerationResult:
        with self._lock:
            out = self.orchestrator.generate(
                prompt, max_tokens, stop_on_eos=True, session_id=session_id,
                generation_options=options,
            )
        token_ids = list(out.get("token_ids") or [])
        finish = "length" if len(token_ids) >= max_tokens else "stop"
        return GenerationResult(text=out.get("text", ""), token_ids=token_ids,
                                finish_reason=finish)

    def stream(self, prompt: Prompt, max_tokens: int, session_id: str,
               options: dict | None = None) -> Iterable[str]:
        with self._lock:
            yield from self.orchestrator.generate_stream(
                prompt, max_tokens, stop_on_eos=True, session_id=session_id,
                generation_options=options,
            )

    def count_tokens(self, prompt: Prompt) -> int:
        from .common import build_input_ids

        tok = self.orchestrator.head.tokenizer
        try:
            ids = build_input_ids(tok, prompt)
        except Exception:
            return 0
        try:
            return int(ids.shape[-1])
        except Exception:
            pass
        if ids and isinstance(ids[0], list):
            return len(ids[0])
        return len(ids or [])

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.orchestrator.head.tokenizer.decode(token_ids)

    def encode_tokens(self, prompt: Prompt) -> list[int]:
        from .common import build_input_ids

        ids = build_input_ids(self.orchestrator.head.tokenizer, prompt)
        try:
            return [int(x) for x in ids[0].tolist()]
        except Exception:
            return [int(x) for x in ids]

    def embed(self, prompt: Prompt, session_id: str) -> list[float]:
        if not self.supports_embeddings:
            raise ValueError("embeddings are not available in thin-head mode")
        from .common import build_input_ids

        import torch

        with self._lock, torch.no_grad():
            try:
                input_ids = build_input_ids(self.orchestrator.head.tokenizer, prompt).to(
                    self.orchestrator.device
                )
                prompt_ids = input_ids[0].tolist()
                hidden = self.orchestrator.head.embed(input_ids)
                hidden = self.orchestrator.transport.route(
                    self.orchestrator.order, session_id, hidden,
                    list(range(len(prompt_ids))), prompt_ids, "prefill"
                ).to(self.orchestrator.device)
                pooled = self.orchestrator.head.norm(hidden)[:, -1, :][0]
                vector = pooled.detach().float().cpu().tolist()
            finally:
                for name in self.orchestrator.order:
                    try:
                        self.orchestrator.transport.reset(name, session_id)
                    except Exception:
                        pass
            return [float(x) for x in vector]


class OpenAIHTTPError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        type_: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.type = type_
        self.param = param
        self.code = code


def _require_starlette():
    try:
        from starlette.applications import Starlette
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import JSONResponse, StreamingResponse
        from starlette.routing import Route
    except ImportError as e:
        raise RuntimeError(
            "drift serve requires the HTTP extras: install starlette and uvicorn"
        ) from e
    return Starlette, BaseHTTPMiddleware, Request, JSONResponse, StreamingResponse, Route


def _now() -> int:
    return int(time.time())


def _make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _json_sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'), ensure_ascii=False)}\n\n"


def _content_to_text(content, param: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for i, part in enumerate(content):
            if not isinstance(part, dict):
                raise OpenAIHTTPError(400, f"{param}[{i}] must be an object", param=param)
            ptype = part.get("type")
            if ptype in ("text", "input_text"):
                parts.append(str(part.get("text") or ""))
            elif ptype in ("image_url", "input_image"):
                raise OpenAIHTTPError(
                    400,
                    "multimodal chat content is not supported by this DRIFT endpoint yet",
                    param=param,
                    code="unsupported_multimodal_content",
                )
            else:
                raise OpenAIHTTPError(
                    400, f"unsupported content part type: {ptype!r}", param=param
                )
        return "\n".join(p for p in parts if p)
    raise OpenAIHTTPError(400, f"{param} must be a string or content-part array", param=param)


def normalize_chat_messages(messages: list[dict]) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise OpenAIHTTPError(400, "`messages` must be a non-empty array", param="messages")
    normalized: list[dict] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise OpenAIHTTPError(400, f"messages[{i}] must be an object",
                                  param=f"messages[{i}]")
        role = msg.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise OpenAIHTTPError(400, f"unsupported message role: {role!r}",
                                  param=f"messages[{i}].role")
        if msg.get("tool_calls"):
            raise OpenAIHTTPError(
                400,
                "assistant tool calls are not supported by this DRIFT endpoint yet",
                param=f"messages[{i}].tool_calls",
                code="unsupported_tools",
            )
        text = _content_to_text(msg.get("content"), f"messages[{i}].content")
        if role == "developer":
            role = "system"
        normalized.append({"role": role, "content": text})
    return normalized


def render_chat_prompt(messages: list[dict]) -> str:
    lines: list[str] = []
    for msg in normalize_chat_messages(messages):
        text = msg.get("content") or ""
        if text:
            lines.append(f"{msg['role'].capitalize()}: {text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _validate_model(body: dict, backend: OpenAIBackend) -> str:
    model = body.get("model") or backend.model_id
    if model != backend.model_id:
        raise OpenAIHTTPError(
            404,
            f"model {model!r} is not served by this DRIFT endpoint",
            type_="invalid_request_error",
            param="model",
            code="model_not_found",
        )
    return model


def _validate_max_tokens(body: dict, backend: OpenAIBackend) -> int:
    raw = body.get("max_tokens", body.get("max_completion_tokens", backend.default_max_tokens))
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise OpenAIHTTPError(400, "`max_tokens` must be an integer", param="max_tokens")
    if n < 1:
        raise OpenAIHTTPError(400, "`max_tokens` must be >= 1", param="max_tokens")
    return n


def _validate_stage1_options(body: dict) -> None:
    known = {
        "model", "messages", "max_tokens", "max_completion_tokens", "stream",
        "stream_options", "temperature", "top_p", "top_k", "min_p", "n", "stop",
        "stop_token_ids",
        "presence_penalty", "frequency_penalty", "repetition_penalty", "seed", "user", "tools",
        "tool_choice", "functions", "function_call", "logprobs", "top_logprobs",
        "response_format", "logit_bias",
    }
    unknown = sorted(set(body) - known)
    if unknown:
        raise OpenAIHTTPError(
            400,
            f"unsupported request field(s): {', '.join(unknown)}",
            code="unsupported_parameter",
        )
    try:
        n = int(body.get("n", 1))
    except (TypeError, ValueError):
        raise OpenAIHTTPError(400, "`n` must be an integer", param="n")
    if n != 1:
        raise OpenAIHTTPError(400, "only n=1 is supported in this stage", param="n")
    if body.get("logit_bias") not in (None, {}):
        raise OpenAIHTTPError(
            400, "`logit_bias` is not supported by this endpoint stage yet",
            param="logit_bias", code="unsupported_parameter"
        )
    response_format = body.get("response_format")
    if response_format not in (None, {}):
        if not isinstance(response_format, dict):
            raise OpenAIHTTPError(400, "`response_format` must be an object",
                                  param="response_format")
        if response_format.get("type", "text") != "text":
            raise OpenAIHTTPError(
                400,
                "only text response_format is supported by this endpoint stage",
                param="response_format",
                code="unsupported_parameter",
            )
    unsupported = [
        "tools", "tool_choice", "functions", "function_call", "logprobs",
        "top_logprobs",
    ]
    for key in unsupported:
        if key in body and body[key] not in (None, [], {}, "none"):
            raise OpenAIHTTPError(
                400, f"`{key}` is not supported by this endpoint stage yet",
                param=key, code="unsupported_parameter"
            )


def _include_stream_usage(body: dict) -> bool:
    opts = body.get("stream_options") or {}
    if not isinstance(opts, dict):
        raise OpenAIHTTPError(400, "`stream_options` must be an object",
                              param="stream_options")
    return bool(opts.get("include_usage"))


def _float_param(body: dict, key: str, default: float) -> float:
    if body.get(key) is None:
        return default
    try:
        return float(body[key])
    except (TypeError, ValueError):
        raise OpenAIHTTPError(400, f"`{key}` must be a number", param=key)


def _int_param(body: dict, key: str, default: int) -> int:
    if body.get(key) is None:
        return default
    try:
        return int(body[key])
    except (TypeError, ValueError):
        raise OpenAIHTTPError(400, f"`{key}` must be an integer", param=key)


def _generation_options(body: dict, backend: OpenAIBackend) -> dict:
    opts = {
        "temperature": _float_param(body, "temperature", 0.0),
        "top_p": _float_param(body, "top_p", 1.0),
        "top_k": _int_param(body, "top_k", 0),
        "min_p": _float_param(body, "min_p", 0.0),
        "presence_penalty": _float_param(body, "presence_penalty", 0.0),
        "frequency_penalty": _float_param(body, "frequency_penalty", 0.0),
        "repetition_penalty": _float_param(body, "repetition_penalty", 1.0),
        "seed": None if body.get("seed") is None else _int_param(body, "seed", 0),
    }
    if opts["temperature"] < 0:
        raise OpenAIHTTPError(400, "`temperature` must be >= 0", param="temperature")
    if not 0 < opts["top_p"] <= 1:
        raise OpenAIHTTPError(400, "`top_p` must be > 0 and <= 1", param="top_p")
    if opts["top_k"] < 0:
        raise OpenAIHTTPError(400, "`top_k` must be >= 0", param="top_k")
    if not 0 <= opts["min_p"] <= 1:
        raise OpenAIHTTPError(400, "`min_p` must be >= 0 and <= 1", param="min_p")
    for key in ("presence_penalty", "frequency_penalty"):
        if not -2 <= opts[key] <= 2:
            raise OpenAIHTTPError(400, f"`{key}` must be between -2 and 2", param=key)
    if opts["repetition_penalty"] <= 0:
        raise OpenAIHTTPError(
            400, "`repetition_penalty` must be > 0", param="repetition_penalty"
        )

    non_greedy = (
        opts["temperature"] > 0
        or opts["top_p"] < 1
        or opts["top_k"] > 0
        or opts["min_p"] > 0
        or opts["presence_penalty"] != 0
        or opts["frequency_penalty"] != 0
        or opts["repetition_penalty"] != 1
    )
    if non_greedy and not getattr(backend, "supports_sampling", True):
        raise OpenAIHTTPError(
            400,
            "sampling parameters are not available in DRIFT thin-head mode",
            code="unsupported_sampling",
        )
    return opts


def _stop_strings(body: dict) -> list[str]:
    stop = body.get("stop")
    if stop in (None, ""):
        return []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list) and all(isinstance(x, str) for x in stop):
        return [x for x in stop if x]
    raise OpenAIHTTPError(400, "`stop` must be a string or string array", param="stop")


def _stop_token_ids(body: dict) -> list[int]:
    raw = body.get("stop_token_ids")
    if raw in (None, []):
        return []
    if not isinstance(raw, list) or not all(isinstance(x, int) for x in raw):
        raise OpenAIHTTPError(
            400, "`stop_token_ids` must be an integer array", param="stop_token_ids"
        )
    return [int(x) for x in raw]


def _check_context(backend: OpenAIBackend, prompt: Prompt, max_tokens: int) -> None:
    limit = getattr(backend, "context_length", None)
    if not limit:
        return
    prompt_tokens = backend.count_tokens(prompt)
    if prompt_tokens + max_tokens > int(limit):
        raise OpenAIHTTPError(
            400,
            f"requested tokens exceed model context window ({prompt_tokens}+{max_tokens}>{limit})",
            code="context_length_exceeded",
        )


def _apply_stops(result: GenerationResult, backend: OpenAIBackend,
                 stop_strings: list[str], stop_token_ids: list[int]) -> GenerationResult:
    text = result.text
    token_ids = list(result.token_ids) if result.token_ids is not None else None
    finish = result.finish_reason

    if token_ids is not None and stop_token_ids:
        stop_set = set(stop_token_ids)
        for i, tid in enumerate(token_ids):
            if tid in stop_set:
                token_ids = token_ids[:i]
                text = backend.decode_tokens(token_ids)
                finish = "stop"
                break

    if stop_strings:
        found = [(text.find(s), s) for s in stop_strings if s and text.find(s) >= 0]
        if found:
            idx, _ = min(found, key=lambda x: x[0])
            text = text[:idx]
            finish = "stop"

    return GenerationResult(text=text, token_ids=token_ids, finish_reason=finish)


def _split_on_stop(text: str, stop_strings: list[str]) -> tuple[str, bool]:
    if not stop_strings:
        return text, False
    found = [(text.find(s), s) for s in stop_strings if s and text.find(s) >= 0]
    if not found:
        return text, False
    idx, _ = min(found, key=lambda x: x[0])
    return text[:idx], True


def _validate_completion_options(body: dict) -> None:
    known = {
        "model", "prompt", "suffix", "max_tokens", "temperature", "top_p",
        "top_k", "min_p", "n", "stream", "stream_options", "logprobs", "echo",
        "stop", "stop_token_ids", "presence_penalty", "frequency_penalty", "repetition_penalty",
        "best_of", "logit_bias", "user", "seed",
    }
    unknown = sorted(set(body) - known)
    if unknown:
        raise OpenAIHTTPError(
            400,
            f"unsupported request field(s): {', '.join(unknown)}",
            code="unsupported_parameter",
        )
    _validate_stage1_options({
        k: v for k, v in body.items()
        if k not in {"prompt", "suffix", "echo", "best_of"}
    })
    if body.get("suffix") not in (None, ""):
        raise OpenAIHTTPError(
            400, "`suffix` is not supported by this endpoint stage yet",
            param="suffix", code="unsupported_parameter"
        )
    if body.get("best_of") not in (None, 1):
        raise OpenAIHTTPError(
            400, "`best_of` is not supported by this endpoint stage yet",
            param="best_of", code="unsupported_parameter"
        )


def _decode_token_prompt(backend: OpenAIBackend, item, param: str) -> str:
    if not isinstance(item, list) or not all(isinstance(x, int) for x in item):
        raise OpenAIHTTPError(400, f"{param} must be a string or token id array",
                              param=param)
    return backend.decode_tokens([int(x) for x in item])


def normalize_completion_prompts(body: dict, backend: OpenAIBackend) -> list[str]:
    if "prompt" not in body:
        raise OpenAIHTTPError(400, "`prompt` is required", param="prompt")
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list):
        if not prompt:
            raise OpenAIHTTPError(400, "`prompt` array must not be empty", param="prompt")
        if all(isinstance(x, str) for x in prompt):
            return list(prompt)
        if all(isinstance(x, int) for x in prompt):
            return [_decode_token_prompt(backend, prompt, "prompt")]
        out: list[str] = []
        for i, item in enumerate(prompt):
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, list):
                out.append(_decode_token_prompt(backend, item, f"prompt[{i}]"))
            else:
                raise OpenAIHTTPError(
                    400,
                    "prompt array entries must be strings or token id arrays",
                    param=f"prompt[{i}]",
                )
        return out
    raise OpenAIHTTPError(400, "`prompt` must be a string or array", param="prompt")


def normalize_embedding_inputs(body: dict, backend: OpenAIBackend) -> list[Prompt]:
    if "input" not in body:
        raise OpenAIHTTPError(400, "`input` is required", param="input")
    raw = body.get("input")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        if not raw:
            raise OpenAIHTTPError(400, "`input` array must not be empty", param="input")
        if all(isinstance(x, str) for x in raw):
            return list(raw)
        if all(isinstance(x, int) for x in raw):
            return [backend.decode_tokens([int(x) for x in raw])]
        out: list[Prompt] = []
        for i, item in enumerate(raw):
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, list) and all(isinstance(x, int) for x in item):
                out.append(backend.decode_tokens([int(x) for x in item]))
            else:
                raise OpenAIHTTPError(
                    400,
                    "input array entries must be strings or token id arrays",
                    param=f"input[{i}]",
                )
        return out
    raise OpenAIHTTPError(400, "`input` must be a string or array", param="input")


def _format_embedding(vector: list[float], encoding_format: str):
    if encoding_format == "float":
        return vector
    if encoding_format == "base64":
        packed = struct.pack(f"<{len(vector)}f", *vector)
        return base64.b64encode(packed).decode("ascii")
    raise OpenAIHTTPError(
        400, "`encoding_format` must be `float` or `base64`",
        param="encoding_format",
    )


def _usage(backend: OpenAIBackend, prompt: str, result: GenerationResult) -> dict:
    completion_tokens = (
        len(result.token_ids) if result.token_ids is not None else backend.count_tokens(result.text)
    )
    prompt_tokens = backend.count_tokens(prompt)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def create_app(backend: OpenAIBackend):
    Starlette, BaseHTTPMiddleware, Request, JSONResponse, StreamingResponse, Route = (
        _require_starlette()
    )

    class RequestIDMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            request_id = request.headers.get("x-request-id") or _make_id("req")
            request.state.request_id = request_id
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response

    async def openai_error_handler(request: Request, exc: OpenAIHTTPError):
        request_id = getattr(request.state, "request_id", _make_id("req"))
        return JSONResponse(
            {
                "error": {
                    "message": exc.message,
                    "type": exc.type,
                    "param": exc.param,
                    "code": exc.code,
                }
            },
            status_code=exc.status_code,
            headers={"x-request-id": request_id},
        )

    async def health(request: Request):
        return JSONResponse({"status": "ok", "model": backend.model_id})

    async def models(request: Request):
        return JSONResponse({
            "object": "list",
            "data": [{
                "id": backend.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "drift",
                "capabilities": {
                    "chat": True,
                    "completion": True,
                    "embedding": bool(getattr(backend, "supports_embeddings", False)),
                },
            }],
        })

    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise OpenAIHTTPError(400, "request body must be valid JSON")
        if not isinstance(body, dict):
            raise OpenAIHTTPError(400, "request body must be a JSON object")
        model = _validate_model(body, backend)
        _validate_stage1_options(body)
        max_tokens = _validate_max_tokens(body, backend)
        options = _generation_options(body, backend)
        stop_strings = _stop_strings(body)
        stop_token_ids = _stop_token_ids(body)
        prompt = normalize_chat_messages(body.get("messages"))
        _check_context(backend, prompt, max_tokens)
        stream = bool(body.get("stream", False))
        include_usage = _include_stream_usage(body)
        created = _now()
        rid = _make_id("chatcmpl")
        session_id = _make_id("openai")

        if not stream:
            result = backend.generate(prompt, max_tokens, session_id, options=options)
            result = _apply_stops(result, backend, stop_strings, stop_token_ids)
            return JSONResponse({
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason,
                }],
                "usage": _usage(backend, prompt, result),
            })

        def events():
            full_text = ""
            first = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"},
                             "finish_reason": None}],
            }
            if include_usage:
                first["usage"] = None
            yield _json_sse(first)
            try:
                for piece in backend.stream(prompt, max_tokens, session_id, options=options):
                    if not piece:
                        continue
                    visible, stopped = _split_on_stop(full_text + piece, stop_strings)
                    piece = visible[len(full_text):]
                    full_text = visible
                    if not piece and not stopped:
                        continue
                    chunk = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": piece},
                                     "finish_reason": None}],
                    }
                    if include_usage:
                        chunk["usage"] = None
                    yield _json_sse(chunk)
                    if stopped:
                        break
                final: dict = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                if include_usage:
                    final["usage"] = None
                yield _json_sse(final)
                if include_usage:
                    yield _json_sse({
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": _usage(
                            backend, prompt, GenerationResult(text=full_text, token_ids=None)
                        ),
                    })
            except GeneratorExit:
                raise
            except Exception as e:
                yield _json_sse({
                    "error": {
                        "message": f"{type(e).__name__}: {e}",
                        "type": "server_error",
                        "param": None,
                        "code": "generation_failed",
                    }
                })
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def completions(request: Request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise OpenAIHTTPError(400, "request body must be valid JSON")
        if not isinstance(body, dict):
            raise OpenAIHTTPError(400, "request body must be a JSON object")
        model = _validate_model(body, backend)
        _validate_completion_options(body)
        max_tokens = _validate_max_tokens(body, backend)
        options = _generation_options(body, backend)
        stop_strings = _stop_strings(body)
        stop_token_ids = _stop_token_ids(body)
        prompts = normalize_completion_prompts(body, backend)
        for prompt in prompts:
            _check_context(backend, prompt, max_tokens)
        stream = bool(body.get("stream", False))
        include_usage = _include_stream_usage(body)
        echo = bool(body.get("echo", False))
        created = _now()
        rid = _make_id("cmpl")

        if stream:
            if len(prompts) != 1:
                raise OpenAIHTTPError(
                    400,
                    "streaming completions support only one prompt in this stage",
                    param="prompt",
                )
            prompt = prompts[0]
            session_id = _make_id("openai")

            def events():
                generated_text = ""
                if echo and prompt:
                    chunk = {
                        "id": rid,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{"text": prompt, "index": 0, "logprobs": None,
                                     "finish_reason": None}],
                    }
                    if include_usage:
                        chunk["usage"] = None
                    yield _json_sse(chunk)
                try:
                    for piece in backend.stream(prompt, max_tokens, session_id, options=options):
                        if not piece:
                            continue
                        visible, stopped = _split_on_stop(generated_text + piece, stop_strings)
                        piece = visible[len(generated_text):]
                        generated_text = visible
                        if not piece and not stopped:
                            continue
                        chunk = {
                            "id": rid,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [{"text": piece, "index": 0, "logprobs": None,
                                         "finish_reason": None}],
                        }
                        if include_usage:
                            chunk["usage"] = None
                        yield _json_sse(chunk)
                        if stopped:
                            break
                    final: dict = {
                        "id": rid,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{"text": "", "index": 0, "logprobs": None,
                                     "finish_reason": "stop"}],
                    }
                    if include_usage:
                        final["usage"] = None
                    yield _json_sse(final)
                    if include_usage:
                        yield _json_sse({
                            "id": rid,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": _usage(
                                backend, prompt,
                                GenerationResult(text=generated_text, token_ids=None),
                            ),
                        })
                except GeneratorExit:
                    raise
                except Exception as e:
                    yield _json_sse({
                        "error": {
                            "message": f"{type(e).__name__}: {e}",
                            "type": "server_error",
                            "param": None,
                            "code": "generation_failed",
                        }
                    })
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        for i, prompt in enumerate(prompts):
            result = backend.generate(
                prompt, max_tokens, f"{_make_id('openai')}-{i}", options=options
            )
            result = _apply_stops(result, backend, stop_strings, stop_token_ids)
            usage = _usage(backend, prompt, result)
            total_prompt_tokens += usage["prompt_tokens"]
            total_completion_tokens += usage["completion_tokens"]
            text = f"{prompt}{result.text}" if echo else result.text
            choices.append({
                "text": text,
                "index": i,
                "logprobs": None,
                "finish_reason": result.finish_reason,
            })
        return JSONResponse({
            "id": rid,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
        })

    async def embeddings(request: Request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise OpenAIHTTPError(400, "request body must be valid JSON")
        if not isinstance(body, dict):
            raise OpenAIHTTPError(400, "request body must be a JSON object")
        model = _validate_model(body, backend)
        if not getattr(backend, "supports_embeddings", False):
            raise OpenAIHTTPError(
                400,
                "this DRIFT mode cannot produce embeddings because the head does not hold hidden states",
                code="unsupported_embeddings",
            )
        encoding_format = body.get("encoding_format", "float")
        if body.get("dimensions") is not None:
            raise OpenAIHTTPError(
                400, "`dimensions` is not supported by this embedding endpoint yet",
                param="dimensions", code="unsupported_parameter"
            )
        inputs = normalize_embedding_inputs(body, backend)
        data = []
        total_tokens = 0
        for i, prompt in enumerate(inputs):
            _check_context(backend, prompt, 0)
            total_tokens += backend.count_tokens(prompt)
            vector = backend.embed(prompt, f"{_make_id('openai-emb')}-{i}")
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": _format_embedding(vector, encoding_format),
            })
        return JSONResponse({
            "object": "list",
            "model": model,
            "data": data,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        })

    async def tokenize(request: Request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise OpenAIHTTPError(400, "request body must be valid JSON")
        if not isinstance(body, dict):
            raise OpenAIHTTPError(400, "request body must be a JSON object")
        if "messages" in body:
            prompt: Prompt = normalize_chat_messages(body["messages"])
        else:
            content = body.get("content", body.get("prompt", ""))
            if not isinstance(content, str):
                raise OpenAIHTTPError(
                    400, "`content` must be a string when `messages` is omitted",
                    param="content",
                )
            prompt = content
        tokens = backend.encode_tokens(prompt)
        return JSONResponse({"tokens": tokens, "count": len(tokens)})

    async def detokenize(request: Request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise OpenAIHTTPError(400, "request body must be valid JSON")
        if not isinstance(body, dict):
            raise OpenAIHTTPError(400, "request body must be a JSON object")
        tokens = body.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(x, int) for x in tokens):
            raise OpenAIHTTPError(400, "`tokens` must be an integer array", param="tokens")
        return JSONResponse({"content": backend.decode_tokens([int(x) for x in tokens])})

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/completions", completions, methods=["POST"]),
        Route("/v1/embeddings", embeddings, methods=["POST"]),
        Route("/tokenize", tokenize, methods=["POST"]),
        Route("/detokenize", detokenize, methods=["POST"]),
        Route("/v1/tokenize", tokenize, methods=["POST"]),
        Route("/v1/detokenize", detokenize, methods=["POST"]),
    ]
    app = Starlette(routes=routes, exception_handlers={OpenAIHTTPError: openai_error_handler})
    app.add_middleware(RequestIDMiddleware)
    return app


def _parse_serve_nodes(spec: str) -> list[dict]:
    from .run import _parse_nodes

    return _parse_nodes(spec)


def build_backend_from_args(args) -> DriftBackend:
    from .common import load_config, pick_device
    from .run import _expand_members, _select_endpoints, build_over_nodes

    cfg = load_config(args.config)
    model_id = args.model or cfg["model_id"]
    dtype = cfg.get("dtype", "float16")
    head_device = pick_device(cfg.get("device"))
    n_new = args.max_new_tokens or cfg["generation"]["max_new_tokens"]

    endpoints = _select_endpoints(args, cfg)
    if args.expand:
        endpoints = _expand_members(endpoints)
    print(f"[serve] {len(endpoints)} node(s); splitting {model_id} ...", flush=True)
    orch, plan = build_over_nodes(model_id, dtype, head_device, endpoints,
                                  chain=args.chain, thin=args.thin, int8=args.int8)
    for s in plan:
        print(f"[serve] node {s['host']}:{s['port']} layers "
              f"[{s['start']}:{s['end']}) device={s['device']}", flush=True)
    return DriftBackend(orch, args.served_model_name or model_id, default_max_tokens=n_new)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="drift serve",
        description="serve DRIFT through an OpenAI-compatible HTTP API")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--nodes", help="comma-separated host:port of running `drift node`s")
    ap.add_argument("--no-discover", action="store_true", help="skip LAN auto-discovery")
    ap.add_argument("--discover-timeout", type=float, default=3.0)
    ap.add_argument("--model", help="override model_id")
    ap.add_argument("--served-model-name",
                    help="model id exposed over /v1/models (default: --model/config model_id)")
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--chain", action="store_true",
                    help="peer-to-peer chain: nodes stream to each other, not through the head")
    ap.add_argument("--thin", action="store_true",
                    help="zero-weight head: embed+lm_head move to the edge nodes (implies --chain)")
    ap.add_argument("--int8", action="store_true",
                    help="send the hidden state as int8 (half the wire bytes; lossy, relaxed gate)")
    ap.add_argument("--expand", action="store_true",
                    help="treat --nodes as seeds and split across the discovered membership")
    args = ap.parse_args(argv)

    backend = build_backend_from_args(args)
    app = create_app(backend)
    try:
        import uvicorn
    except ImportError as e:
        raise RuntimeError("drift serve requires uvicorn") from e
    print(f"[serve] OpenAI-compatible API listening on http://{args.host}:{args.port}/v1",
          flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
