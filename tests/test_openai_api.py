import json
import base64
import asyncio
import struct
import threading

from starlette.testclient import TestClient

from drift.openai_api import GenerationResult, create_app


class FakeBackend:
    model_id = "drift-test"
    default_max_tokens = 8
    context_length = 64

    def __init__(self):
        self.prompts = []
        self.sessions = []
        self.options = []
        self.supports_sampling = True
        self.supports_embeddings = True
        self.lock = threading.Lock()

    def generate(self, prompt, max_tokens: int, session_id: str,
                 options: dict | None = None) -> GenerationResult:
        with self.lock:
            self.prompts.append(prompt)
            self.sessions.append(session_id)
            self.options.append(options or {})
        return GenerationResult(
            text="hello from drift",
            token_ids=[1, 2, 3],
            logprobs=[
                {
                    "token_id": 1,
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"token_id": 1, "logprob": -0.1},
                        {"token_id": 9, "logprob": -2.0},
                    ],
                },
                {"token_id": 2, "logprob": -0.2, "top_logprobs": []},
                {"token_id": 3, "logprob": -0.3, "top_logprobs": []},
            ],
        )

    def stream(self, prompt, max_tokens: int, session_id: str,
               options: dict | None = None):
        with self.lock:
            self.prompts.append(prompt)
            self.sessions.append(session_id)
            self.options.append(options or {})
        yield "hello"
        yield " from"
        yield " drift"

    def count_tokens(self, text) -> int:
        if isinstance(text, list):
            text = " ".join(str(m.get("content") or "") for m in text)
        return len([p for p in text.split() if p])

    def decode_tokens(self, token_ids: list[int]) -> str:
        return " ".join(f"tok{x}" for x in token_ids)

    def encode_tokens(self, prompt) -> list[int]:
        if isinstance(prompt, list):
            text = " ".join(str(m.get("content") or "") for m in prompt)
        else:
            text = str(prompt)
        return [len(part) for part in text.split()]

    def embed(self, prompt, session_id: str) -> list[float]:
        with self.lock:
            self.prompts.append(prompt)
            self.sessions.append(session_id)
        return [0.25, 0.5, 0.75]


def client():
    backend = FakeBackend()
    return backend, TestClient(create_app(backend))


def sse_json_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        events.append(json.loads(line.removeprefix("data: ")))
    return events


def test_models_endpoint():
    _, c = client()
    res = c.get("/v1/models")

    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "drift-test"
    assert body["data"][0]["capabilities"] == {
        "chat": True,
        "completion": True,
        "embedding": True,
    }


def test_ready_endpoint_reports_capabilities_without_auth():
    backend = FakeBackend()
    c = TestClient(create_app(backend, api_keys=["secret"]))
    res = c.get("/ready")

    assert res.status_code == 200
    assert res.json()["capabilities"]["sampling"] is True


def test_chat_completion_non_streaming():
    backend, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [
            {"role": "developer", "content": "Be brief"},
            {"role": "user", "content": [{"type": "text", "text": "Say hello"}]},
        ],
        "max_tokens": 4,
        "response_format": {"type": "text"},
    })

    assert res.status_code == 200
    assert res.headers["x-request-id"].startswith("req-")
    body = res.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "drift-test"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "hello from drift",
    }
    assert body["usage"]["completion_tokens"] == 3
    assert backend.prompts[0] == [
        {"role": "system", "content": "Be brief"},
        {"role": "user", "content": "Say hello"},
    ]


def test_chat_accepts_all_text_roles_and_tool_followups():
    backend, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [
            {"role": "system", "content": "System rule"},
            {"role": "developer", "content": "Developer rule"},
            {"role": "user", "content": [{"type": "text", "text": "Question"}]},
            {
                "role": "assistant",
                "content": "I will call a tool",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "Tool answer"},
        ],
    })

    assert res.status_code == 200
    assert backend.prompts[0] == [
        {"role": "system", "content": "System rule"},
        {"role": "system", "content": "Developer rule"},
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "content": (
                "I will call a tool\n"
                "Tool calls: [{\"id\": \"call-1\", \"type\": \"function\", "
                "\"function\": {\"name\": \"lookup\", \"arguments\": "
                "\"{\\\"q\\\":\\\"x\\\"}\"}}]"
            ),
        },
        {"role": "tool", "content": "Tool answer"},
    ]


def test_chat_completion_streaming():
    _, c = client()
    with c.stream("POST", "/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Stream"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }) as res:
        text = res.read().decode()

    assert res.status_code == 200
    assert "chat.completion.chunk" in text
    assert '"role":"assistant"' in text
    assert '"content":"hello"' in text
    assert '"usage"' in text
    assert "data: [DONE]" in text
    events = sse_json_events(text)
    assert events[-2]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["choices"] == []
    assert events[-1]["usage"]["completion_tokens"] > 0


def test_rejects_unknown_model_with_openai_error_shape():
    _, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "missing-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }, headers={"x-request-id": "req-test"})

    assert res.status_code == 404
    assert res.headers["x-request-id"] == "req-test"
    body = res.json()
    assert body["error"]["code"] == "model_not_found"


def test_malformed_json_and_unknown_fields_are_openai_shaped():
    _, c = client()
    bad_json = c.post(
        "/v1/chat/completions",
        content="{not-json",
        headers={"content-type": "application/json"},
    )
    unknown = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "unknown_parameter": True,
    })

    assert bad_json.status_code == 400
    assert bad_json.json()["error"]["message"] == "request body must be valid JSON"
    assert unknown.status_code == 400
    assert unknown.json()["error"]["code"] == "unsupported_parameter"


def test_passes_sampling_options_to_backend():
    backend, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.05,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "repetition_penalty": 1.1,
        "seed": 123,
    })

    assert res.status_code == 200
    assert backend.options[0] == {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.05,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "repetition_penalty": 1.1,
        "seed": 123,
    }


def test_rejects_sampling_when_backend_cannot_sample():
    backend = FakeBackend()
    backend.supports_sampling = False
    c = TestClient(create_app(backend))
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
    })

    assert res.status_code == 400
    assert res.json()["error"]["code"] == "unsupported_sampling"


def test_rejects_multimodal_content_explicitly():
    _, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}],
        }],
    })

    assert res.status_code == 400
    assert res.json()["error"]["code"] == "unsupported_multimodal_content"


def test_json_object_response_format_returns_valid_json_text():
    _, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "response_format": {"type": "json_object"},
    })

    assert res.status_code == 200
    body = res.json()
    assert json.loads(body["choices"][0]["message"]["content"]) == {
        "response": "hello from drift"
    }


def test_json_schema_response_format_fills_required_fields():
    _, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "ok": {"type": "boolean"},
                    },
                    "required": ["answer", "ok"],
                    "additionalProperties": False,
                },
            },
        },
    })

    assert res.status_code == 200
    content = json.loads(res.json()["choices"][0]["message"]["content"])
    assert content == {"answer": "hello from drift", "ok": False}


def test_required_tool_choice_returns_tool_calls():
    _, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Use a tool"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }],
        "tool_choice": "required",
    })

    assert res.status_code == 200
    choice = res.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "query": ""
    }


def test_auto_tool_choice_promotes_model_emitted_tool_call_json():
    class ToolJSONBackend(FakeBackend):
        def generate(self, prompt, max_tokens: int, session_id: str,
                     options: dict | None = None) -> GenerationResult:
            with self.lock:
                self.prompts.append(prompt)
                self.sessions.append(session_id)
                self.options.append(options or {})
            return GenerationResult(
                text='{"tool_call":{"name":"lookup","arguments":{"query":"drift"}}}',
                token_ids=[1],
            )

    backend = ToolJSONBackend()
    c = TestClient(create_app(backend))
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Use a tool"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }],
        "tool_choice": "auto",
    })

    assert res.status_code == 200
    choice = res.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": "{\"query\":\"drift\"}",
    }


def test_legacy_functions_and_function_call_are_accepted():
    _, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Use a legacy function"}],
        "functions": [{
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }],
        "function_call": {"name": "lookup"},
    })

    assert res.status_code == 200
    choice = res.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup"


def test_finish_reason_length_is_propagated():
    class LengthBackend(FakeBackend):
        def generate(self, prompt, max_tokens: int, session_id: str,
                     options: dict | None = None) -> GenerationResult:
            with self.lock:
                self.prompts.append(prompt)
                self.sessions.append(session_id)
                self.options.append(options or {})
            return GenerationResult(
                text="tok tok",
                token_ids=[1, 2],
                finish_reason="length",
            )

    c = TestClient(create_app(LengthBackend()))
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 2,
    })

    assert res.status_code == 200
    assert res.json()["choices"][0]["finish_reason"] == "length"


def test_chat_completion_n_and_logprobs():
    backend, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Two answers"}],
        "n": 2,
        "logprobs": True,
        "top_logprobs": 2,
        "seed": 10,
    })

    assert res.status_code == 200
    body = res.json()
    assert [ch["index"] for ch in body["choices"]] == [0, 1]
    assert len(body["choices"]) == 2
    assert body["choices"][0]["logprobs"]["content"][0]["token"] == "tok1"
    assert body["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.1
    assert body["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0] == {
        "token": "tok1",
        "logprob": -0.1,
        "bytes": [116, 111, 107, 49],
    }
    assert body["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["token"] == "tok9"
    assert [opt["seed"] for opt in backend.options] == [10, 11]
    assert all(opt["_return_logprobs"] for opt in backend.options)


def test_chat_streaming_n_and_logprobs_uses_openai_chunk_shape():
    _, c = client()
    with c.stream("POST", "/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Stream two"}],
        "stream": True,
        "n": 2,
        "logprobs": True,
        "stream_options": {"include_usage": True},
    }) as res:
        text = res.read().decode()

    assert res.status_code == 200
    events = sse_json_events(text)
    indexes = [event["choices"][0]["index"] for event in events if event["choices"]]
    assert 0 in indexes and 1 in indexes
    assert any(
        event["choices"]
        and event["choices"][0].get("logprobs", {}).get("content")
        for event in events
    )
    assert events[-1]["choices"] == []
    assert events[-1]["usage"]["completion_tokens"] == 6


def test_chat_top_logprobs_requires_logprobs_true():
    _, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "top_logprobs": 1,
    })

    assert res.status_code == 400
    assert res.json()["error"]["param"] == "top_logprobs"


def test_completion_non_streaming_single_prompt():
    backend, c = client()
    res = c.post("/v1/completions", json={
        "model": "drift-test",
        "prompt": "Complete this",
        "max_tokens": 3,
    })

    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "hello from drift"
    assert body["choices"][0]["logprobs"] is None
    assert body["usage"]["completion_tokens"] == 3
    assert backend.prompts == ["Complete this"]


def test_completion_n_and_logprobs():
    backend, c = client()
    res = c.post("/v1/completions", json={
        "model": "drift-test",
        "prompt": "Complete this",
        "n": 2,
        "logprobs": 2,
        "seed": 5,
    })

    assert res.status_code == 200
    body = res.json()
    assert [ch["index"] for ch in body["choices"]] == [0, 1]
    assert body["choices"][0]["logprobs"]["tokens"] == ["tok1", "tok2", "tok3"]
    assert body["choices"][0]["logprobs"]["token_logprobs"] == [-0.1, -0.2, -0.3]
    assert body["choices"][0]["logprobs"]["top_logprobs"][0] == {
        "tok1": -0.1,
        "tok9": -2.0,
    }
    assert body["usage"]["completion_tokens"] == 6
    assert backend.prompts == ["Complete this", "Complete this"]
    assert [opt["seed"] for opt in backend.options] == [5, 6]
    assert all(opt["_return_logprobs"] for opt in backend.options)


def test_completion_non_streaming_prompt_array_and_token_ids():
    backend, c = client()
    res = c.post("/v1/completions", json={
        "model": "drift-test",
        "prompt": ["first", [10, 11]],
        "echo": True,
    })

    assert res.status_code == 200
    body = res.json()
    assert len(body["choices"]) == 2
    assert body["choices"][0]["text"] == "firsthello from drift"
    assert body["choices"][1]["text"] == "tok10 tok11hello from drift"
    assert backend.prompts == ["first", "tok10 tok11"]


def test_completion_streaming_single_prompt():
    _, c = client()
    with c.stream("POST", "/v1/completions", json={
        "model": "drift-test",
        "prompt": "Stream completion",
        "stream": True,
        "stream_options": {"include_usage": True},
    }) as res:
        text = res.read().decode()

    assert res.status_code == 200
    assert "text_completion" in text
    assert '"text":"hello"' in text
    assert '"usage"' in text
    assert "data: [DONE]" in text
    events = sse_json_events(text)
    assert events[-2]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["choices"] == []
    assert events[-1]["usage"]["completion_tokens"] > 0


def test_completion_streaming_rejects_prompt_arrays_for_now():
    _, c = client()
    res = c.post("/v1/completions", json={
        "model": "drift-test",
        "prompt": ["a", "b"],
        "stream": True,
    })

    assert res.status_code == 400
    assert res.json()["error"]["param"] == "prompt"


def test_non_streaming_applies_stop_strings():
    _, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "stop": " from",
    })

    assert res.status_code == 200
    body = res.json()
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_non_streaming_applies_stop_token_ids():
    _, c = client()
    res = c.post("/v1/completions", json={
        "model": "drift-test",
        "prompt": "Hello",
        "stop_token_ids": [2],
    })

    assert res.status_code == 200
    body = res.json()
    assert body["choices"][0]["text"] == "tok1"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 1


def test_streaming_stops_on_stop_string_boundary():
    _, c = client()
    with c.stream("POST", "/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "stop": " from",
    }) as res:
        text = res.read().decode()

    assert res.status_code == 200
    assert '"content":"hello"' in text
    assert '"content":" from"' not in text


def test_context_length_error_shape():
    backend = FakeBackend()
    backend.context_length = 2
    c = TestClient(create_app(backend))
    res = c.post("/v1/completions", json={
        "model": "drift-test",
        "prompt": "one two",
        "max_tokens": 1,
    })

    assert res.status_code == 400
    assert res.json()["error"]["code"] == "context_length_exceeded"


def test_tokenize_and_detokenize_helpers():
    _, c = client()
    tok = c.post("/tokenize", json={"content": "one three"})
    detok = c.post("/v1/detokenize", json={"tokens": [3, 5]})

    assert tok.status_code == 200
    assert tok.json() == {"tokens": [3, 5], "count": 2}
    assert detok.status_code == 200
    assert detok.json() == {"content": "tok3 tok5"}


def test_embeddings_float_response():
    backend, c = client()
    res = c.post("/v1/embeddings", json={
        "model": "drift-test",
        "input": ["alpha beta", [1, 2]],
    })

    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "list"
    assert body["data"][0] == {
        "object": "embedding",
        "index": 0,
        "embedding": [0.25, 0.5, 0.75],
    }
    assert body["data"][1]["embedding"] == [0.25, 0.5, 0.75]
    assert body["usage"]["prompt_tokens"] > 0
    assert backend.prompts == ["alpha beta", "tok1 tok2"]


def test_embeddings_string_array_token_ids_and_empty_input_errors():
    backend, c = client()
    string_res = c.post("/v1/embeddings", json={
        "model": "drift-test",
        "input": "solo",
    })
    token_res = c.post("/v1/embeddings", json={
        "model": "drift-test",
        "input": [1, 2, 3],
    })
    empty_array = c.post("/v1/embeddings", json={
        "model": "drift-test",
        "input": [],
    })
    empty_string = c.post("/v1/embeddings", json={
        "model": "drift-test",
        "input": "",
    })

    assert string_res.status_code == 200
    assert string_res.json()["data"][0]["index"] == 0
    assert token_res.status_code == 200
    assert backend.prompts[-2:] == ["solo", "tok1 tok2 tok3"]
    assert empty_array.status_code == 400
    assert empty_array.json()["error"]["param"] == "input"
    assert empty_string.status_code == 400
    assert empty_string.json()["error"]["param"] == "input"


def test_embeddings_context_overflow_and_bad_encoding_errors():
    backend = FakeBackend()
    backend.context_length = 1
    c = TestClient(create_app(backend))
    too_long = c.post("/v1/embeddings", json={
        "model": "drift-test",
        "input": "one two",
    })
    bad_encoding = c.post("/v1/embeddings", json={
        "model": "drift-test",
        "input": "one",
        "encoding_format": "hex",
    })

    assert too_long.status_code == 400
    assert too_long.json()["error"]["code"] == "context_length_exceeded"
    assert bad_encoding.status_code == 400
    assert bad_encoding.json()["error"]["param"] == "encoding_format"


def test_embeddings_base64_response():
    _, c = client()
    res = c.post("/v1/embeddings", json={
        "model": "drift-test",
        "input": "alpha",
        "encoding_format": "base64",
    })

    assert res.status_code == 200
    encoded = res.json()["data"][0]["embedding"]
    assert base64.b64decode(encoded) == struct.pack("<3f", 0.25, 0.5, 0.75)


def test_embeddings_unsupported_mode_error():
    backend = FakeBackend()
    backend.supports_embeddings = False
    c = TestClient(create_app(backend))
    res = c.post("/v1/embeddings", json={
        "model": "drift-test",
        "input": "alpha",
    })

    assert res.status_code == 400
    assert res.json()["error"]["code"] == "unsupported_embeddings"


def test_api_key_auth_protects_openai_routes():
    backend = FakeBackend()
    c = TestClient(create_app(backend, api_keys=["secret"]))

    missing = c.get("/v1/models", headers={"x-request-id": "req-auth"})
    bearer = c.get("/v1/models", headers={"authorization": "Bearer secret"})
    header = c.get("/v1/models", headers={"x-api-key": "secret"})

    assert missing.status_code == 401
    assert missing.headers["x-request-id"] == "req-auth"
    assert missing.json()["error"]["code"] == "invalid_api_key"
    assert bearer.status_code == 200
    assert header.status_code == 200


def test_cors_preflight_when_enabled():
    backend = FakeBackend()
    c = TestClient(create_app(backend, cors_origins=["https://example.com"]))
    res = c.options("/v1/models", headers={
        "origin": "https://example.com",
        "access-control-request-method": "GET",
    })

    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "https://example.com"


def test_concurrent_requests_get_isolated_sessions():
    backend = FakeBackend()
    app = create_app(backend, max_concurrent_requests=4)

    async def run():
        import httpx

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await asyncio.gather(*[
                c.post("/v1/chat/completions", json={
                    "model": "drift-test",
                    "messages": [{"role": "user", "content": f"hello {i}"}],
                    "seed": i,
                })
                for i in range(10)
            ])

    results = asyncio.run(run())

    assert all(res.status_code == 200 for res in results)
    assert len(backend.sessions) == 10
    assert len(set(backend.sessions)) == 10
    assert sorted(opt["seed"] for opt in backend.options) == list(range(10))
