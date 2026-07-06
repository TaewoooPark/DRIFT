import json
import base64
import struct

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

    def generate(self, prompt, max_tokens: int, session_id: str,
                 options: dict | None = None) -> GenerationResult:
        self.prompts.append(prompt)
        self.sessions.append(session_id)
        self.options.append(options or {})
        return GenerationResult(text="hello from drift", token_ids=[1, 2, 3])

    def stream(self, prompt, max_tokens: int, session_id: str,
               options: dict | None = None):
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


def test_rejects_json_response_format_until_supported():
    _, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "response_format": {"type": "json_object"},
    })

    assert res.status_code == 400
    assert res.json()["error"]["param"] == "response_format"


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
