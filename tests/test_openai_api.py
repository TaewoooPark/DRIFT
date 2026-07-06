from starlette.testclient import TestClient

from drift.openai_api import GenerationResult, create_app


class FakeBackend:
    model_id = "drift-test"
    default_max_tokens = 8

    def __init__(self):
        self.prompts = []
        self.sessions = []

    def generate(self, prompt, max_tokens: int, session_id: str) -> GenerationResult:
        self.prompts.append(prompt)
        self.sessions.append(session_id)
        return GenerationResult(text="hello from drift", token_ids=[1, 2, 3])

    def stream(self, prompt, max_tokens: int, session_id: str):
        self.prompts.append(prompt)
        self.sessions.append(session_id)
        yield "hello"
        yield " from"
        yield " drift"

    def count_tokens(self, text) -> int:
        if isinstance(text, list):
            text = " ".join(str(m.get("content") or "") for m in text)
        return len([p for p in text.split() if p])


def client():
    backend = FakeBackend()
    return backend, TestClient(create_app(backend))


def test_models_endpoint():
    _, c = client()
    res = c.get("/v1/models")

    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "drift-test"


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


def test_rejects_sampling_until_generation_controls_land():
    _, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
    })

    assert res.status_code == 400
    body = res.json()
    assert body["error"]["code"] == "unsupported_sampling"


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
