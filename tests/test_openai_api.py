from starlette.testclient import TestClient

from drift.openai_api import GenerationResult, create_app


class FakeBackend:
    model_id = "drift-test"
    default_max_tokens = 8

    def __init__(self):
        self.prompts = []
        self.sessions = []

    def generate(self, prompt: str, max_tokens: int, session_id: str) -> GenerationResult:
        self.prompts.append(prompt)
        self.sessions.append(session_id)
        return GenerationResult(text="hello from drift", token_ids=[1, 2, 3])

    def stream(self, prompt: str, max_tokens: int, session_id: str):
        self.prompts.append(prompt)
        self.sessions.append(session_id)
        yield "hello"
        yield " from"
        yield " drift"

    def count_tokens(self, text: str) -> int:
        return len([p for p in text.split() if p])


def client():
    return TestClient(create_app(FakeBackend()))


def test_models_endpoint():
    res = client().get("/v1/models")

    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "drift-test"


def test_chat_completion_non_streaming():
    c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 4,
    })

    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "drift-test"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "hello from drift",
    }
    assert body["usage"]["completion_tokens"] == 3


def test_chat_completion_streaming():
    c = client()
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
    res = client().post("/v1/chat/completions", json={
        "model": "missing-model",
        "messages": [{"role": "user", "content": "Hello"}],
    })

    assert res.status_code == 404
    body = res.json()
    assert body["error"]["code"] == "model_not_found"


def test_rejects_sampling_until_generation_controls_land():
    res = client().post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
    })

    assert res.status_code == 400
    body = res.json()
    assert body["error"]["code"] == "unsupported_sampling"
