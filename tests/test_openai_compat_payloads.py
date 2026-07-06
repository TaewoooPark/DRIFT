from starlette.testclient import TestClient

from drift.openai_api import GenerationResult, create_app


class CompatBackend:
    model_id = "drift-test"
    default_max_tokens = 16
    context_length = 128
    supports_sampling = True
    supports_embeddings = True

    def __init__(self):
        self.options = []

    def generate(self, prompt, max_tokens: int, session_id: str,
                 options: dict | None = None) -> GenerationResult:
        self.options.append(options or {})
        return GenerationResult(text="compat answer END", token_ids=[4, 5, 6])

    def stream(self, prompt, max_tokens: int, session_id: str,
               options: dict | None = None):
        self.options.append(options or {})
        yield "compat"
        yield " answer"
        yield " END"

    def count_tokens(self, prompt) -> int:
        if isinstance(prompt, list):
            prompt = " ".join(str(m.get("content") or "") for m in prompt)
        return len(str(prompt).split())

    def decode_tokens(self, token_ids: list[int]) -> str:
        return " ".join(f"tok{x}" for x in token_ids)

    def encode_tokens(self, prompt) -> list[int]:
        if isinstance(prompt, list):
            prompt = " ".join(str(m.get("content") or "") for m in prompt)
        return [len(x) for x in str(prompt).split()]

    def embed(self, prompt, session_id: str) -> list[float]:
        return [0.1, 0.2]


def client(api_key: str | None = None):
    backend = CompatBackend()
    keys = [api_key] if api_key else None
    return backend, TestClient(create_app(backend, api_keys=keys))


def test_openai_sdk_style_chat_payload_snapshot():
    _, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [
            {"role": "system", "content": "Be direct."},
            {"role": "user", "content": "Hello"},
        ],
        "temperature": 0,
        "max_tokens": 5,
    })

    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"id", "object", "created", "model", "choices", "usage"}
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"


def test_langchain_style_stop_payload_truncates_text():
    _, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "stop": [" END"],
    })

    assert res.status_code == 200
    assert res.json()["choices"][0]["message"]["content"] == "compat answer"


def test_litellm_vllm_style_sampling_payload_is_accepted():
    backend, c = client()
    res = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.8,
        "top_p": 0.92,
        "top_k": 20,
        "min_p": 0.02,
        "repetition_penalty": 1.05,
        "seed": 11,
        "user": "compat-suite",
    })

    assert res.status_code == 200
    assert backend.options[0]["top_k"] == 20
    assert backend.options[0]["min_p"] == 0.02


def test_llama_cpp_style_helpers_and_legacy_completion():
    _, c = client()
    completion = c.post("/v1/completions", json={
        "model": "drift-test",
        "prompt": "Write:",
        "echo": True,
    })
    tokenize = c.post("/tokenize", json={"content": "hello world"})
    detokenize = c.post("/detokenize", json={"tokens": [5, 5]})

    assert completion.status_code == 200
    assert completion.json()["choices"][0]["text"].startswith("Write:")
    assert tokenize.json()["tokens"] == [5, 5]
    assert detokenize.json()["content"] == "tok5 tok5"


def test_embedding_payload_snapshot():
    _, c = client()
    res = c.post("/v1/embeddings", json={
        "model": "drift-test",
        "input": ["hello", "world"],
    })

    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"object", "model", "data", "usage"}
    assert body["data"][0]["object"] == "embedding"


def test_responses_api_minimal_text_payload():
    _, c = client()
    res = c.post("/v1/responses", json={
        "model": "drift-test",
        "input": "Hello",
        "max_output_tokens": 4,
    })

    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["type"] == "output_text"
    assert body["output_text"] == "compat answer END"
    assert body["usage"]["total_tokens"] > 0


def test_chat_input_tokens_endpoint():
    _, c = client()
    res = c.post("/v1/chat/completions/input_tokens", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "hello world"}],
    })

    assert res.status_code == 200
    assert res.json()["object"] == "input_tokens"
    assert res.json()["input_tokens"] == 2


def test_metrics_endpoint_prometheus_shape():
    _, c = client()
    res = c.get("/metrics")

    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]
    assert "drift_openai_requests_total" in res.text
    assert 'drift_openai_model_info{model="drift-test"} 1' in res.text


def test_negative_compat_cases_are_openai_shaped():
    _, c = client(api_key="secret")
    bad_auth = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
    })
    tool_call = c.post("/v1/chat/completions", json={
        "model": "drift-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [{"type": "function", "function": {"name": "x"}}],
    }, headers={"authorization": "Bearer secret"})
    bad_embedding = c.post("/v1/embeddings", json={
        "model": "drift-test",
        "input": "hello",
        "dimensions": 8,
    }, headers={"authorization": "Bearer secret"})
    bad_response = c.post("/v1/responses", json={
        "model": "drift-test",
        "input": "hello",
        "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
    }, headers={"authorization": "Bearer secret"})

    assert bad_auth.status_code == 401
    assert bad_auth.json()["error"]["type"] == "authentication_error"
    assert tool_call.status_code == 400
    assert tool_call.json()["error"]["code"] == "unsupported_parameter"
    assert bad_embedding.status_code == 400
    assert bad_embedding.json()["error"]["param"] == "dimensions"
    assert bad_response.status_code == 400
    assert bad_response.json()["error"]["param"] == "response_format"
